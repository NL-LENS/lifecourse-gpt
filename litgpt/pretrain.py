# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import pprint
import time
import warnings
from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

from litgpt import Tokenizer
from litgpt.args import EvalArgs, LogArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.constants import _TORCH_EQUAL_2_7, _TORCH_EQUAL_2_8
from litgpt.data import DataModule, TinyLlama
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP, model_class
from litgpt.parser_config import save_hyperparameters
from litgpt.types import LoggerChoice
from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    copy_config_files,
    extend_checkpoint_dir,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
)


def setup(
    model_name: str,
    model_config: Config | None = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Path | None = None,
    resume: bool | Literal["auto"] | Path = False,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    log: LogArgs = LogArgs(),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: LoggerChoice = "tensorboard",
    seed: int = 42,
    prepare_data_only: bool = False,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``. Overrides the `model_name` if specified.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.

        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        prepare_data_only: Preprocess the dataset and exit, training nothing. Useful when
            preprocessing is slow, needs a different machine than training, or has to
            happen once before several runs share the result. Everything the data module
            needs is read from the same config, so the run that trains later builds the
            identical cache and simply finds it already there.
    """
    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    if model_config is None:
        # Support both model_name options: meta-llama/Meta-Llama-3-8B & Meta-Llama-3-8B
        try:
            model_config = Config.from_name(model_name)
        except ValueError:
            print(f"Model name {model_name} is not supported.\n")
            available_models = "\n".join(sorted(name_to_config))
            print(f"Available values:\n{available_models}")
            quit()

    hparams = capture_hparams()
    data = TinyLlama() if data is None else data

    config = Config.from_name(model_name) if model_config is None else model_config
    # A data module whose batches only one architecture can read says so. Checked up
    # front because the mismatch is otherwise silent: `--model_config.*` replaces the
    # named config wholesale, `model_class_name` included, and a plain GPT will happily
    # fit cross entropy to a batch laid out for a different likelihood.
    expected_class = getattr(data, "model_class_name", None)
    if expected_class is not None and config.model_class_name != expected_class:
        raise ValueError(
            f"{type(data).__name__} produces batches for model_class_name={expected_class!r}, but the model "
            f"config has {config.model_class_name!r}. `--model_config` replaces the config named by "
            f"`model_name` entirely, so set `--model_config.model_class_name {expected_class}` as well."
        )
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    if prepare_data_only:
        # Before the logger and Fabric: this writes a dataset, and has no business
        # creating run directories or claiming devices.
        data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=config.block_size)
        data.prepare_data()
        return

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"pretrain-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        log_args=asdict(log),
    )

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=[logger])

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb", "mlflow"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        seed=seed,
        initial_checkpoint_dir=initial_checkpoint_dir,
        resume=resume,
        config=config,
        data=data,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer=tokenizer,
        train=train,
        eval=eval,
        optimizer=optimizer,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Path | None,
    resume: bool | Literal["auto"] | Path,
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Path | None,
    tokenizer: Tokenizer | None,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: str | dict,
    num_nodes: int = 1,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = model_class(config)(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = torch.compile(model)
    model = fabric.setup(model)

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(optimizer, model.parameters(), **extra_kwargs)
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train, model.max_seq_length)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    state = {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()

    # work around PyTorch issue https://github.com/pytorch/pytorch/issues/152162
    # which does not like the lazy initialization to be called in dynamo.
    # TODO: Happens with PyTorch 2.7+
    if (
        (_TORCH_EQUAL_2_7 or _TORCH_EQUAL_2_8)
        and (model._forward_module.__class__.__name__ == "OptimizedModule")
        and (model._forward_module._orig_mod.__class__.__name__ == "FullyShardedDataParallel")
    ):
        from torch.distributed.fsdp._runtime_utils import _root_pre_forward

        _root_pre_forward(model._forward_module._orig_mod, model._forward_module._orig_mod, [], {})

    fit(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        state=state,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        train=train,
        eval=eval,
    )

    # Save final checkpoint
    save_checkpoint(fabric, state, tokenizer_dir, out_dir / "final" / "lit_model.pth")

    total_tokens = state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size

    # Print formatted output
    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Total tokens  : {total_tokens:,}")
    fabric.print(f"| - Training Time : {(time.perf_counter() - train_time):.2f} s")
    fabric.print(f"| - Tok/sec       : {total_tokens / train_time:.2f} tok/s")
    fabric.print("| " + "-" * 40)

    if fabric.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        fabric.print("| Memory Usage")
        fabric.print(f"| - Memory Used   : {memory_used:.2f} GB")
    fabric.print(separator)


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Path | None,
    train: TrainArgs,
    eval: EvalArgs,
    num_nodes: int = 1,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    if eval.initial_validation:
        val_losses = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
    else:
        fabric.print("Verifying settings ...")
        validate(fabric, model, val_dataloader, max_iters=2, verbose=False)  # sanity check
        val_losses = None

    throughput = ThroughputMonitor(fabric, window_size=5)

    arch = model_class(model.config)
    with torch.device("meta"):
        meta_model = arch(model.config)
        example = arch.example_batch(train.micro_batch_size, meta_model.max_seq_length, model.config)
        model_fwd = lambda: meta_model(**arch.model_inputs(example))  # noqa: F821
        model_loss = lambda y: arch.compute_loss(y, example, model.config)["loss"]  # noqa: F821
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, example

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices, num_nodes)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    # One meter per term `compute_loss` reports, created on first sight so that the loop
    # needs no knowledge of the architecture. Components such as Delphi's `loss_ce` and
    # `loss_dt` are tracked separately because they typically move on different scales.
    running_losses: dict[str, RunningMean] = {}
    fabric.barrier()
    total_t0 = time.perf_counter()

    warmup_iters = train.warmup_iters(devices, num_nodes, max_iters, train_dataloader)

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(optimizer.defaults["lr"], state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        batch = _unpack_batch(train_data, model.max_seq_length)

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            losses = forward_loss(model, batch)
            fabric.backward(losses["loss"] / train.gradient_accumulation_iters(devices, num_nodes))

        for name, value in losses.items():
            if name not in running_losses:
                running_losses[name] = RunningMean(
                    window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=False
                ).to(fabric.device)
            running_losses[name].update(value.detach())

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            # expensive device-to-host synchronization
            loss_values = {name: running.compute().item() for name, running in running_losses.items()}
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
            )
            metrics = {
                **loss_values,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * model.max_seq_length,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size),
                "learning_rate": lr,
            }
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {format_losses(loss_values)},"
                f" val: {format_losses(val_losses)} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_losses = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
            td = time.perf_counter() - t0

            fabric.print(
                f"iter {state['iter_num']}: val loss {format_losses(val_losses)}, val time: {td * 1000:.2f} ms"
            )
            metrics = {f"val_{name}": value for name, value in val_losses.items()}
            metrics["val_ppl"] = perplexity(val_losses)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            save_checkpoint(fabric, state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth")

    # Final validation
    if eval.final_validation:
        val_losses = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        metrics = {f"val_{name}": value for name, value in val_losses.items()}
        metrics["val_ppl"] = perplexity(val_losses)
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(f"Final evaluation | val loss: {format_losses(val_losses)} | val ppl: {metrics['val_ppl']:.3f}")


def _unpack_batch(batch: torch.Tensor | dict[str, torch.Tensor], max_seq_length: int) -> dict[str, torch.Tensor]:
    """Splits one dataloader batch into model inputs and loss targets.

    Text pretraining streams a flat `(B, T + 1)` block of packed tokens, so the causal
    shift happens here. A `DataModule` that emits dicts has already shifted its samples
    (the life-event modules must, since every token's target carries its own age), so
    they are only cut to `max_seq_length`: `(B, T)` tensors along the sequence axis and
    `(B, 1, T, T)` attention masks along both. `labels` is renamed to `targets`, the name
    `compute_loss` reads for text as well.
    """
    if not isinstance(batch, dict):
        return {
            "input_ids": batch[:, 0:max_seq_length].contiguous().long(),
            "targets": batch[:, 1 : (max_seq_length + 1)].contiguous().long(),
        }

    unpacked = {}
    for key, value in batch.items():
        value = value[..., :max_seq_length, :max_seq_length] if value.dim() == 4 else value[:, :max_seq_length]
        if not value.is_floating_point() and value.dtype != torch.bool:
            value = value.long()
        unpacked["targets" if key == "labels" else key] = value.contiguous()
    return unpacked


def forward_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Runs the model over one unpacked batch and returns every loss term to report.

    What goes into the model and which likelihood it is trained under are both decided by
    the architecture, see `GPT.model_inputs` and `GPT.compute_loss`. `loss` is always the
    quantity that gets backpropagated.
    """
    arch = model_class(model.config)
    logits = model(**arch.model_inputs(batch))
    return arch.compute_loss(logits, batch, model.config)


def format_losses(losses: dict[str, float] | None) -> str:
    """Renders the total, with the components in parentheses when there are any."""
    if losses is None:
        return "n/a"
    components = ", ".join(
        f"{name.removeprefix('loss_')} {value:.3f}" for name, value in losses.items() if name != "loss"
    )
    return f"{losses['loss']:.3f}" + (f" ({components})" if components else "")


def perplexity(losses: dict[str, float]) -> float:
    """Perplexity of the next-event distribution.

    Derived from `loss_ce` alone where it exists. The total also carries `loss_dt`,
    which is a density over a continuous waiting time rather than a distribution over
    the vocabulary, so exponentiating the sum would not be a perplexity of anything.
    """
    return math.exp(losses.get("loss_ce", losses["loss"]))


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int, verbose: bool = True
) -> dict[str, float]:
    fabric.barrier()
    if verbose:
        fabric.print("Validating ...")
    model.eval()

    losses = defaultdict(list)
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        for name, value in forward_loss(model, _unpack_batch(batch, model.max_seq_length)).items():
            losses[name].append(value)

    val_losses = {name: torch.stack(values).mean().item() for name, values in losses.items()}
    model.train()
    fabric.barrier()
    return val_losses


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs, block_size: int
) -> tuple[DataLoader, DataLoader]:
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    if train.max_steps is not None:
        warnings.warn(
            "`train.max_steps` is intended for profiling or debug runs only. "
            "For full pretraining runs, prefer `train.max_tokens` or `train.max_time`.",
            UserWarning,
        )
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))
