
CREATE OR REPLACE TABLE gba_person AS
SELECT * FROM read_parquet('data/lake/silver/GBA_PERSON/*.parquet');

CREATE TABLE person_static AS
SELECT PERSON_ID, NOTE.GBAGESLACHT AS GENDER,
TRY_CAST(concat(NOTE.GBAGEBOORTEJAAR::INTEGER, '-', NOTE.GBAGEBOORTEMAAND::VARCHAR, '-01') AS DATE) AS BIRTHDATE
FROM gba_person
WHERE BIRTHDATE IS NOT NULL AND GENDER != '-';

COPY (FROM person_static) TO 'data/lake/silver/person_static.parquet' (FORMAT parquet);
