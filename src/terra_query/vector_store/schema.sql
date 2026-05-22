-- chip_embeddings: one row per (model_id, bands, chip_id).
-- chip_location_id is the per-cycle-shared grid id (e.g. "r034_c028");
-- the loader strips the source/year prefix from chip_id.
-- footprint + center are both stored in EPSG:26916 (the working CRS).

CREATE TABLE IF NOT EXISTS chip_embeddings (
    model_id          TEXT          NOT NULL,
    bands             TEXT          NOT NULL,
    chip_id           TEXT          NOT NULL,
    chip_location_id  TEXT          NOT NULL,
    source_cycle      TEXT          NOT NULL,
    inside_aoi        BOOLEAN       NOT NULL,
    embedding         vector(768)   NOT NULL,
    footprint         geometry(Polygon, 26916) NOT NULL,
    center_26916      geometry(Point,   26916) NOT NULL,
    PRIMARY KEY (model_id, bands, chip_id)
);

CREATE INDEX IF NOT EXISTS chip_embeddings_hnsw
    ON chip_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chip_embeddings_footprint_gist
    ON chip_embeddings USING gist (footprint);

CREATE INDEX IF NOT EXISTS chip_embeddings_center_gist
    ON chip_embeddings USING gist (center_26916);

CREATE INDEX IF NOT EXISTS chip_embeddings_location_idx
    ON chip_embeddings (model_id, bands, chip_location_id);
