"""Vector store: Postgres + PostGIS + pgvector.

The production storage layer for chip embeddings. One row per
(model_id, bands, chip_id). HNSW index on the vector, GIST indexes
on footprint + center geometries. Same schema runs locally (docker
compose) and on a hosted Postgres; only the DSN changes.
"""
