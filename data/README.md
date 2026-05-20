# data/

Every artifact this project produces or consumes lives under one of
four top-level buckets. The buckets are organized by **provenance**
(where the bits came from), not by file format.

| bucket | what's here | regenerable? | git |
|---|---|---|---|
| `human_authored/` | hand-curated, project-defining content. Written by a person, never recreated by code. | no | tracked (small) |
| `source_downloads/` | content pulled verbatim from external sources. | yes - rerun the matching ingest CLI | large blobs ignored; manifests / indexes tracked |
| `pipeline_outputs/` | content produced by this project's code from `human_authored/` + `source_downloads/`. | yes - rerun the pipeline | mostly ignored |
| `verification/` | eyeball-check artifacts and per-run logs that exist for confirming the pipeline is doing the right thing. | yes - regenerable on demand | small enough to track |

The names describe provenance, not file type. A GeoJSON belongs in
`human_authored/` if a person authored it, in `pipeline_outputs/` if a
script generated it, in `verification/` if it's only there for
inspection. The same rule applies to any file format the project
adopts later.

If you're about to write a file, ask: where did this come from? That
tells you the bucket. Specific subdirectory layout under each bucket
is owned by whichever pipeline stage produces it, not by this
document.

The canonical source of every path the code uses is
`src/terra_query/core/paths.py`. Nothing else should hardcode a path
under `data/`.
