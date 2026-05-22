# eval set - Bond Falls block

This directory holds the **scorecard** for the Terra-query MVP.
Not the product output. The product output (unmapped feature
candidates on a map) emerges from the pipeline later; this list is
the ground truth used to test whether the model is finding the
things it should be finding.

Two files:

- `known_features.geojson` - hand-authored GeoJSON FeatureCollection
  of Point features, WGS84 (EPSG:4326).
- `README.md` - this file.

The AOI polygon lives in `../aoi/bond_falls_block.geojson`. Every
point in `known_features.geojson` falls inside that polygon.

## Why this exists

Without a hand-authored list of real features at known coordinates,
"is retrieval working?" has no answer and the demo can silently lie.
The eval set fixes that: when the model returns "top 10 matches for
'waterfall'", we check whether Bond Falls (which we know is at
(46.40967, -89.13265)) appears in those 10. If yes, the model can
find waterfalls. We then trust it generally, including for queries
the eval set does not directly label (e.g. "cabin").

## In-scope feature types

waterfalls/cascades, rapids, cliffs and outcrops, beaver/kettle
ponds, old cabins/camps/ruins, mines and prospect pits, old logging
grades / unmapped trails.

Explicit out-of-scope:

- sub-canopy features invisible in both aerial and the LiDAR render
- scenic overlooks / vistas (viewshed property, not appearance)

## File schema (`known_features.geojson`)

Top-level: `{"type": "FeatureCollection", "name": "bond_falls_known_features", "crs": "CRS84", "features": [...]}`.

Each feature is a Point in WGS84 with these `properties`:

| field | required | type | meaning |
|---|---|---|---|
| `id` | yes | string | stable kebab-case slug, e.g. `bond-falls` |
| `name` | yes | string | human label |
| `type` | yes | enum | feature kind, see enum below |
| `category` | yes | enum | `positive_in_scope`, `negative_mapped`, `out_of_scope` |
| `source` | yes | string | where the coordinate came from (e.g. `osm_way:615053222 (polygon centroid)`, `osm_node:1587709054`) |
| `findable_aerial` | yes | bool \| null | set against the project's own NAIP raster; see `findable_aerial_source` for which cycle |
| `findable_aerial_source` | yes | string | which NAIP cycle the flag was set against, e.g. `naip_2022` |
| `findable_lidar` | yes | bool \| null | left null until LiDAR renders exist for this AOI |
| `notes` | yes | string | optional content, but the field is always present |
| `provenance` | no | string | exact URL or dataset version for reproducibility; required for auxiliary additions (HTMC/MRDS/GNIS), optional otherwise |
| `mvp_required` | no | bool | **MVP success criterion**. If true, this feature MUST be returned for an appropriate text query by the end of the MVP. |

`type` enum:
`waterfall`, `rapids`, `cliff_outcrop`, `beaver_kettle_pond`,
`cabin`, `cabin_ruin`, `mine_pit`, `quarry_pit`, `cemetery`,
`logging_grade`, `dam`, `road`, `parking`, `boardwalk`, `other_mapped`.

`cabin` was added to label the `small-cabin-ne-aoi` feature;
`cabin_ruin` remains reserved for explicitly-ruined structures.
`quarry_pit` is for surface gravel/sand/stone pits as distinct from
underground `mine_pit`.

`category` enum:

- `positive_in_scope` - feature the system should find. Scores the
  retrieval gate.
- `negative_mapped` - existing mapped infrastructure (road, parking,
  dam, boardwalk, etc.) that the anti-join must filter out.
- `out_of_scope` - reserved for features we deliberately exclude
  (e.g. viewshed-only features); currently unused.

## Current contents

12 features inside the AOI:

- 8 `positive_in_scope`: 2 waterfalls (Bond Falls, Upper Bond Falls),
  4 beaver/kettle ponds (all unnamed natural=water or natural=wetland
  polygons from OSM), 1 cemetery (Barclay Cemetery), and 1
  unnamed small cabin (see "Cabin addition" section below).
- 4 `negative_mapped`: Bond Falls Main Dam, main visitor parking, a
  representative boardwalk footway, Bond Falls Road.

All except the cabin are sourced from OpenStreetMap. The cabin was
identified via Google Earth historical imagery (see provenance below).

## Cabin addition (`small-cabin-ne-aoi`)

The first `mvp_required` feature in this eval set.

- **Why added**: a real cabin exists at the coordinate but is fully
  canopy-occluded in every NAIP cycle we have (all summer leaf-on).
  Aerial retrieval verified this experimentally: the chip containing
  the cabin (`r034_c028`) did not surface in any top-10 across the
  initial 8-config (model, bands) sweep for the `abandoned_cabin`
  concept. This is the canonical aerial-only failure case the LiDAR
  modality must close.
- **Why `mvp_required: true`**: hard MVP success criterion. The MVP
  cannot ship until a text query for `cabin` / `abandoned cabin`
  returns this location.
- **How it scores**: `findable_aerial = false` excludes it from
  findable GT in the aerial-only harness (so it doesn't unfairly
  drag down aerial metrics). It contributes to STRICT GT
  (n_gt_strict = 4 because under 50% chip overlap, the point is
  contained by 4 chips). Section D of the gate report shows the
  strict-vs-findable gap as the headline "LiDAR must close this"
  signal.
- **Provenance**: spotted on Google Earth historical imagery dated
  2013-05-09 (May = pre-leaf-out in the UP, so the cabin was
  briefly visible from above). No OSM record found at this
  coordinate. Coordinate authority: Google Earth pin to ~5 m.

A single hand-labeled cabin coordinate is included specifically to
anchor the LiDAR-modality test. Cabin **discovery** remains the
product output (the cabin is one labeled instance, not a
comprehensive list of all cabins in the AOI).

## How this set scores the retrieval gate

The embedding model produces a vector per image chip across the AOI.
For each `positive_in_scope` feature in this list we issue a
natural-language query for its `type` (or close synonyms via prompt
ensembling) and check whether the chip containing the feature's
coordinate ranks in the top-K results. The gate reports recall@K and
MRR per concept against a random baseline.

A clearly-better-than-random ranking across feature types is the
gate-pass. Failure on a specific type (e.g. cemetery, with only one
example) is informative but not by itself a kill. Failure across
multiple types is a kill.

## Honest limitations of the current set

These matter when interpreting future retrieval results.

1. **All current OSM positives are mapped in OSM** (the cabin is the
   exception). This eval set measures whether text retrieval works on
   cataloged feature types, not whether the system finds genuinely
   unmapped features. That second test happens by visual spot-check
   of candidates the pipeline produces at the end of the MVP.
2. **No `cabin_ruin`, `mine_pit`, `quarry_pit` (positive),
   `cliff_outcrop`, `rapids`, or `logging_grade` positives.**
   The AOI was queried against modern OSM and yielded no features of
   these types. They're in scope; future enrichment (LiDAR-derived
   candidates, manual HTMC cross-check with proper georeferencing)
   may add them.
3. **Cabin coordinates beyond the one labeled instance are NOT in
   this list and never will be**, by design. Cabin discovery is the
   product output. Including hand-labeled cabin coordinates would
   defeat the project's whole point. Cabin retrieval is tested
   **indirectly** (via correlated types like buildings/structures)
   and **directly** at the end of the pipeline by visual spot-check
   of candidate LiDAR bumps against modern aerial imagery.
4. **The cemetery is mapped in OSM** with `landuse=cemetery`. The
   anti-join will filter it out of pipeline output. It survives as
   a scoring positive (does the model retrieve a real cemetery when
   queried?), not as an end-product feature.
5. **`findable_aerial`** is set against `naip_2022` (see
   `findable_aerial_source` and the NAIP re-verification section
   below). Only positives with `findable_aerial=true` count toward
   the gate's "at least 3 positives findable from aerial" threshold;
   currently 5 of 7 OSM positives are TRUE so the gate clears.
6. **`findable_lidar` is null on every feature**. It gets set when
   LiDAR renders for this AOI exist.

## Auxiliary data layer pulls (recorded for honesty)

Auxiliary seed sources beyond OSM were attempted; outcome was:

- **MRDS / USMIN** (USGS mines): WFS query inside the AOI returned 0
  features. Sanity-checked with a Western UP bbox that returned 20
  mines, so the service works. The Bond Falls block genuinely has no
  USGS-recorded mineral resource sites.
- **GNIS** (USGS geographic names, Michigan extract): 1 hit inside
  the AOI, Bond Falls itself (feature_id 1619278, map_name Paulding).
  Modern GNIS removed the historical "Civil" feature class around
  2014, so no vanished structures (former post offices, schools,
  populated places) come back from this source. No additions.
- **HTMC** (USGS Historical Topographic Map Collection): visual
  inspection of Watersmeet 1954 and Paulding/Trout Creek 1982 quads
  identified two candidates (Barclay Cemetery, a gravel pit on
  Paulding 1982). Both were REJECTED on verification: the cemetery's
  authoritative OSM coordinate is ~900 m from the topo-derived
  estimate (cemetery is real and is now in the eval set via OSM, not
  via HTMC); the gravel pit has no authoritative coordinate and the
  same crude georeferencing error applies. Net HTMC additions to
  this eval set: 0. Revisit HTMC enrichment later using rasterio +
  pyproj for proper reprojection.

## NAIP re-verification against the actual ingested raster

The `findable_aerial` flag was first set by inspecting each
coordinate on free public web aerial layers (the National Map /
Google satellite mosaic). Once the project ingested its own NAIP
into the working CRS, the flag was re-verified against that raster
(latest cycle: `naip_2022`, 0.6 m native, leaf-on summer). The
`findable_aerial_source` field records the cycle used.

Outcome of the re-verification:

- 5 of 7 `positive_in_scope` features findable on the real NAIP
  (`bond-falls`, `upper-bond-falls`, `unnamed-pond-w-of-falls`,
  `unnamed-pond-s-of-falls`, `unnamed-pond-e-of-flowage`). Gate
  threshold is 3; we clear.
- 4 of 4 `negative_mapped` features visible on the NAIP, so the
  anti-join has real infrastructure to filter.
- The dam-on-dam overlay check passes: `bond-falls-main-dam` sits
  on visible dam structure in the NAIP.

The two positives that come back FALSE:

- `unnamed-wetland-w-of-falls` - wetland blends with summer leaf-on
  forest canopy at 0.6 m. The OSM polygon centroid lands in mottled
  vegetation that does not read as a discrete wetland. The
  Sentinel-2 leaf-off winter passes ingested alongside NAIP may
  surface this at the retrieval step.
- `barclay-cemetery` - the cemetery is small (~11-14 graves); at
  0.6 m the markers are sub-pixel and the grassy clearing is
  borderline distinguishable from surrounding canopy gaps.

Per-feature 200 m x 200 m NAIP chips with a red crosshair on the
exact coordinate are at
`data/verification/eval_chips/<eval_set_id>/<feature_id>.png`. The
AOI-wide overlay (AOI polygon + all eval points labeled by id) is at
`data/verification/<experiment_id>/gate/overlay_check.png`. Both are
regenerated by `python -m terra_query.eval.review`.

NAIP ingest covers cycles 2012, 2014, 2016, 2018, 2020, 2022 plus
two Sentinel-2 winter scenes (2025-03-03, 2026-01-12).
Cross-cycle and leaf-off inspection lives at the gate, not in this
file.

## Demo-success definition

The POC works when:

- Typing `waterfall` ranks Bond Falls in the top results on the map.
- On the full eval set, at least one in-scope positive per type that
  has a positive present in the set appears in the top-K for a
  natural text query for that type, clearly better than chance.

## Coordinate authority

- OSM-sourced features: WGS84 coordinates are OSM polygon centroids
  (for ways) or node positions (for nodes), fetched live when this
  set was built. Authoritative to roughly ~5 m of the mapped feature.
- Bond Falls coordinate is cross-confirmed by Wikipedia, World
  Waterfall Database, and USGS GNIS (feature_id 1619278, decimal
  46.4096728, -89.1326541).
- Barclay Cemetery has a GNIS feature_id (1622554) cross-linked in
  its OSM record.

## How to use this file

- **Retrieval gate**: iterate over `features`, group by `type`, run
  text retrieval per type, compute recall@K (or chosen metric) using
  point-in-chip membership. See
  [src/terra_query/eval/cli/run_n0_retrieval.py](../../../src/terra_query/eval/cli/run_n0_retrieval.py).
- **Demo script**: "Search for waterfall, expect Bond Falls in top
  results."
- **Anti-join regression test**: the `negative_mapped` features
  should disappear from candidate lists after the anti-join runs.
- **When extending**: add a new feature as a new GeoJSON `Feature`
  with the schema above. Keep the file valid GeoJSON; coordinates in
  WGS84; one Point per feature.
