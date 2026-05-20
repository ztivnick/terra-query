# eval set - Bond Falls block

This directory holds the **scorecard** for the Terra-query MVP.
Not the product output. The product output (unmapped feature
candidates on a map) emerges from the pipeline at later steps; this
list is the ground truth used to test whether the model is finding
the things it should be finding.

Two files:

- `known_features.geojson` - hand-authored GeoJSON FeatureCollection
  of Point features, WGS84 (EPSG:4326).
- `README.md` - this file.

The AOI polygon lives in `../aoi/bond_falls_block.geojson`. Every
point in `known_features.geojson` falls inside that polygon.

## Why this exists (N0)

Without a hand-authored list of real features at known coordinates,
"is retrieval working?" has no answer and the demo can silently lie.
The eval set fixes that: when the model returns "top 10 matches for
'waterfall'", we check whether Bond Falls (which we know is at
(46.40967, -89.13265)) appears in those 10. If yes, the model can
find waterfalls. We then trust it generally, including for queries
the eval set does not directly label (e.g. "cabin").

## In-scope feature types (verbatim from N0)

waterfalls/cascades, rapids, cliffs and outcrops, beaver/kettle
ponds, old cabins/camps/ruins, mines and prospect pits, old logging
grades / unmapped trails.

Explicit out-of-scope per strategy doc Part C:
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
| `findable_aerial` | yes | bool \| null | set during S1 build step 4; null if not yet checked |
| `findable_lidar` | yes | bool \| null | left null at S1; updated when LiDAR renders exist (S8) |
| `notes` | yes | string | optional content, but the field is always present |
| `provenance` | no | string | exact URL or dataset version for reproducibility; required for auxiliary additions (HTMC/MRDS/GNIS), optional otherwise |

`type` enum:
`waterfall`, `rapids`, `cliff_outcrop`, `beaver_kettle_pond`,
`cabin_ruin`, `mine_pit`, `quarry_pit`, `cemetery`, `logging_grade`,
`dam`, `road`, `parking`, `boardwalk`, `other_mapped`.

(`quarry_pit` and `cemetery` were added during S1 step 2b to fit a
specific candidate; `quarry_pit` is for surface gravel/sand/stone
pits as distinct from underground `mine_pit`.)

`category` enum:
- `positive_in_scope` - feature the system should find. Scores the
  N7 retrieval gate.
- `negative_mapped` - existing mapped infrastructure (road, parking,
  dam, boardwalk, etc.) that the N5 anti-join must filter out.
- `out_of_scope` - reserved for features we deliberately exclude
  (e.g. viewshed-only features); currently unused.

## Current contents (snapshot at end of S1)

11 features inside the AOI:

- 7 `positive_in_scope`: 2 waterfalls (Bond Falls, Upper Bond Falls),
  4 beaver/kettle ponds (all unnamed natural=water or natural=wetland
  polygons from OSM), 1 cemetery (Barclay Cemetery).
- 4 `negative_mapped`: Bond Falls Main Dam, main visitor parking, a
  representative boardwalk footway, Bond Falls Road.

Every feature is sourced from OpenStreetMap as of the S1 build run.
All 11 OSM ids were re-confirmed live via Overpass at the close of
step 2b.

## How this set scores N7 (the main kill gate)

At S5 (N7), the embedding model produces a vector per image chip
across the AOI. For each `positive_in_scope` feature in this list,
we issue a natural-language query for its `type` (or close synonyms
via prompt ensembling, see strategy doc N7) and check whether the
chip containing the feature's coordinate ranks in the top-K results.
Exact K and the scoring metric (recall@K, MRR, or both) get pinned
at S5. The S1 gate just requires the list exists and is honest.

A clearly-better-than-random ranking across feature types is the
gate-pass. Failure on a specific type (e.g. cemetery, with only one
example) is informative but not by itself a kill. Failure across
multiple types is a kill.

## Honest limitations of the current set

These matter when interpreting future N7 results.

1. **All current positives are mapped in OSM.** This eval set
   measures whether text retrieval works on cataloged feature types,
   not whether the system finds genuinely unmapped features. That
   second test happens by visual spot-check of candidates the
   pipeline produces at the end of the MVP.
2. **No `cabin_ruin`, `mine_pit`, `quarry_pit` (positive),
   `cliff_outcrop`, `rapids`, or `logging_grade` positives at S1.**
   The AOI was queried against modern OSM and yielded no features of
   these types. The strategy doc names them in scope; future
   enrichment (S8 LiDAR-derived candidates, manual HTMC cross-check
   with proper georeferencing) may add them.
3. **Cabin coordinates are NOT in this list and never will be**, by
   design. Cabin discovery is the product output. Including hand-
   labeled cabin coordinates would defeat the project's whole point.
   Cabin retrieval is tested **indirectly** (via correlated types
   like buildings/structures) and **directly** at the end of the
   pipeline by visual spot-check of candidate LiDAR bumps against
   modern aerial imagery.
4. **The cemetery is mapped in OSM** with `landuse=cemetery`. The
   N5 anti-join at S11 will filter it out of pipeline output. It
   survives as an N7-scoring positive (does the model retrieve a
   real cemetery when queried?), not as an end-product feature.
5. **`findable_aerial` is null on every feature** until S1 step 4
   runs. The values get set there by eye-check against a free public
   aerial layer. Only positives with `findable_aerial=true` count
   toward the gate's "at least 3 positives findable from aerial"
   threshold.
6. **`findable_lidar` is null on every feature**. It gets set at S8
   when LiDAR renders exist.

## Auxiliary data layer pulls during S1 step 2b (recorded for honesty)

Strategy doc N0 has auxiliary seed sources beyond OSM. Attempted at
S1; outcome was:

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
  authoritative OSM coordinate is ~900 m from my topo-derived
  estimate (cemetery is real and is now in the eval set via OSM, not
  via HTMC); the gravel pit has no authoritative coordinate and the
  same crude georeferencing error applies. Net HTMC additions to
  this eval set: 0. Revisit HTMC enrichment only after S2 (working
  CRS pinned) using rasterio + pyproj for proper reprojection.

## Demo-success definition (anchors later gates)

The POC works when:

- Typing `waterfall` ranks Bond Falls in the top results on the map
  (N9 gate restated).
- On the full eval set, at least one in-scope positive per type that
  has a positive present in the set appears in the top-K for a
  natural text query for that type, clearly better than chance. K
  and the exact metric get pinned at S5/N7.

## Coordinate authority

- OSM-sourced features: WGS84 coordinates are OSM polygon centroids
  (for ways) or node positions (for nodes), fetched at the S1 build
  run and re-confirmed live before close of step 2b. Authoritative
  to roughly ~5 m of the mapped feature.
- Bond Falls coordinate is cross-confirmed by Wikipedia, World
  Waterfall Database, and USGS GNIS (feature_id 1619278, decimal
  46.4096728, -89.1326541).
- Barclay Cemetery has a GNIS feature_id (1622554) cross-linked in
  its OSM record.

## How to use this file

- At S5 (N7): iterate over `features`, group by `type`, run text
  retrieval per type, compute recall@K (or chosen metric) using
  point-in-chip membership.
- At S7 (N9): use as the demo script. "Search for waterfall, expect
  Bond Falls in top results."
- At S11 (N5): the `negative_mapped` features should disappear from
  candidate lists after the anti-join runs. Use this list to
  regression-test the anti-join.
- When extending: add a new feature as a new GeoJSON `Feature` with
  the schema above. Run the validator (see `docs/architecture_plans/
  s01-target-and-eval-set.md` build step 5).
