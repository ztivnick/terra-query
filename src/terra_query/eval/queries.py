"""Concept -> prompt-ensemble mapping for the retrieval gate.

12 concepts: 8 with ground truth (recall scored) + 4 without (visual
top-K only). Each concept has 6-10 synonym prompts that are averaged
at query time into a single L2-normalized text vector.

The GT concepts come from the unique `type` values in the eval set.
The no-GT concepts cover in-scope feature classes (rapids,
cliffs/outcrops, mines, logging grades) where the eval set has no
labeled feature in this AOI; they exist so the qualitative top-K
shows how the model behaves on real user queries that have no ground
truth here.

The mapping is data, not behavior. Edit-don't-tune: do NOT inspect
which prompts rank eval chips higher and rewrite the set; that would
fine-tune on the gate and defeat it as an honest signal.
"""

from __future__ import annotations

# concepts with labeled ground truth in the eval set (recall scored)
GT_CONCEPTS: dict[str, list[str]] = {
    "waterfall": [
        "an aerial photo of a waterfall",
        "an aerial photo of a cascading waterfall in a forest",
        "an aerial photo of a river waterfall",
        "an aerial photo of falling water on a river",
        "an aerial photo of a natural waterfall",
        "an aerial photo of whitewater at a falls",
        "an aerial view of a cascade in a river",
        "satellite imagery of a waterfall in dense forest",
    ],
    "pond": [
        "an aerial photo of a small pond in a forest",
        "an aerial photo of a beaver pond",
        "an aerial photo of a kettle pond",
        "an aerial photo of a small still water body",
        "an aerial photo of a forest pond",
        "satellite imagery of a small lake surrounded by trees",
        "an aerial view of dark still water in the woods",
        "an aerial photo of a small natural pond",
    ],
    "dam": [
        "an aerial photo of a dam",
        "an aerial photo of a concrete dam on a river",
        "an aerial photo of a hydroelectric dam",
        "an aerial photo of a small earthen dam",
        "an aerial view of a dam structure across a river",
        "satellite imagery of a dam holding back water",
        "an aerial photo of a man-made water control structure",
        "an aerial photo of a dam with a reservoir",
    ],
    "road": [
        "an aerial photo of a small road through a forest",
        "an aerial photo of a paved rural road",
        "an aerial photo of a two-lane country road",
        "an aerial view of a road cutting through trees",
        "satellite imagery of a forest road",
        "an aerial photo of a narrow road in the woods",
        "an aerial photo of an asphalt road in a forest clearing",
        "an aerial view of a road and shoulder in dense forest",
    ],
    "parking_lot": [
        "an aerial photo of a parking lot",
        "an aerial photo of a small parking area in a forest",
        "an aerial view of a paved parking lot",
        "an aerial photo of a gravel parking area",
        "satellite imagery of a visitor parking lot",
        "an aerial photo of cars parked in a forest clearing",
        "an aerial view of a small paved parking area near a trailhead",
        "an aerial photo of a parking lot at a park",
    ],
    "boardwalk": [
        "an aerial photo of a wooden boardwalk in a forest",
        "an aerial photo of a wooden walkway over a wetland",
        "an aerial view of a viewing platform near a waterfall",
        "an aerial photo of a wooden footbridge in the woods",
        "satellite imagery of a wooden boardwalk for visitors",
        "an aerial photo of a wooden path through the trees",
        "an aerial view of a wooden observation deck",
        "an aerial photo of a raised wooden walkway in a park",
    ],
    "cemetery": [
        "an aerial photo of a cemetery",
        "an aerial photo of a small rural cemetery",
        "an aerial photo of a graveyard in a clearing",
        "an aerial view of a forest cemetery with headstones",
        "satellite imagery of a small graveyard surrounded by trees",
        "an aerial photo of a burial ground in the woods",
        "an aerial view of a rural cemetery with rows of graves",
        "an aerial photo of a small memorial cemetery",
    ],
    "abandoned_cabin": [
        # cabin is `findable_aerial=false` (canopy-occluded), so it
        # contributes to strict GT but NOT findable GT for this concept.
        # Findable aggregates skip concepts with n_gt_findable=0 so this
        # doesn't unfairly drag down aerial metrics — it shows up as the
        # headline aerial-only gap in section D, exactly the way cemetery
        # does.
        "an aerial photo of an abandoned cabin in a forest",
        "an aerial photo of an old wooden hunting cabin",
        "an aerial photo of a small ruined building in the woods",
        "an aerial photo of an abandoned log cabin",
        "an aerial view of a small old wooden structure in a forest clearing",
        "satellite imagery of a dilapidated rural cabin",
        "an aerial photo of an old camp shelter in dense forest",
        "an aerial photo of a ruin of a small building in the woods",
    ],
}

# concepts in the POC scope but with no labeled feature in this AOI.
# qualitative top-K only; no recall metric.
NO_GT_CONCEPTS: dict[str, list[str]] = {
    "mine_pit": [
        "an aerial photo of an abandoned mine pit",
        "an aerial photo of a small prospect pit in a forest",
        "an aerial photo of a mining tailings pile",
        "an aerial view of bare ground at an old mine site",
        "satellite imagery of a small open pit mine in the woods",
        "an aerial photo of disturbed earth at an abandoned mine",
        "an aerial photo of a small quarry in a forest",
        "an aerial view of scarred ground from old mining activity",
    ],
    "cliff_outcrop": [
        "an aerial photo of a rock cliff in a forest",
        "an aerial photo of a rocky outcrop on a hillside",
        "an aerial photo of an exposed bedrock ridge",
        "an aerial view of a stone bluff above a river",
        "satellite imagery of a small cliff face in the woods",
        "an aerial photo of bare rock protruding from a forested slope",
        "an aerial photo of a rocky escarpment",
        "an aerial view of a small cliff edge in dense forest",
    ],
    "old_logging_road": [
        "an aerial photo of an old logging road in a forest",
        "an aerial photo of an overgrown forest road",
        "an aerial photo of an abandoned dirt track in the woods",
        "an aerial view of a faint linear cut through dense forest",
        "satellite imagery of an old skidder trail",
        "an aerial photo of a grassy logging road in the woods",
        "an aerial photo of an unmaintained forest road",
        "an aerial view of a narrow overgrown track through trees",
    ],
    "river_rapids": [
        "an aerial photo of river rapids in a forest",
        "an aerial photo of whitewater on a river",
        "an aerial photo of a rocky river reach with rapids",
        "an aerial view of foamy whitewater between rocks in a river",
        "satellite imagery of a rapid section of a river",
        "an aerial photo of a riffle on a forested river",
        "an aerial photo of fast-moving water around boulders",
        "an aerial view of a turbulent stretch of river in the woods",
    ],
}


def all_concepts() -> dict[str, list[str]]:
    """GT + no-GT, merged. Stable order: GT first, no-GT second."""
    merged: dict[str, list[str]] = {}
    merged.update(GT_CONCEPTS)
    merged.update(NO_GT_CONCEPTS)
    return merged


# concept -> eval feature type that produces its ground truth (None for no-GT)
CONCEPT_TO_N0_TYPE: dict[str, str | None] = {
    "waterfall": "waterfall",
    "pond": "beaver_kettle_pond",
    "dam": "dam",
    "road": "road",
    "parking_lot": "parking",
    "boardwalk": "boardwalk",
    "cemetery": "cemetery",
    "abandoned_cabin": "cabin",
    "mine_pit": None,
    "cliff_outcrop": None,
    "old_logging_road": None,
    "river_rapids": None,
}
