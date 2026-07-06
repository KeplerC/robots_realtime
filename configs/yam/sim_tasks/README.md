# YAM sim task launch configs

One YAML per task, launched with:

```
uv run rr-session configs/yam/sim_tasks/<task>.yaml
```

## Base task natures (9)

Each file below corresponds to a unique task class — i.e. a distinct "thing to
do" against the bimanual YAM sim.

| Config                      | Task class                     | Goal                                   |
|-----------------------------|--------------------------------|----------------------------------------|
| `bottles.yaml`              | (legacy bottle scene)          | Grasp plastic bottles from the table.  |
| `red_only.yaml`             | (legacy bottle scene, red)     | Grasp a single red bottle.             |
| `tape_handover.yaml`        | `TapeHandoverTask`             | Hand a yellow tape roll across.        |
| `pick_up_tiger.yaml`        | `PickUpTigerTask`              | Pick up a plush tiger.                 |
| `place_plate_in_rack.yaml`  | `PlacePlateInRackTask`         | Place a plate upright in a dish rack.  |
| `double_marker_in_cup.yaml` | `DoubleMarkerInCupTask`        | Drop two markers into a cup.           |
| `hang_tool_on_pegboard.yaml`| `HangToolOnPegboardTask`       | Hang a hand tool on a pegboard hook.   |
| `medical_tray.yaml`         | `MedicalTrayTask`              | Arrange pill bottles in a tray.        |
| `hang_spoon_on_hook.yaml`   | `HangSpoonOnHookTask`          | Hang a spoon on a hook.                |

`tape_handover` / `pick_up_tiger` use the defaults `placement_mode=poisson`
and `spawn_region=full`. Alternatives live in `variants/`.

## Variants (`variants/`)

Same underlying task classes, different evaluation conditions. Use these when
you specifically want to evaluate a different sampling strategy, color scheme,
or distribution shift. Pick up a variant by launching `sim_tasks/variants/<name>.yaml`.

| Variant                                  | Wraps           | What it changes                                           |
|------------------------------------------|-----------------|-----------------------------------------------------------|
| `tape_handover_grid.yaml`                | TapeHandoverTask| Grid placement instead of Poisson.                        |
| `tape_handover_ood_yellow_right.yaml`    | TapeHandoverOODTask | Yellow tape spawns on the right (OOD color/side).    |
| `tape_handover_ood_grey_left.yaml`       | TapeHandoverOODTask | Grey tape spawns on the left.                        |
| `tape_handover_ood_both.yaml`            | TapeHandoverOODTask | Both colors swapped.                                  |
| `tape_handover_table_high.yaml`          | TablePoseOOD    | Table raised by 0–15 cm.                                  |
| `tape_handover_table_shift.yaml`         | TablePoseOOD    | Table shifted in x/y.                                     |
| `tape_handover_distractor.yaml`          | DistractorOOD   | Extra distractor objects on the table.                    |
| `tape_handover_object_ood.yaml`          | ObjectOOD       | Task object swapped for an unseen mesh.                   |
| `pick_up_tiger_right_half.yaml`          | PickUpTigerTask | Tiger spawns in right half of table only.                 |
| `pick_up_tiger_middle_strip.yaml`        | PickUpTigerTask | Tiger spawns in a middle strip only.                      |
| `tiger_table_high.yaml`                  | TablePoseOOD    | Tiger task with table raised 0–15 cm.                     |
| `tiger_object_ood.yaml`                  | ObjectOOD       | Tiger task with an unseen object.                         |

## How the "task nature" dedup works

The 19 registered `eyeball_task` keys collapse into **7 base task classes + 6
OOD wrappers** plus a handful of `partial(...)` parameter tweaks:

- `TapeHandoverTask(placement_mode=...)`
- `TapeHandoverOODTask(ood_mode=...)` (subclass — color swap)
- `PickUpTigerTask(spawn_region=...)`
- `TablePoseOOD(...)`, `DistractorOOD(...)`, `ObjectOOD(...)` wrapping a base.

If you don't need the variants, delete the `variants/` folder — the base 9
configs are the full working set.
