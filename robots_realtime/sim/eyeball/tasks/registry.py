"""TASK_REGISTRY mapping task names to task classes/partials."""

from functools import partial

from .base import Task
from .tape import TapeHandoverTask, TapeHandoverOODTask
from .tiger import PickUpTigerTask
from .marker import DoubleMarkerInCupTask
from .plate import PlacePlateInRackTask
from .medical_tray import MedicalTrayTask
from .pegboard import HangToolOnPegboardTask
from .spoon_hanging import HangSpoonOnHookTask
from .wrapper import TablePoseOOD, ObjectOOD, DistractorOOD

TASK_REGISTRY: dict[str, type[Task] | partial[Task]] = {
    "tape_handover_random": partial(TapeHandoverTask, placement_mode="poisson"),
    "tape_handover_grid": partial(TapeHandoverTask, placement_mode="grid"),
    "tape_handover_ood_yellow_right": partial(TapeHandoverOODTask, ood_mode="yellow_right"),
    "tape_handover_ood_grey_left": partial(TapeHandoverOODTask, ood_mode="grey_left"),
    "tape_handover_ood_both": partial(TapeHandoverOODTask, ood_mode="both"),
    "place_plate_in_rack": PlacePlateInRackTask,
    "double_marker_in_cup": DoubleMarkerInCupTask,
    "pick_up_tiger": partial(PickUpTigerTask, spawn_region="full"),
    "pick_up_tiger_right_half": partial(PickUpTigerTask, spawn_region="right_half"),
    "pick_up_tiger_middle_strip": partial(PickUpTigerTask, spawn_region="middle_strip"),
    "hang_tool_on_pegboard": HangToolOnPegboardTask,
    "medical_tray": MedicalTrayTask,
    "hang_spoon_on_hook": HangSpoonOnHookTask,
    # OOD wrappers (camera/visual/lighting variants need eye_camera — omitted)
    "tape_handover_table_high":  lambda: TablePoseOOD(TapeHandoverTask(placement_mode="poisson"), dz=(0.0, 0.15)),
    "tape_handover_table_shift": lambda: TablePoseOOD(TapeHandoverTask(placement_mode="poisson"), dx=(-0.05, 0.05), dy=(-0.03, 0.03)),
    "tape_handover_distractor":  lambda: DistractorOOD(TapeHandoverTask(placement_mode="poisson")),
    "tape_handover_object_ood":  lambda: ObjectOOD(TapeHandoverTask(placement_mode="poisson")),
    "tiger_table_high":          lambda: TablePoseOOD(PickUpTigerTask(), dz=(0.0, 0.15)),
    "tiger_object_ood":          lambda: ObjectOOD(PickUpTigerTask()),
}
