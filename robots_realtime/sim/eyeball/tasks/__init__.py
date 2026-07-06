"""Tasks package — re-exports everything for zero-breakage backwards compatibility."""

from .base import Task
from .tape import TapeHandoverTask, TapeHandoverOODTask
from .tiger import PickUpTigerTask, PickUpTigerTaskRightHalf, PickUpTigerTaskMiddleStrip
from .marker import DoubleMarkerInCupTask
from .plate import PlacePlateInRackTask
from .pegboard import HangToolOnPegboardTask
from .medical_tray import MedicalTrayTask
from .spoon_hanging import HangSpoonOnHookTask
from .wrapper import TaskWrapper, TablePoseOOD, CameraOOD, VisualOOD, LightingOOD, ObjectOOD, DistractorOOD
from .registry import TASK_REGISTRY

__all__ = [
    "Task",
    "TapeHandoverTask", "TapeHandoverOODTask",
    "PickUpTigerTask", "PickUpTigerTaskRightHalf", "PickUpTigerTaskMiddleStrip",
    "DoubleMarkerInCupTask",
    "PlacePlateInRackTask",
    "HangToolOnPegboardTask",
    "MedicalTrayTask",
    "HangSpoonOnHookTask",
    "TaskWrapper", "TablePoseOOD", "CameraOOD", "VisualOOD", "LightingOOD", "ObjectOOD", "DistractorOOD",
    "TASK_REGISTRY",
]
