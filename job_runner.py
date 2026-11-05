"""
Safety net for heavy Quick Infill jobs.

Operators that mix in ModalJobMixin:
  * estimate how long the job will take and ask first when it is longer than
    the "Ask Before Jobs Over" setting, so a huge remesh is never started by
    accident (the main cause of force-quitting Blender mid-job), and
  * can be stopped with Esc: the job stops at the next step (each offset,
    island, undercut direction or object), and nothing in the scene changes.
    The cursor shows the progress percentage.

Why not a background thread? meshlib keeps Python's lock for the whole of a
long call, so Blender's Python UI (and any modal operator) can't run while it
works anyway; a thread would add risk without making Blender responsive.
"""

import bpy

import time

from .meshlib_utils import JobControl, JobCancelled, set_active_job
from .blender_meshlib_utils import EstimateLearner


class JobSpec:
    """What a job does, in three steps.

    prepare(): read bpy data.  compute(control): meshlib only.
    finish(): write results; returns an operator result set.
    """

    def __init__(self, prepare, compute, finish, items=1):
        self.prepare = prepare
        self.compute = compute
        self.finish = finish
        self.items = max(1, items)


def format_seconds(sec):
    if sec < 90:
        return f"{sec:.0f} seconds"
    if sec < 5400:
        return f"{sec / 60:.0f} minutes"
    return f"{sec / 3600:.1f} hours"


class ModalJobMixin:
    """Mix into an Operator. Subclasses implement job_setup(context) -> JobSpec | None
    and job_estimate(context) -> seconds."""

    def job_setup(self, context):
        raise NotImplementedError

    def job_estimate(self, context):
        return 0.0

    def invoke(self, context, event):
        limit = float(getattr(context.scene.quick_infill_settings, "ask_above_seconds", 0.0))
        self._predicted = 0.0
        if limit > 0:
            try:
                self._predicted = float(self.job_estimate(context))
            except Exception as e:  # an estimate must never block the tool
                print(f"[Quick Infill] estimate failed: {e}")
            est = EstimateLearner.correct(self.bl_idname, self._predicted)
            if est > limit:
                return context.window_manager.invoke_confirm(
                    self, event,
                    title=f"{self.bl_label}: long job",
                    message=(f"This is estimated to take about {format_seconds(est)} because the voxel grid is very "
                             f"large for the selected objects. Try a coarser voxel size or Islands. "
                             f"Press Esc while it runs to stop it after the current step."),
                    confirm_text="Run Anyway",
                    icon='WARNING',
                )
        return self.execute(context)

    def execute(self, context):
        try:
            spec = self.job_setup(context)
            if spec is None:
                return {'CANCELLED'}
            spec.prepare()
        except Exception as e:
            self.report({'ERROR'}, f"{self.bl_label} failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        t0 = time.perf_counter()
        interactive = not bpy.app.background and context.window is not None
        wm = context.window_manager
        control = JobControl(spec.items,
                             on_progress=(lambda f: wm.progress_update(int(f * 100))) if interactive else None,
                             watch_esc=interactive)
        if interactive:
            wm.progress_begin(0, 100)
        set_active_job(control)
        try:
            spec.compute(control)
            if control.cancelled:
                raise JobCancelled()
        except JobCancelled:
            self.report({'WARNING'}, f"{self.bl_label} cancelled. Nothing was changed.")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"{self.bl_label} failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        finally:
            set_active_job(None)
            if interactive:
                wm.progress_end()

        try:
            result = spec.finish()
            # Learn from the real time (only when the job was estimated first).
            predicted = getattr(self, "_predicted", 0.0)
            if predicted:
                EstimateLearner.record(self.bl_idname, predicted, time.perf_counter() - t0)
            return result
        except Exception as e:
            self.report({'ERROR'}, f"{self.bl_label} failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
