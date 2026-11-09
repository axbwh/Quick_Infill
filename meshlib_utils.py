"""
Meshlib import utilities for Quick Infill.

meshlib's own package __init__ registers its bundled DLL folder
(meshlib_core.libs) with Windows, so nothing extra is needed here. The add-on
used to add DLL folders to PATH by hand; that found the wrong (older, other
Python version) meshlib build on Blender 5.2 and has been removed.
"""

import threading

_modules = None

# CUDA work (FastWindingNumber offsets) must not run concurrently: several
# simultaneous CUDA jobs on the GPU that also drives the display can trip a
# driver timeout (TDR) and freeze the whole machine. CPU-side meshlib work
# (decimation, voxel booleans, undercut fixing) may still run in parallel.
CUDA_LOCK = threading.Lock()


def get_meshlib():
    """Return (mrmeshpy, mrcudapy), importing them once (on first use, not at startup)."""
    global _modules
    if _modules is None:
        try:
            import meshlib.mrmeshpy as mm
            import meshlib.mrcudapy as mc
        except ImportError as e:
            raise ImportError(
                f"Quick Infill: failed to load meshlib - {e}\n"
                "Reinstall the add-on via Edit > Preferences > Get Extensions > Install from Disk"
            )
        _prevent_quit_hang(mm)
        _modules = (mm, mc)
    return _modules


def _prevent_quit_hang(mm):
    """Stop Blender from hanging when it quits after meshlib was loaded.

    meshlib installs an exception handler that prints a stack trace using the
    Windows debug engine (dbgeng). When Blender quits, a harmless exception
    inside a thread that is shutting down reaches that handler while Windows
    holds its DLL loader lock; dbgeng then initialises for the first time,
    needs that same lock, and the process deadlocks (Blender never closes).
    Initialising dbgeng once now, outside the lock, avoids it: in testing,
    quitting hung in 2 of 4 normal and 4 of 5 background runs without this,
    and in 0 of 14 runs with it. Costs about 0.6 s, once per session.
    """
    try:
        mm.getCurrentStacktrace()
    except Exception:
        pass


class JobCancelled(Exception):
    """The user pressed Esc; the running meshlib job stopped early."""


def _esc_held_in_this_app():
    """True while Esc is held down and one of this process's windows has focus.

    meshlib keeps Python's lock for the whole of a long call, so Blender's own
    event loop (and any modal operator) can't see key presses during a job.
    The keyboard is therefore read directly from Windows between steps.
    """
    try:
        import ctypes
        import os
        user32 = ctypes.windll.user32
        if not (user32.GetAsyncKeyState(0x1B) & 0x8001):  # VK_ESCAPE held, or pressed since last check
            return False
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
        return pid.value == os.getpid()
    except Exception:  # not Windows, or no user32
        return False


class JobControl:
    """Progress + cancel state for one running Quick Infill job.

    The pipeline calls check() between steps (each offset, island, undercut
    direction and object); a pressed Esc stops the job at the next step.
    (No callback goes into meshlib itself - see progress_callback().)
    """

    def __init__(self, total_items=1, on_progress=None, watch_esc=True):
        self._cancel = threading.Event()
        self.total = max(1, total_items)
        self.done_items = 0
        self.step = 0.0
        self.on_progress = on_progress  # fn(fraction), main thread only
        self.watch_esc = watch_esc
        self._main = threading.main_thread()
        if watch_esc:
            _esc_held_in_this_app()  # clear a press left over from before the job

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def poll(self):
        """Check Esc and report progress; safe to call often."""
        if self.watch_esc and not self._cancel.is_set() and _esc_held_in_this_app():
            self._cancel.set()
        if self.on_progress is not None and threading.current_thread() is self._main:
            try:
                self.on_progress(self.fraction())
            except Exception:
                pass

    def check(self):
        self.poll()
        if self._cancel.is_set():
            raise JobCancelled()

    def item_done(self):
        self.done_items += 1
        self.step = 0.0
        self.poll()

    def fraction(self):
        return min(1.0, (self.done_items + min(self.step, 0.999)) / self.total)

    def callback(self):
        def cb(f):
            self.step = float(f)
            self.poll()
            return not self._cancel.is_set()
        return cb


# The job currently running (one at a time); operations read it to attach callbacks.
ACTIVE_JOB = None


def set_active_job(control):
    global ACTIVE_JOB
    ACTIVE_JOB = control


def progress_callback():
    """Deliberately always None: never give meshlib a Python progress callback.

    meshlib holds Python's lock for the whole call, and on large grids it calls
    the progress callback from its own worker threads (e.g. the CUDA winding
    number "by parts" path, parallel decimation). Those threads then wait for
    Python's lock forever and Blender deadlocks (seen in testing). Cancelling is
    instead checked between steps with check_cancelled().
    """
    return None


def check_cancelled():
    if ACTIVE_JOB is not None:
        ACTIVE_JOB.check()


def reraise_if_cancelled(exc):
    """meshlib reports a cancel as RuntimeError('Operation was canceled')."""
    if isinstance(exc, JobCancelled):
        raise exc
    if ACTIVE_JOB is not None and ACTIVE_JOB.cancelled:
        raise JobCancelled() from exc


def get_mrmeshnumpy():
    get_meshlib()
    import meshlib.mrmeshnumpy as mn
    return mn


def get_mrmeshpy():
    return get_meshlib()[0]


def get_mrcudapy():
    return get_meshlib()[1]
