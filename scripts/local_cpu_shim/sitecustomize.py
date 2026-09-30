"""LOCAL CPU-ONLY HARNESS WORKAROUND (opt-in via FDAGENT_CPU_HARNESS_SHIM=1 in scripts/reproduce.sh).

The official FDB-v3 runner calls ``model.cuda()`` unconditionally on its NeMo ASR model
(run_tool_benchmark.py:load_asr_model), which raises on machines without CUDA. Loaded through
PYTHONPATH into the harness process only, this shim:
  * activates only when the process is run_tool_benchmark.py;
  * patches only lightning's DeviceDtypeModuleMixin.cuda (the method NeMo models inherit);
  * when CUDA is unavailable returns the model unchanged (stays on CPU); with CUDA, original behaviour;
  * removes itself from PYTHONPATH so child processes never load it.
No harness, scorer or benchmark file is modified. Not needed on the official GPU machine.
"""

import os
import sys

_TAG = "[LOCAL CPU-ONLY HARNESS WORKAROUND]"
_HERE = os.path.dirname(os.path.abspath(__file__))

_pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and os.path.abspath(p) != _HERE]
if _pp:
    os.environ["PYTHONPATH"] = os.pathsep.join(_pp)
else:
    os.environ.pop("PYTHONPATH", None)

if sys.argv and os.path.basename(sys.argv[0]) == "run_tool_benchmark.py":
    try:
        import torch
        from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin as _Mixin
    except Exception as e:  # fail loudly: the workaround could not be installed
        print(f"{_TAG} could not install shim: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    else:
        _original_cuda = _Mixin.cuda

        def _cuda(self, device=None):
            if torch.cuda.is_available():
                return _original_cuda(self, device)
            print(f"{_TAG} CUDA unavailable: {type(self).__name__}.cuda() skipped; model stays on CPU",
                  file=sys.stderr, flush=True)
            return self

        _Mixin.cuda = _cuda
        print(f"{_TAG} active (torch {torch.__version__}, cuda_available={torch.cuda.is_available()})",
              file=sys.stderr, flush=True)
