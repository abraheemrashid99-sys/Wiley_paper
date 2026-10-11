"""Audit the device handling of the code on a machine without a GPU.

    VDEV_LOG=audit.tsv PYTHONPATH=tools/device_audit \
        python scripts/reproduce.py --config configs/smoke.yaml --out runs/audit --export runs/audit_export device=cuda

Python imports a module named ``sitecustomize`` at start-up, so with this
folder on ``PYTHONPATH`` the audit is active in every process (also in the
ones started by the scripts) as long as ``VDEV_LOG`` is set.

It emulates a second device: every tensor carries a tag, 'gpu' or 'cpu'.
A request for the device ``cuda`` is served in CPU memory and tagged 'gpu';
``tensor.device`` answers accordingly. Each operation that would fail on a
real GPU is written to the log with the line of the project that caused it:

* MIXED       operands on different devices (a CPU tensor indexed by a GPU
              tensor included; a GPU tensor indexed by a CPU tensor is allowed);
* mixed-0dim  the same with a zero-dimensional CPU operand (PyTorch allows
              this for arithmetic, so these lines are to be read, not fixed);
* numpy       conversion of a GPU tensor to NumPy;
* generator   a CPU random generator used to draw directly on the GPU.

An empty log (apart from ``mixed-0dim`` lines inside ``torch/nn/utils/clip_grad.py``,
where PyTorch moves a scalar itself) means that no script mixes devices.
It cannot show what only real hardware shows: memory limits and speed.
"""

import os

if os.environ.get("VDEV_LOG"):
    import linecache
    import sys

    import torch
    from torch.overrides import TorchFunctionMode

    FAKE, CPU = torch.device("cuda"), torch.device("cpu")
    ROOT = os.environ.get("VDEV_ROOT", os.getcwd())
    seen: dict = {}

    def tag_of(t):
        return getattr(t, "_vdev", None)

    def set_tag(t, tag):
        try:
            t._vdev = tag
        except Exception:
            pass

    def tensors(obj, out):
        if isinstance(obj, torch.Tensor):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                tensors(v, out)
        elif isinstance(obj, dict):
            for v in obj.values():
                tensors(v, out)
        return out

    def site():
        frame = sys._getframe(3)
        inner, own = None, None
        while frame is not None:
            name = frame.f_code.co_filename
            if inner is None and "torch/overrides.py" not in name:
                inner = (name, frame.f_lineno)
            if own is None and name.startswith(ROOT) and "sitecustomize" not in name:
                own = (name, frame.f_lineno)
                break
            frame = frame.f_back
        own = own or inner
        return f"{os.path.relpath(own[0], ROOT)}:{own[1]}", f"{inner[0].split('site-packages/')[-1].split('dist-packages/')[-1]}:{inner[1]}"

    def report(kind, name, detail=""):
        own, inner = site()
        key = (kind, name, own, inner)
        if key not in seen:
            seen[key] = detail
            path, _, number = own.rpartition(":")
            line = linecache.getline(os.path.join(ROOT, path), int(number)).strip()
            with open(os.environ["VDEV_LOG"], "a", encoding="utf-8") as handle:
                handle.write(f"{kind}\t{name}\t{own}\t{inner}\t{detail}\t{line}\t{' '.join(sys.argv[:1])}\n")

    def as_tag(device):
        return "gpu" if torch.device(device).type == "cuda" else "cpu"

    def is_device(x):
        if isinstance(x, torch.device):
            return True
        return isinstance(x, str) and x.split(":")[0] in ("cpu", "cuda", "mps")

    FACTORIES = {"zeros", "ones", "full", "empty", "arange", "tensor", "randn", "rand", "randint", "linspace", "eye", "as_tensor", "from_numpy", "randperm", "logspace", "normal", "tril_indices", "triu_indices", "scalar_tensor"}
    TRANSFER = {"to", "cpu", "cuda", "copy_", "type", "float", "double", "long", "int", "bool", "half"}

    class VDev(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            name = getattr(func, "__name__", str(func))
            owner = getattr(func, "__self__", None)
            # --- property getters / setters
            if name == "__get__":
                prop = getattr(owner, "__name__", "")
                if prop == "device":
                    return FAKE if tag_of(args[0]) == "gpu" else CPU
                if prop == "is_cuda":
                    return tag_of(args[0]) == "gpu"
                out = func(*args, **kwargs)
                if isinstance(out, torch.Tensor) and out is not args[0]:
                    set_tag(out, tag_of(args[0]))
                return out
            if name == "__set__":
                out = func(*args, **kwargs)
                if getattr(owner, "__name__", "") == "data" and isinstance(args[1], torch.Tensor):
                    set_tag(args[0], tag_of(args[1]))
                return out

            ins = tensors(args, []) + tensors(kwargs, [])
            tags = {tag_of(t) for t in ins} - {None}
            target = None
            # --- requested device
            if kwargs.get("device") is not None:
                target = as_tag(kwargs["device"])
                kwargs["device"] = CPU
            if name == "to":
                new = []
                for a in args[1:]:
                    if is_device(a):
                        target = as_tag(a)
                        a = CPU
                    elif isinstance(a, torch.Tensor):
                        target = tag_of(a) or target
                    new.append(a)
                args = (args[0], *new)
            elif name == "cuda":
                target, func, args, kwargs = "gpu", torch.Tensor.clone, (args[0],), {}
            elif name == "cpu":
                target = "cpu"
            if kwargs.get("generator") is not None and target == "gpu":
                report("generator", name, "CPU generator used to create a tensor on the GPU")
            if name in ("numpy", "__array__", "__array_wrap__") and ins and tag_of(ins[0]) == "gpu":
                report("numpy", name, "numpy conversion of a GPU tensor")
            if name in ("pin_memory",) and ins and tag_of(ins[0]) == "gpu":
                report("pin", name)
            # --- mixed devices
            if len(tags) == 2 and name not in TRANSFER:
                self_tag = tag_of(args[0]) if args and isinstance(args[0], torch.Tensor) else None
                if name == "__getitem__" and self_tag == "gpu":
                    pass  # a GPU tensor may be indexed by a CPU tensor
                else:
                    dims = ",".join(f"{tag_of(t)}{tuple(t.shape)}" for t in ins if tag_of(t))
                    scalar = all(t.dim() == 0 for t in ins if tag_of(t) == "cpu")
                    report("mixed-0dim" if scalar else "MIXED", name, dims)

            out = func(*args, **kwargs)

            if target is None:
                if "gpu" in tags:
                    target = "gpu"
                elif "cpu" in tags:
                    target = "cpu"
                elif not ins and name in FACTORIES:
                    target = "cpu"
            in_place = name.endswith("_") and not name.endswith("__") or name.startswith("__i") or name == "__setitem__"
            if in_place and args and isinstance(args[0], torch.Tensor):
                target = tag_of(args[0]) or target  # the destination keeps its device
            if target is not None:
                if isinstance(out, torch.Tensor) and any(out is t for t in ins) and tag_of(out) not in (None, target) and name in ("to", "cpu", "type"):
                    out = out.clone()  # a real transfer returns a new tensor
                for t in tensors(out, []):
                    if any(t is i for i in ins) and tag_of(t) is not None and not (name in ("to", "cpu") and t is out):
                        continue
                    set_tag(t, target)
            return out

    VDev().__enter__()

