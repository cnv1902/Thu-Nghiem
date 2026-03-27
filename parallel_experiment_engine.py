from importlib import import_module
from typing import Any, Dict, List

from joblib import Parallel, delayed

from gpu_backend import cleanup_gpu_memory


Task = Dict[str, Any]


def _resolve_symbol(dotted_path: str):
    """
    Resolve callable from dotted path: 'module.submodule:function_name'.
    """
    if ":" not in dotted_path:
        raise ValueError("method_path must use format 'module.submodule:function_name'")

    module_name, symbol_name = dotted_path.split(":", 1)
    module = import_module(module_name)
    if not hasattr(module, symbol_name):
        raise AttributeError(f"{symbol_name} not found in {module_name}")
    return getattr(module, symbol_name)


def execute_training_task(task: Task) -> Dict[str, Any]:
    """
    Worker-safe function (top-level) for joblib loky backend.

    Expected task keys:
    - method_path: dotted path to top-level callable
    - method_kwargs: kwargs for the method
    - metadata: any serializable metadata to keep with result
    - cleanup_gpu: bool
    """
    method_path = task["method_path"]
    method_kwargs = task.get("method_kwargs", {})
    metadata = task.get("metadata", {})
    do_cleanup = task.get("cleanup_gpu", True)

    fn = _resolve_symbol(method_path)

    try:
        y_pred, alpha = fn(**method_kwargs)
        return {
            "ok": True,
            "y_pred": y_pred,
            "alpha": alpha,
            "error": None,
            "metadata": metadata,
        }
    except Exception as ex:
        return {
            "ok": False,
            "y_pred": None,
            "alpha": None,
            "error": str(ex),
            "metadata": metadata,
        }
    finally:
        if do_cleanup:
            cleanup_gpu_memory()


def run_tasks_parallel(
    tasks: List[Task],
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run a list of independent training tasks using process-based parallelism.
    """
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(execute_training_task)(task) for task in tasks
    )
