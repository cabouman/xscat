"""Optional interactive viewing with ``mbirjax.slice_viewer`` (skipped when mbirjax is not installed)."""
import os


def slice_viewer(*arrays, title=None, slice_label=None, block=True, **kwargs):
    """Show volumes side by side in ``mbirjax.slice_viewer`` if available; otherwise print a note and return."""
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.05')
    try:
        import mbirjax as mj
    except ImportError:
        print('[viewer] mbirjax is not installed; the PNG figures are the record of this run')
        return False
    opts = {}
    if title is not None:
        opts['title'] = title
    if slice_label is not None:
        opts['slice_label'] = slice_label
    opts.update(kwargs)
    try:
        import matplotlib.pyplot as plt
        mj.slice_viewer(*arrays, **opts)
        plt.show(block=block)
    except Exception as exc:
        print(f'[viewer] mbirjax.slice_viewer failed: {exc}')
        return False
    return True
