import io
import os
import re
import sys
import warnings
from contextlib import contextmanager, redirect_stdout


def suppress_spotpy_syntax_warnings() -> None:
    """Hide known Python 3.12 invalid-escape warnings emitted by spotpy."""
    warnings.filterwarnings(
        "ignore",
        message=r"invalid escape sequence '\\[a-zA-Z]'",
        category=SyntaxWarning,
        module=r"spotpy\.(analyser|likelihoods)",
    )

class _LineFilteringStream(io.TextIOBase):
    def __init__(self, wrapped, drop_patterns: list[re.Pattern[str]]):
        self._wrapped = wrapped
        self._drop_patterns = drop_patterns
        self._buffer = ""

    def writable(self):
        return True

    def write(self, s):
        if not s:
            return 0

        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if any(p.search(line) for p in self._drop_patterns):
                continue
            self._wrapped.write(f"{line}\n")
        return len(s)

    def flush(self):
        if self._buffer:
            line = self._buffer
            self._buffer = ""
            if not any(p.search(line) for p in self._drop_patterns):
                self._wrapped.write(line)
        self._wrapped.flush()


@contextmanager
def spotpy_stdout_control(*, rank: int, execution_mode: str):
    """
    Reduce SPOTPY print noise in MPI runs without hiding useful progress output.
    - Non-root ranks: suppress all stdout (prevents interleaved clutter).
    - Root rank: filter only the known 'Initializing...' chatter.
    """
    if execution_mode == "serial":
        yield
        return

    if rank != 0:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            yield
        return

    drop_patterns = [
        re.compile(r"^\s*Initializing the\b"),
        re.compile(r"^\s*The objective function will be\b"),
        re.compile(r"^\s*Initialize database\.\.\.\s*$"),
    ]
    stream = _LineFilteringStream(sys.stdout, drop_patterns)
    with redirect_stdout(stream):
        yield
    stream.flush()
