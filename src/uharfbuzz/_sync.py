"""Synchronize native object graphs, without serializing independent graphs.

The condition protects only ownership bookkeeping, never a HarfBuzz call.
Aliases share a State; fonts and plans depend on their native owners' states.
Callbacks may read their objects, but cannot mutate an active native operation.
Contended reentry fails instead of waiting in a lock cycle (including callbacks
that wait for a worker using the same object).
"""

from threading import Condition, get_ident, local


_condition = Condition()
_local = local()
_callbacks = {}


class State:
    def __init__(self):
        self.dependencies = ()
        self.owner = None
        self.depth = 0

    def depend_on(self, *states):
        with _condition:
            self.dependencies = tuple(state for state in states if state is not self)


def _closure(states):
    result = set()
    pending = list(states)
    while pending:
        state = pending.pop()
        if state not in result:
            result.add(state)
            pending.extend(state.dependencies)
    return result


class locked:
    def __init__(self, *states, write=()):
        self.roots = states
        self.write = write
        self.error = None

    def __enter__(self):
        ident = get_ident()
        stack = getattr(_local, "stack", ())
        with _condition:
            while True:
                states = _closure(self.roots)
                if any(state.owner == ident for state in self.write):
                    raise RuntimeError("cannot mutate an object during an active HarfBuzz operation")
                conflicts = {state.owner for state in states if state.owner not in (None, ident)}
                if not conflicts:
                    break
                if stack or any(_callbacks.get(owner) for owner in conflicts):
                    raise RuntimeError("contended HarfBuzz callback access would deadlock")
                _condition.wait()
            self.states = states
            for state in states:
                state.owner = ident
                state.depth += 1
        _local.stack = (*stack, self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _local.stack = _local.stack[:-1]
        with _condition:
            for state in self.states:
                state.depth -= 1
                if not state.depth:
                    state.owner = None
            _condition.notify_all()
        if exc_type is None and self.error is not None:
            raise self.error


class callback:
    """Capture exceptions across HarfBuzz's noexcept callback boundary."""

    def __enter__(self):
        ident = get_ident()
        with _condition:
            _callbacks[ident] = _callbacks.get(ident, 0) + 1
            _condition.notify_all()

    def __exit__(self, exc_type, exc, tb):
        ident = get_ident()
        with _condition:
            _callbacks[ident] -= 1
            if not _callbacks[ident]:
                del _callbacks[ident]
        if exc_type is not None:
            stack = getattr(_local, "stack", ())
            if stack:
                if stack[-1].error is None:
                    stack[-1].error = exc
                return True
