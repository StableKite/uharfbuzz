# Free-threaded CPython

The extension supports free-threaded CPython without `PYTHON_GIL` or `-X gil`
overrides. It uses that interpreter's full ABI; ordinary CPython keeps the
Limited API default. Both extension modules declare free-threading support.
Vendored HarfBuzz is built with its native multithreading support enabled.

Mutable native objects are synchronized for each complete operation, including
copying borrowed glyph arrays into Python values. Fonts include their faces,
parents and function tables in the synchronization graph. Subset plans include
their inputs and faces. Borrowed set/map wrappers share their native owner's
state; iterators additionally protect their cursor and remain exhausted.

Ownership of a whole graph is reserved atomically. The bookkeeping condition
is released before calling HarfBuzz or Python callbacks. Independent graphs
can execute concurrently; operations sharing a graph serialize. A shared face
or function table therefore also serializes its dependent fonts. A sequence
of separate calls is not a transaction: callers needing, for example, an
atomic buffer clear/add/shape sequence must synchronize that sequence or use
separate buffers.

Callbacks can read the active objects. Mutating an active object raises
`RuntimeError`, rather than invalidating pointers still in use by HarfBuzz.
Contended nested access also raises instead of waiting in a lock cycle. This
includes a callback waiting for a worker accessing the same graph: the worker
receives `RuntimeError`. Callback exceptions are re-raised by the surrounding
Python operation after HarfBuzz returns and its locks are released.

Python callback data, parent fonts and source faces are retained while the
native objects can reference them. Face table bytes and paint color lines are
copied before their callback-only storage expires. Caller-owned feature,
variation, codepoint, shaper and repacking collections are snapshotted before
their lengths or data are used for native allocations.

An arbitrary Python callback must synchronize its own external shared state.
Native capsule callbacks and their external buffers must themselves support
concurrent use; uharfbuzz retains their capsules and userdata owners but cannot
make arbitrary native code thread-safe.

`tests/test_threading.py` checks natural GIL state in a fresh interpreter,
overlapping independent callbacks, shared objects and aliases, ownership,
callback reentry and exceptions, mutable inputs and persistent paint data.
Run it together with the existing suite, without GIL overrides:

```sh
python -m pytest tests
```
