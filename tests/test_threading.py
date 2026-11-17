"""Native concurrency/lifetime regressions; never force the interpreter's GIL off."""

from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import threading
import weakref

import pytest
import uharfbuzz as hb


FONT = Path(__file__).parent / "data/OpenSans.subset.ttf"


def font():
    return hb.Font(hb.Face(FONT.read_bytes()))


def parallel(function, workers=8):
    barrier = threading.Barrier(workers)

    def run(index):
        barrier.wait(timeout=10)
        return function(index)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run, index) for index in range(workers)]
        return [future.result(timeout=30) for future in futures]


@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="requires FT interpreter")
def test_import_preserves_naturally_disabled_gil():
    env = os.environ.copy()
    env.pop("PYTHON_GIL", None)
    script = """
import sys
assert 'gil' not in sys._xoptions
assert not sys._is_gil_enabled()
import uharfbuzz
import uharfbuzz._harfbuzz_test
assert not sys._is_gil_enabled()
f = uharfbuzz.Font(uharfbuzz.Face(open(sys.argv[1], 'rb').read()))
b = uharfbuzz.Buffer(); b.add_str('test'); b.guess_segment_properties()
uharfbuzz.shape(f, b)
assert b.glyph_infos
assert not sys._is_gil_enabled()
"""
    subprocess.run([sys.executable, "-Werror", "-c", script, str(FONT)], env=env, check=True, timeout=30)


def test_independent_native_calls_overlap_in_callbacks():
    # This barrier cannot complete if a module-wide lock replaces the GIL.
    barrier = threading.Barrier(4)

    def run(index):
        f = font()
        funcs = hb.FontFuncs()

        def advance(f, glyph, data):
            if sysconfig.get_config_var("Py_GIL_DISABLED"):
                assert not sys._is_gil_enabled()
            barrier.wait(timeout=10)
            return index + 100

        funcs.set_glyph_h_advance_func(advance)
        f.funcs = funcs
        return f.get_glyph_h_advance(1)

    assert parallel(run, workers=4) == [100, 101, 102, 103]


def test_shared_buffer_readers_and_writers():
    b = hb.Buffer()
    f = font()

    def run(index):
        for _ in range(400):
            if index % 2:
                b.clear_contents()
                b.add_str("abc")
            else:
                infos = b.glyph_infos
                positions = b.glyph_positions
                assert all(isinstance(info.codepoint, int) for info in infos)
                assert positions is None or all(isinstance(pos.x_advance, int) for pos in positions)

    parallel(run)
    b.clear_contents()
    b.add_str("abc")
    b.guess_segment_properties()
    hb.shape(f, b)
    assert len(b.glyph_infos) == 3


def test_shared_sets_maps_aliases_and_iterators():
    input = hb.SubsetInput()
    aliases = [input.unicode_set for _ in range(8)]
    mapping = hb.Map()

    def run(index):
        for value in range(index * 40, (index + 1) * 40):
            aliases[index].add(value)
            mapping[value] = value + 1
            list(aliases[index])
            list(mapping.items())

    parallel(run)
    assert set(input.unicode_set) == set(range(320))
    assert dict(mapping.items()) == {value: value + 1 for value in range(320)}
    iterator = iter(input.unicode_set)

    def consume(_):
        return list(iterator)

    results = parallel(consume)
    assert sorted(value for result in results for value in result) == list(range(320))


def test_shared_font_mutation_and_shaping():
    f = font()

    def run(index):
        b = hb.Buffer()
        for _ in range(100):
            if index % 2:
                f.scale = (1000 + index, 1000 + index)
                f.set_variations({})
            else:
                b.clear_contents()
                b.add_str("test")
                b.guess_segment_properties()
                hb.shape(f, b)
                assert len(b.glyph_infos) == 4

    parallel(run)


def test_callback_userdata_and_parent_are_retained():
    class Data:
        value = 471

    data = Data()
    alive = weakref.ref(data)
    funcs = hb.FontFuncs()
    funcs.set_glyph_h_advance_func(lambda f, g, d: d.value, data)
    parent = font()
    parent.funcs = funcs
    child = hb.Font(parent)
    del data, funcs, parent
    gc.collect()
    assert alive() is not None  # Check before invoking a formerly dangling pointer.
    assert child.get_glyph_h_advance(1) == 471
    del child
    gc.collect()
    assert alive() is None


def test_face_table_callback_owns_data_and_returned_bytes():
    original = hb.Face(FONT.read_bytes())

    class Data:
        pass

    data = Data()
    alive = weakref.ref(data)

    def table(face, tag, userdata):
        assert userdata is alive()
        return bytes(bytearray(original.reference_table(tag).data))

    face = hb.Face.create_for_tables(table, data)
    del data
    gc.collect()
    assert alive() is not None
    blob = face.reference_table("head")
    expected = original.reference_table("head").data
    for _ in range(1000):
        bytes(bytearray(b"x" * len(expected)))
    assert blob.data == expected
    f = hb.Font(face)
    del face
    gc.collect()
    assert f.get_nominal_glyph(ord("A")) is not None
    del f, blob
    gc.collect()
    assert alive() is None


def test_callback_can_read_but_cannot_mutate_active_buffer():
    f = font()
    b = hb.Buffer()
    b.add_str("test")
    b.guess_segment_properties()
    seen = []

    def message(text):
        seen.append(len(b.glyph_infos))
        with pytest.raises(RuntimeError, match="cannot mutate"):
            b.clear_contents()
        return True

    b.set_message_func(message)
    hb.shape(f, b)
    assert seen and len(b.glyph_infos) == 4


def test_callback_exception_propagates_and_releases_locks():
    f = font()
    funcs = hb.FontFuncs()

    def advance(*args):
        raise ValueError("callback failure")

    funcs.set_glyph_h_advance_func(advance)
    f.funcs = funcs
    with pytest.raises(ValueError, match="callback failure"):
        f.get_glyph_h_advance(1)
    funcs.set_glyph_h_advance_func(lambda *args: 123)
    assert f.get_glyph_h_advance(1) == 123


def test_callback_worker_on_same_object_fails_without_deadlock():
    f = font()
    funcs = hb.FontFuncs()
    with ThreadPoolExecutor(max_workers=1) as executor:
        def advance(*args):
            future = executor.submit(lambda: f.scale)
            with pytest.raises(RuntimeError, match="deadlock"):
                future.result(timeout=10)
            return 123

        funcs.set_glyph_h_advance_func(advance)
        f.funcs = funcs
        assert f.get_glyph_h_advance(1) == 123


def test_failed_conversion_does_not_poison_object():
    b = hb.Buffer()
    with pytest.raises((TypeError, OverflowError)):
        b.add_codepoints([ord("a"), object()])
    b.add_str("test")
    assert len(b) == 4


def test_subset_plan_retains_owners_and_returns_stable_mapping():
    f = hb.Face(FONT.read_bytes())
    input = hb.SubsetInput()
    input.unicode_set.update([ord("a"), ord("b")])
    plan = hb.SubsetPlan(f, input)
    del f, input
    gc.collect()
    result = plan.execute()
    mapping = plan.old_to_new_glyph_mapping
    del plan
    gc.collect()
    assert result.glyph_count > 0 and len(mapping) > 0


def test_binary_operations_reserve_both_graphs_without_lock_order_cycles():
    a, b = hb.Set(range(50)), hb.Set(range(50, 100))
    x, y = hb.Map({1: 10}), hb.Map({2: 20})

    def run(index):
        for _ in range(200):
            if index % 2:
                a.update(b)
                x.update(y)
            else:
                b.update(a)
                y.update(x)

    parallel(run)
    assert set(a) == set(b) == set(range(100))
    assert dict(x.items()) == dict(y.items()) == {1: 10, 2: 20}


def test_exhausted_iterators_stay_exhausted():
    for iterable in (hb.Set([1]), hb.Map({1: 2}).items()):
        iterator = iter(iterable)
        next(iterator)
        for _ in range(3):
            with pytest.raises(StopIteration):
                next(iterator)


def test_callback_cannot_replace_active_font_functions():
    f = font()
    funcs = hb.FontFuncs()

    def advance(*args):
        with pytest.raises(RuntimeError, match="cannot mutate"):
            funcs.set_glyph_h_advance_func(lambda *args: 0)
        with pytest.raises(RuntimeError, match="cannot mutate"):
            f.funcs = hb.FontFuncs()
        return 321

    funcs.set_glyph_h_advance_func(advance)
    f.funcs = funcs
    assert f.get_glyph_h_advance(1) == 321


def test_subset_input_alias_cannot_mutate_active_plan_construction():
    original = hb.Face(FONT.read_bytes())
    input = hb.SubsetInput()
    alias = input.unicode_set
    alias.add(ord("A"))
    seen = []

    def table(face, tag, data):
        with pytest.raises(RuntimeError, match="cannot mutate"):
            alias.clear()
        seen.append(tag)
        return original.reference_table(tag).data

    face = hb.Face.create_for_tables(table, None)
    plan = hb.SubsetPlan(face, input)
    assert seen and plan.execute().glyph_count > 0


def test_gradient_snapshot_survives_native_callback():
    path = FONT.parent / "test_glyphs-glyf_colr_1.ttf"
    f = hb.Font(hb.Face(path.read_bytes()))
    funcs = hb.PaintFuncs()
    retained = []

    def gradient(line, *args):
        retained.append((line, line.color_stops, line.extend))

    funcs.set_linear_gradient_func(gradient)
    funcs.set_radial_gradient_func(gradient)
    funcs.set_sweep_gradient_func(gradient)
    for glyph in (6, 10, 92, 106, 116, 123, 154, 165, 175):
        f.paint_glyph(glyph, funcs)
    assert retained
    del funcs, f
    gc.collect()
    for line, stops, extend in retained:
        assert line.color_stops == stops and line.extend == extend
        line.color_stops.clear()
        assert line.color_stops == stops


def test_glyph_name_callback_copies_bytes_with_a_bound():
    f = font()
    funcs = hb.FontFuncs()
    f.funcs = funcs
    funcs.set_glyph_name_func(lambda *args: "test")
    assert f.get_glyph_name(1) == "test"
    funcs.set_glyph_name_func(lambda *args: "x" * 100)
    assert f.get_glyph_name(1) is None


def test_shape_and_variation_input_snapshots_under_concurrent_edits():
    features = {"kern": [(0, 4, 1)]}
    variations = {"wght": 400.0}

    def run(index):
        f = font()
        b = hb.Buffer()
        for _ in range(100):
            if index % 2:
                features["kern"].clear()
                features["kern"].extend([(0, 4, 1), (0, 2, 0)])
                variations.clear()
                variations.update(wght=400.0, wdth=100.0)
            else:
                f.set_variations(variations)
                b.clear_contents()
                b.add_str("test")
                b.guess_segment_properties()
                hb.shape(f, b, features, ["ot", "fallback"])
                assert len(b.glyph_infos) == 4

    parallel(run)
