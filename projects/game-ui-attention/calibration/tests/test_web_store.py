"""store.py 单测（仅 stdlib）：图库/retire/快照式发号/shuffle/JSONL append/删除权/回传校验。"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if str(_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_DIR))

import hashing  # noqa: E402
import store as store_mod  # noqa: E402


def _make_store(tmp_path: Path, **kw) -> store_mod.Storage:
    return store_mod.Storage(
        tmp_path / "wc",
        tmp_path / "wi",
        duration_seconds=kw.get("duration_seconds", 3.0),
        seed_strategy=kw.get("seed_strategy", hashing.SEED_STRATEGY_PER_SESSION),
        global_seed=kw.get("global_seed"),
        client_code_hash="c" * 64,
        server_code_hash="d" * 64,
    )


def _add(st: store_mod.Storage, sha: str, source: str = "a.png", w: int = 64, h: int = 48) -> None:
    st.add_task(sha, sha + ".png", source, w, h, "unknown")


# --------------------------------------------------------------------------- tasks


def test_add_and_list_tasks(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    _add(st, "a" * 64)
    _add(st, "b" * 64)
    tasks = st.list_tasks()
    assert len(tasks) == 2
    assert {t["sha256"] for t in tasks} == {"a" * 64, "b" * 64}
    assert all(t["active"] == 1 for t in tasks)


def test_add_task_idempotent_keeps_source_name(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    r1 = st.add_task("a" * 64, "a" * 64 + ".png", "first.png", 64, 48, "unknown")
    r2 = st.add_task("a" * 64, "a" * 64 + ".png", "second.png", 64, 48, "reward")
    assert r1["already_existed"] is False
    assert r2["already_existed"] is True
    assert r2["source_name"] == "first.png"  # 不覆盖首次 source_name（study_id 稳定）
    t = st.get_task("a" * 64)
    assert t["screen_type"] == "reward"  # 更新 screen_type
    assert t["active"] == 1


def test_retire_soft_delete_keeps_history(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    _add(st, "a" * 64)
    assert st.retire_task("a" * 64) is True
    assert st.retire_task("a" * 64) is False  # 幂等（已非 active 视为未命中）
    assert st.active_images() == []
    assert st.get_task("a" * 64)["active"] == 0  # 历史仍可解析
    assert "a" * 64 in st.all_sha256s()  # sha256 引用完整


# --------------------------------------------------------------------------- sessions


def test_create_session_requires_images(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    with pytest.raises(ValueError):
        st.create_session()


def test_create_session_snapshot_semantics(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    _add(st, "a" * 64)
    _add(st, "b" * 64)
    s1 = st.create_session()
    assert len(s1["images"]) == 2

    _add(st, "c" * 64)  # 加稿不影响已发会话
    s2 = st.create_session()
    assert len(s2["images"]) == 3

    reloaded = st.get_session(s1["session_id"])
    assert reloaded is not None
    assert [im["sha256"] for im in reloaded["images"]] == [im["sha256"] for im in s1["images"]]  # 冻结
    assert len(reloaded["images"]) == 2


def test_shuffle_deterministic_given_global_seed(tmp_path: Path) -> None:
    st = _make_store(tmp_path, seed_strategy="global:42", global_seed=42)
    for i in range(5):
        _add(st, f"{i:064d}")
    s1 = st.create_session()
    s2 = st.create_session()
    assert [im["sha256"] for im in s1["images"]] == [im["sha256"] for im in s2["images"]]
    assert s1["shuffle_seed"] == s2["shuffle_seed"] == 42
    assert s1["config_hash"] == s2["config_hash"]  # 同种子策略+同图集合+同代码+同时长 → 同 config_hash


def test_shuffle_is_permutation(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    shas = [f"{i:064d}" for i in range(6)]
    for s in shas:
        _add(st, s)
    cfg = st.create_session()
    got = [im["sha256"] for im in cfg["images"]]
    assert sorted(got) == sorted(shas)  # 顺序打乱但集合不变
    assert isinstance(cfg["shuffle_seed"], int)


def test_session_id_unpredictable(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    _add(st, "a" * 64)
    s1 = st.create_session()
    s2 = st.create_session()
    assert s1["session_id"] != s2["session_id"]
    assert len(s1["session_id"]) >= 40


def test_session_config_writes_config_hash(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    _add(st, "a" * 64)
    cfg = st.create_session()
    assert len(cfg["config_hash"]) == 64
    assert cfg["study_id"]
    assert cfg["duration_seconds"] == 3.0
    assert cfg["schema_version"] == hashing.SESSION_SCHEMA_VERSION


# --------------------------------------------------------------------------- responses


def test_append_and_read_responses(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    r1 = st.append_response({"participant_id": "p1", "_server": {"accepted": True}})
    r2 = st.append_response({"participant_id": "p2", "_server": {"accepted": True}})
    assert r1 != r2
    rows = st.read_responses()
    assert [r["participant_id"] for r in rows] == ["p1", "p2"]


def test_concurrent_appends_no_loss(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    n_threads, per_thread = 8, 25
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            st.append_response({"participant_id": f"p{tid}", "i": i, "_server": {"accepted": True}})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = st.read_responses()
    assert len(rows) == n_threads * per_thread  # 不丢行


def test_delete_participant_removes_all_rows(tmp_path: Path) -> None:
    st = _make_store(tmp_path)
    st.append_response({"participant_id": "p1"})
    st.append_response({"participant_id": "p2"})
    st.append_response({"participant_id": "p1"})
    removed = st.delete_participant("p1")
    assert removed == 2
    assert [r["participant_id"] for r in st.read_responses()] == ["p2"]
    assert st.delete_participant("nobody") == 0


# --------------------------------------------------------------------------- validate_response


def _session(images: list[dict]) -> dict:
    return {"schema_version": hashing.SESSION_SCHEMA_VERSION, "images": images}


def _dims(shas: list[str]) -> dict:
    return {s: (64, 48) for s in shas}


def _entry(sha: str, **kw) -> dict:
    e = {
        "sha256": sha,
        "responded": True,
        "first_look": {"x": 32, "y": 24},
        "boxes": [{"x": 8, "y": 8, "width": 16, "height": 16}],
    }
    e.update(kw)
    return e


def _body(completed: bool, entries: list[dict]) -> dict:
    return {
        "schema_version": hashing.RESPONSE_SCHEMA_VERSION,
        "participant_id": "p1",
        "completed": completed,
        "images": entries,
    }


def test_validate_accepts_complete_response() -> None:
    sha = "a" * 64
    ok, reason = store_mod.validate_response(
        _body(True, [_entry(sha)]), _session([{"sha256": sha}]), {sha}, _dims([sha])
    )
    assert ok is True and reason is None


def test_validate_accepts_incomplete_response() -> None:
    sha = "a" * 64
    ok, reason = store_mod.validate_response(
        _body(False, [_entry(sha, responded=False, first_look=None, boxes=[])]),
        _session([{"sha256": sha}]),
        {sha},
        _dims([sha]),
    )
    assert ok is True  # completed=false 属合法“不完整”，由 consistency.py 排除


def test_validate_rejects_schema_mismatch() -> None:
    sha = "a" * 64
    body = _body(True, [_entry(sha)])
    body["schema_version"] = "wrong"
    ok, reason = store_mod.validate_response(body, _session([{"sha256": sha}]), {sha}, _dims([sha]))
    assert ok is False and "schema_version" in reason


def test_validate_rejects_unknown_sha() -> None:
    sha = "a" * 64
    ok, reason = store_mod.validate_response(
        _body(True, [_entry(sha)]), _session([{"sha256": sha}]), {"b" * 64}, _dims([sha])
    )
    assert ok is False and "sha256" in reason


def test_validate_rejects_out_of_bounds_point() -> None:
    sha = "a" * 64
    ok, reason = store_mod.validate_response(
        _body(True, [_entry(sha, first_look={"x": 999, "y": 999})]),
        _session([{"sha256": sha}]),
        {sha},
        _dims([sha]),
    )
    assert ok is False and "越界" in reason


def test_validate_rejects_completed_with_wrong_count() -> None:
    sha = "a" * 64
    ok, reason = store_mod.validate_response(
        _body(True, [_entry(sha)]), _session([{"sha256": sha}, {"sha256": "b" * 64}]), {sha}, _dims([sha])
    )
    assert ok is False and "图数" in reason


def test_validate_rejects_completed_with_unanswered() -> None:
    sha = "a" * 64
    ok, reason = store_mod.validate_response(
        _body(True, [_entry(sha, responded=False, first_look=None, boxes=[])]),
        _session([{"sha256": sha}]),
        {sha},
        _dims([sha]),
    )
    assert ok is False and "未作答" in reason
