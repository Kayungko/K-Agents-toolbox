"""metrics.eval.ueyes_driver 合成数据自测（L2 G2 任务包①：驱动全流程用合成数据验证）。

合成数据集只能证明计算链正确，不能证明人会看哪里；UEyes 真实数据到达后由
runner 实跑（md5 核对 → 冻结划分 → CB → test×3 窗口 → 表 A-E → S0-S3）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures.synthetic_ueyes import image_to_screen_norm, make_ueyes_dataset  # noqa: E402
from ui_attention.metrics.eval import ueyes_driver as drv  # noqa: E402
from ui_attention.metrics.eval.groundtruth import FixationSet  # noqa: E402


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ueyes-syn") / "extracted"
    return make_ueyes_dataset(root)


# ---------------------------------------------------------------------------
# Part 1：布局发现与 md5
# ---------------------------------------------------------------------------


def test_discover_layout_ok(dataset):
    layout = drv.discover_layout(dataset.root)
    assert layout.info_csv.is_file()
    assert layout.images_dir.is_dir() and layout.logs_dir.is_dir()


def test_discover_layout_missing_dirs(tmp_path):
    with pytest.raises(drv.DatasetError) as exc:
        drv.discover_layout(tmp_path / "nope")
    assert exc.value.exit_code == 5
    (tmp_path / "images").mkdir()
    with pytest.raises(drv.DatasetError):
        drv.discover_layout(tmp_path)


# ---------------------------------------------------------------------------
# 真实 UEyes 格式适配（L2 勘误：image_types.csv、分隔符 ';'、TIME 列带时间戳后缀）
# ---------------------------------------------------------------------------


def test_discover_layout_accepts_image_types_csv(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "eyetracker_logs").mkdir()
    (tmp_path / "image_types.csv").write_text(
        "Image Name;Category;Block;Train/Test\na.png;web;0;Train\n", encoding="utf-8"
    )
    layout = drv.discover_layout(tmp_path)
    assert layout.info_csv.name == "image_types.csv"


def test_load_info_table_real_ueyes_format(tmp_path):
    p = tmp_path / "image_types.csv"
    p.write_text(
        "Image Name;Category;Block;Train/Test\n2021b7.png;desktop;0;Train\n460bf9.png;desktop;0;Test\n",
        encoding="utf-8",
    )
    t = drv.load_info_table(p)
    assert t.delimiter == ";"
    assert t.source_file == "image_types.csv"
    assert t.n_rows == 2
    assert t.split_counts == {"train": 1, "test": 1}
    assert t.metas[0].image_id == "2021b7.png"
    assert t.metas[0].official_split == "train"
    assert t.metas[1].category == "desktop"
    # 逗号格式向后兼容（合成夹具路径）
    p2 = tmp_path / "info.csv"
    p2.write_text("image,category,block,split\nx.png,web,1,test\n", encoding="utf-8")
    t2 = drv.load_info_table(p2)
    assert t2.delimiter == "," and t2.metas[0].official_split == "test"


def test_gazepoint_time_column_with_timestamp_suffix():
    cols = drv._resolve_gazepoint_columns(
        [
            "MEDIA_ID",
            "MEDIA_NAME",
            "CNT",
            "TIME(2022/03/28 14:28:03.787)",
            "TIMETICK(f=10000000)",
            "FPOGD",
            "BPOGX",
            "BPOGY",
            "BPOGV",
            "",
        ]
    )
    assert cols["TIME"] == "TIME(2022/03/28 14:28:03.787)"  # 不得误取 TIMETICK
    assert cols["MEDIA_NAME"] == "MEDIA_NAME"
    assert cols["BPOGV"] == "BPOGV"
    with pytest.raises(drv.DatasetError):
        drv._resolve_gazepoint_columns(["MEDIA_NAME", "TIMETICK(f=10000000)", "FPOGD", "BPOGX", "BPOGY", "BPOGV"])


def test_load_eval_image_alpha_rule(tmp_path):
    """评估复现路径：RGBA 丢 alpha 不合成（官方 cv2 口径）；模式与透明度事实进审计信息。"""
    sys.path.insert(0, str(_TESTS))
    from fixtures import synthetic

    # 半透明 RGBA：保留存储 RGB 值（R=200），不做黑/白合成
    p = synthetic.rgba_png(tmp_path / "alpha.png", width=16, height=12)
    arr, info = drv.load_eval_image(p)
    assert arr.shape == (12, 16, 3) and arr.dtype == np.uint8
    assert info["mode"] == "RGBA" and info["has_alpha"] is True
    assert info["non_opaque_alpha"] is True  # alpha=128 < 255
    assert arr[0, 0, 0] == 200  # 存储值保留，未被合成改写

    # 全不透明 RGBA：无损丢 alpha
    p2 = synthetic.opaque_alpha_png(tmp_path / "opaque.png", width=8, height=8)
    arr2, info2 = drv.load_eval_image(p2)
    assert info2["non_opaque_alpha"] is False
    assert arr2.shape == (8, 8, 3)

    # RGB 直通
    p3 = synthetic.solid_color_png(tmp_path / "rgb.png", width=8, height=6, color=(1, 2, 3))
    arr3, info3 = drv.load_eval_image(p3)
    assert info3["mode"] == "RGB" and info3["has_alpha"] is False
    assert tuple(arr3[0, 0]) == (1, 2, 3)


def test_md5_and_zip_verification(tmp_path):
    z = tmp_path / "fake.zip"
    z.write_bytes(b"ueyes-fake-bytes")
    actual = drv.md5_file(z)
    assert len(actual) == 32
    result = drv.verify_zip_md5(z, expected=actual)
    assert result["ok"] is True
    with pytest.raises(drv.DatasetError) as exc:
        drv.verify_zip_md5(z, expected="0" * 32)
    assert "md5" in exc.value.message
    assert drv.UEYES_ZIP_MD5 == "c2d53e6af0a47e1f459416d6839ec2c1"  # Zenodo 记录值冻结


# ---------------------------------------------------------------------------
# Part 2：info.csv 与划分冻结
# ---------------------------------------------------------------------------


def test_load_info_csv_official(dataset):
    layout = drv.discover_layout(dataset.root)
    metas = drv.load_info_csv(layout.info_csv)
    assert len(metas) == 6
    by_id = {m.image_id: m for m in metas}
    assert by_id["img_0001.png"].category == "webpage"
    assert by_id["img_0001.png"].official_split == "train"
    assert by_id["img_0003.png"].official_split == "test"


def test_load_info_csv_without_split_column(tmp_path):
    ds = make_ueyes_dataset(
        tmp_path / "nosplit",
        include_split_column=False,
        n_participants=1,
        per_image_fixations=5,
        with_near_duplicate=False,
    )
    metas = drv.load_info_csv(ds.root / "info.csv")
    assert all(m.official_split is None for m in metas)


def test_discover_image_paths_fills_sizes(dataset):
    layout = drv.discover_layout(dataset.root)
    metas = drv.discover_image_paths(layout, drv.load_info_csv(layout.info_csv))
    sizes = {m.image_id: (m.width, m.height) for m in metas}
    assert sizes["img_0001.png"] == (320, 200)
    assert sizes["img_0003.png"] == (96, 160)
    assert all(m.rel_path for m in metas)


def test_splits_freeze_official_roundtrip_and_tamper(dataset, tmp_path):
    layout = drv.discover_layout(dataset.root)
    metas = drv.discover_image_paths(layout, drv.load_info_csv(layout.info_csv))
    splits = drv.SplitsFile.build(metas, protocol_version="r2-v0.1")
    assert splits.method == "official_info_csv"
    assert sorted(splits.image_ids("train")) == ["img_0001.png", "img_0002.png", "img_0004.png"]
    assert sorted(splits.image_ids("test")) == ["img_0003.png", "img_0005.png", "img_0006.png"]
    out = tmp_path / "splits.v1.json"
    sha = splits.write(out)
    loaded = drv.SplitsFile.load(out, verify_sha256=sha)
    assert loaded.split_of("img_0003.png") == "test"
    # 篡改 → 哈希复核拒绝
    obj = json.loads(out.read_text(encoding="utf-8"))
    obj["images"][0]["split"] = "test"
    out.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(drv.DatasetError):
        drv.SplitsFile.load(out, verify_sha256=sha)


def test_splits_fallback_hash_deterministic(dataset):
    layout = drv.discover_layout(dataset.root)
    metas = drv.load_info_csv(layout.info_csv)
    metas_no_split = [drv.ImageMeta(m.image_id, m.category, m.block, None) for m in metas]
    s1 = drv.SplitsFile.build(metas_no_split, protocol_version="r2-v0.1")
    s2 = drv.SplitsFile.build(metas_no_split, protocol_version="r2-v0.1")
    assert s1.method == "fallback_hash"
    assert s1.entries == s2.entries  # 完全确定
    # §3.3 桶规则抽查
    assign = drv.fallback_hash_split([m.image_id for m in metas], "r2-v0.1")
    assert set(assign.values()) <= {"train", "val", "test"}
    with pytest.raises(drv.DatasetError):
        drv.SplitsFile.build(metas_no_split, protocol_version="r2-v0.1", allow_fallback=False)


def test_splits_file_bytes_deterministic_across_rebuilds(dataset, tmp_path):
    """冻结哈希跨重跑稳定：同内容 splits 文件字节一致（时间戳不入冻结文件）。"""
    layout = drv.discover_layout(dataset.root)
    metas = drv.discover_image_paths(layout, drv.load_info_csv(layout.info_csv))
    s1 = drv.SplitsFile.build(metas, protocol_version="r2-v0.1")
    s2 = drv.SplitsFile.build(metas, protocol_version="r2-v0.1")
    sha1 = s1.write(tmp_path / "a.json")
    sha2 = s2.write(tmp_path / "b.json")
    assert sha1 == sha2
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    # load 仍可用（generated_at_utc 不在文件内，字段回落空串）
    loaded = drv.SplitsFile.load(tmp_path / "a.json", verify_sha256=sha1)
    assert loaded.generated_at_utc == ""


def test_cb_save_bytes_deterministic(tmp_path):
    """CB npz 确定性字节：手工 zip 固定条目时间戳，同内容重存哈希一致。"""
    from ui_attention.metrics.eval.baselines import CenterBiasBaseline, CenterBiasFitInput
    from ui_attention.metrics.eval.groundtruth import FixationSet

    fix = FixationSet(points=np.array([[10.0, 10.0], [20.0, 30.0]] * 10), weights=np.ones(20))
    cb = CenterBiasBaseline.fit([CenterBiasFitInput(shape=(60, 80), fixations=fix)], source_split_hash="h" * 64)
    sha_a = cb.save(tmp_path / "cb_a.npz")
    sha_b = cb.save(tmp_path / "cb_b.npz")
    assert sha_a == sha_b
    assert (tmp_path / "cb_a.npz").read_bytes() == (tmp_path / "cb_b.npz").read_bytes()
    loaded, sha_l = CenterBiasBaseline.load(tmp_path / "cb_a.npz")
    assert sha_l == sha_a
    assert np.array_equal(loaded.histogram, cb.histogram)


# ---------------------------------------------------------------------------
# Part 3：坐标映射与真值重建
# ---------------------------------------------------------------------------


def test_screen_to_image_roundtrip():
    for w, h in ((320, 200), (160, 120), (96, 160), (240, 150)):
        for x, y in ((1.0, 1.0), (w / 2, h / 2), (w - 1.5, h - 1.5)):
            sx, sy = image_to_screen_norm(x, y, w, h)
            back = drv.screen_to_image_point(sx, sy, w, h)
            assert back is not None
            assert back[0] == pytest.approx(x, abs=1e-6)
            assert back[1] == pytest.approx(y, abs=1e-6)


def test_screen_to_image_oob():
    # 4:3 图 pillarbox：屏幕最左侧落在 pad 区 → 丢弃
    assert drv.screen_to_image_point(0.001, 0.5, 160, 120) is None
    assert drv.screen_to_image_point(0.999, 0.5, 160, 120) is None
    # 16:10 图无 pad：角落仍在图内
    assert drv.screen_to_image_point(0.001, 0.001, 320, 200) is not None


def test_build_groundtruth_windows_and_viewers(dataset):
    layout = drv.discover_layout(dataset.root)
    metas = drv.discover_image_paths(layout, drv.load_info_csv(layout.info_csv))
    gts = {}
    for window in ("1s", "3s", "7s"):
        gts[window] = drv.build_groundtruth(layout, metas, window)

    for img in dataset.images:
        iid = img["image_id"]
        exp = dataset.expected[iid]
        # 窗口前缀截断：嵌套关系 7s ⊇ 3s ⊇ 1s，计数与构造期望一致
        assert gts["1s"][iid].n_fix <= gts["3s"][iid].n_fix <= gts["7s"][iid].n_fix
        # 图内注视计数 == 构造期望（主注视 + 落在图内的角落点）
        assert gts["7s"][iid].n_fix == exp["7s"]
        assert gts["1s"][iid].n_fix == exp["1s"]
        # 越界丢弃单独计数（letterbox pad 区）
        assert gts["7s"][iid].n_dropped_oob == exp["oob"]
        assert gts["7s"][iid].n_viewers == exp["viewers"]
        # BPOGV=0 的行不计入；有效总数 = 图内 + 越界
        assert gts["7s"][iid].n_fix_valid == exp["7s"] + exp["oob"]


def test_build_groundtruth_duration_weighting(dataset):
    layout = drv.discover_layout(dataset.root)
    metas = drv.discover_image_paths(layout, drv.load_info_csv(layout.info_csv))
    gt_count = drv.build_groundtruth(layout, metas, "7s", weighting="count")
    gt_dur = drv.build_groundtruth(layout, metas, "7s", weighting="duration")
    iid = "img_0001.png"
    assert gt_count[iid].fixations.weights[0] == 1.0
    assert gt_dur[iid].fixations.weights[0] != 1.0  # FPOGD 时长加权
    assert len(gt_dur[iid].fixations) == len(gt_count[iid].fixations)


def test_build_groundtruth_unknown_window(dataset):
    layout = drv.discover_layout(dataset.root)
    metas = drv.discover_image_paths(layout, drv.load_info_csv(layout.info_csv))
    with pytest.raises(drv.DatasetError):
        drv.build_groundtruth(layout, metas, "5s")


# ---------------------------------------------------------------------------
# Part 4：dHash 近重复聚簇与泄漏审计
# ---------------------------------------------------------------------------


def _load_gray(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def test_dhash_clusters_near_duplicate(dataset):
    h1 = drv.dhash64(_load_gray(dataset.root / "images" / "block 1" / "img_0001.png"))
    h5 = drv.dhash64(_load_gray(dataset.root / "images" / "block 2" / "img_0005.png"))  # 近重复副本
    h3 = drv.dhash64(_load_gray(dataset.root / "images" / "block 1" / "img_0003.png"))
    assert drv.hamming64(h1, h5) <= 8
    assert drv.hamming64(h1, h3) > 8 or (320, 200) != (96, 160)  # 不同图通常距离大
    assert drv.dhash64(_load_gray(dataset.root / "images" / "block 1" / "img_0001.png")) == h1  # 确定性


def test_duplicate_clusters_and_cross_split_audit(dataset):
    hashes = {}
    for img in dataset.images:
        hashes[img["image_id"]] = drv.dhash64(_load_gray(Path(img["path"])))
    clusters = drv.duplicate_clusters(hashes, max_hamming=8)
    # img_0001 与 img_0005 是近重复且跨官方划分（train/test）→ 审计必须抓到
    pair = next((c for c in clusters if {"img_0001.png", "img_0005.png"} <= set(c)), None)
    assert pair is not None, f"近重复簇未检出：{clusters}"

    layout = drv.discover_layout(dataset.root)
    metas = drv.discover_image_paths(layout, drv.load_info_csv(layout.info_csv))
    splits = drv.SplitsFile.build(metas, protocol_version="r2-v0.1")
    sha = splits.write(dataset.root.parent / "splits.v1.json")
    splits_loaded = drv.SplitsFile.load(dataset.root.parent / "splits.v1.json", verify_sha256=sha)
    audit = drv.run_leakage_audit(
        splits=splits_loaded,
        recorded_splits_sha256=sha,
        hashes=hashes,
        max_hamming=8,
        cb=None,
        excluded_train=["img_x"],
        excluded_test=["img_y"],
    )
    assert audit.splits_sha256_ok is True
    assert audit.cross_split_violations >= 1
    assert any(set(c) >= {"img_0001.png", "img_0005.png"} for c in audit.clusters_cross_split)
    assert audit.excluded_train == ("img_x",) and audit.excluded_test == ("img_y",)
    assert "来源级泄漏" in audit.known_limitation


def test_audit_cb_rules():
    from ui_attention.metrics.eval.baselines import CenterBiasBaseline, CenterBiasFitInput

    fix = FixationSet(points=np.array([[10.0, 10.0]] * 20), weights=np.ones(20))
    cb_train = CenterBiasBaseline.fit([CenterBiasFitInput(shape=(60, 80), fixations=fix)], source_split="train")
    audit = drv.run_leakage_audit(
        splits=drv.SplitsFile(entries=(), protocol_version="p", method="m", generated_at_utc="t", file_sha256="s"),
        recorded_splits_sha256="s",
        hashes={},
        max_hamming=8,
        cb=cb_train,
        cb_built_at_utc="2026-09-11T10:00:00Z",
        test_metrics_started_utc="2026-09-11T11:00:00Z",
    )
    assert audit.cb_only_train is True
    assert audit.cb_built_before_test_metrics is True
    audit_late = drv.run_leakage_audit(
        splits=drv.SplitsFile(entries=(), protocol_version="p", method="m", generated_at_utc="t", file_sha256="s"),
        recorded_splits_sha256="bad",
        hashes={},
        max_hamming=8,
        cb=cb_train,
        cb_built_at_utc="2026-09-11T12:00:00Z",
        test_metrics_started_utc="2026-09-11T11:00:00Z",
    )
    assert audit_late.splits_sha256_ok is False
    assert audit_late.cb_built_before_test_metrics is False


# ---------------------------------------------------------------------------
# Part 5-7：全流程编排（合成数据集 + 假预测器；真实 UEyes 到达后同链路实跑）
# ---------------------------------------------------------------------------


def _fake_predictor_factory(profile_name: str):
    """假预测器：中心高斯概率图（原图分辨率 float64 sum=1）+ 未登记 profile 结构化跳过。"""
    from ui_attention.errors import ErrorCode, UiAttentionError

    if profile_name == "missing-backend-v1":
        raise UiAttentionError(ErrorCode.MODEL_NOT_READY, f"profile 未登记：{profile_name}")

    def predict(image_array: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = image_array.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        g = np.exp(-((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * (min(h, w) / 6.0) ** 2))
        density = g / g.sum()
        meta = {
            "profile": profile_name,
            "backend_id": "fake-eval",
            "backend_version": "0.0.1",
            "weights_sha256": [],
            "runtime": {"device": "cpu", "precision": "fp32", "elapsed_ms": 1.0, "peak_mem_mb": 1.0},
            "semantics": "probability_density",
        }
        return density, meta

    return predict


def test_run_full_evaluation_synthetic(tmp_path):
    from ui_attention.metrics.eval.config import EvalConfig

    ds_root = tmp_path / "extracted"
    make_ueyes_dataset(ds_root, n_participants=4, per_image_fixations=30)
    out_root = tmp_path / "eval-output"
    config = EvalConfig(dataset="synthetic-ueyes", bootstrap_b=200, windows=("1s", "3s", "7s"))

    outputs = drv.run_full_evaluation(
        dataset_root=ds_root,
        out_root=out_root,
        config=config,
        profiles=("fake-eval-v1", "missing-backend-v1"),
        predictor_factory=_fake_predictor_factory,
        dataset_name="synthetic-ueyes",
    )

    # 划分冻结 + 哈希
    assert len(outputs.splits_sha256) == 64
    assert (out_root / "splits.v1.json").is_file()

    # CB：每窗口 npz + sha256（train-only）
    for w in ("1s", "3s", "7s"):
        assert Path(outputs.cb_paths[w]).is_file()
        assert len(outputs.cb_hashes[w]) == 64

    # 未登记 profile 结构化跳过，不崩溃
    assert outputs.skipped_profiles == ["missing-backend-v1"]

    # 逐图 CSV：3 模型（uniform/center_bias/fake-eval-v1）× 3 test 图 × 7 指标 = 63 行/窗口
    for w in ("1s", "3s", "7s"):
        csv_path = Path(outputs.csv_paths[w].split(" ")[0])
        assert csv_path.is_file()
        import csv as csvmod

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csvmod.reader(f))
        assert len(rows) == 63 + 1  # + header
        assert all(r[2] == w for r in rows[1:])  # window 列隔离，绝不混合

    # 表 A-E 落盘 + 基线行齐备（缺行=报告不完整）
    for name in ("A", "B", "C", "D", "E"):
        assert (out_root / "tables" / f"table_{name}.md").is_file()
    assert outputs.tables["A"]["missing_baseline_rows"] == []
    models_in_a = {r["model"] for r in outputs.tables["A"]["rows"] if isinstance(r.get("model"), str)} or None
    assert (out_root / "tables" / "tables.json").is_file()
    del models_in_a

    # 审计：划分哈希一致、CB 仅 train、近重复跨划分簇如实记录（img_0001↔img_0005）
    assert outputs.audit["splits_sha256_ok"] is True
    assert outputs.audit["cb_only_train"] is True
    assert outputs.audit["cross_split_violations"] >= 1

    # S0-S4 判定结构（合成数据不断言 S1/S2 通过——均匀随机注视下 CB≈uniform 属合法结果）
    v = outputs.verdicts
    assert v["S0"]["status"] == "pass"
    assert v["S1"]["status"] in ("pass", "fail")
    assert "fake-eval-v1" in v["S2"]["models"]
    assert v["S2"]["models"]["fake-eval-v1"]["status"] in ("pass", "fail")
    assert v["S2"]["hard_gate"] is True
    assert v["S3"]["status"] == "not_applicable"  # FiWI 不纳入（L1 批准口径）
    assert v["S4"]["status"] == "disabled"

    # summary + 免责声明（对外红线：模型预测/公开数据评估，非游戏 UI 验证）
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert "模型预测" in summary["disclaimer"]
    assert summary["config_hash"] == config.config_hash()


def test_full_evaluation_predictor_shape_mismatch_fails_closed(tmp_path):
    """预测图与评估网格不一致 / 含 NaN → EvalGateError（不产出部分成功报告）。"""
    from ui_attention.metrics.eval.config import EvalConfig

    ds_root = tmp_path / "extracted"
    make_ueyes_dataset(ds_root, n_participants=4, per_image_fixations=30, with_near_duplicate=False)

    def bad_factory(profile_name: str):
        def predict(image_array: np.ndarray):
            h, w = image_array.shape[:2]
            bad = np.ones((h + 1, w), dtype=np.float64)  # 形状错误
            return bad, {"backend_version": "0"}

        return predict

    with pytest.raises(drv.EvalGateError):
        drv.run_full_evaluation(
            dataset_root=ds_root,
            out_root=tmp_path / "out-bad",
            config=EvalConfig(dataset="synthetic-ueyes", bootstrap_b=50, windows=("3s",)),
            profiles=("bad-v1",),
            predictor_factory=bad_factory,
            dataset_name="synthetic-ueyes",
            windows=("3s",),
        )


# ---------------------------------------------------------------------------
# 备选划分 v2（L2 对照补跑指令①：dHash 簇约束 + §3.3 哈希分桶）与对照报告⑤
# ---------------------------------------------------------------------------


def test_fallback_cluster_constrained_split(dataset):
    """簇约束备选划分：确定性、整簇同侧、生成参数完整记录。"""
    hashes = {}
    for img in dataset.images:
        hashes[img["image_id"]] = drv.dhash64(_load_gray(Path(img["path"])))
    ids = sorted(hashes)
    a1, p1 = drv.fallback_cluster_constrained_split(ids, hashes, protocol_version="r2-v0.1", max_hamming=8)
    a2, p2 = drv.fallback_cluster_constrained_split(ids, hashes, protocol_version="r2-v0.1", max_hamming=8)
    assert a1 == a2  # 完全确定
    assert p1["split_counts"]["train"] + p1["split_counts"]["val"] + p1["split_counts"]["test"] == len(ids)
    assert p1["clusters_total"] >= 1  # 预置近重复对
    # 整簇同侧：img_0001 与 img_0005（近重复）必须同 split
    assert a1["img_0001.png"] == a1["img_0005.png"]
    # 与纯 §3.3 分桶相比，簇约束只移动簇内成员
    base = drv.fallback_hash_split(ids, "r2-v0.1")
    moved = [i for i in ids if a1[i] != base[i]]
    assert set(moved) <= {"img_0001.png", "img_0005.png"}
    assert p1["images_moved_by_cluster_rule"] == len(moved)
    # 桶规则抽查：非簇内图与纯分桶一致
    assert a1["img_0003.png"] == base["img_0003.png"]


def test_splits_file_build_fallback_cluster(dataset):
    layout = drv.discover_layout(dataset.root)
    metas = drv.discover_image_paths(layout, drv.load_info_csv(layout.info_csv))
    hashes = {m.image_id: drv.dhash64(_load_gray(layout.root / m.rel_path)) for m in metas}
    splits = drv.SplitsFile.build_fallback_cluster(metas, hashes, protocol_version="r2-v0.1", max_hamming=8)
    assert splits.method == "fallback_hash_cluster_constrained"
    assert splits.split_of("img_0001.png") == splits.split_of("img_0005.png")  # 簇同侧
    joined = " ".join(splits.notes)
    assert "SHA256(protocol_version|image_id) mod 100" in joined  # 生成参数记录
    assert "整簇" in joined and "Hamming<=8" in joined


def test_run_full_evaluation_fallback_mode(tmp_path):
    """对照运行全流程：v2 划分文件名/CB 版本标签/variant 标注/描述性判定。

    盐 "synthetic-fb-4" 使 6 图合成集中 img_0004/img_0005 落入 test 桶（85-99），
    保证备选划分下 test 集非空（真实 UEyes 1980 图按 15% 桶比例自然非空）。
    """
    from ui_attention.metrics.eval.config import EvalConfig

    ds_root = tmp_path / "extracted"
    make_ueyes_dataset(ds_root, n_participants=4, per_image_fixations=30)
    out_root = tmp_path / "eval-fallback"
    config = EvalConfig(dataset="synthetic-ueyes", bootstrap_b=200, protocol_version="synthetic-fb-4")
    outputs = drv.run_full_evaluation(
        dataset_root=ds_root,
        out_root=out_root,
        config=config,
        profiles=("fake-eval-v1",),
        predictor_factory=_fake_predictor_factory,
        dataset_name="synthetic-ueyes",
        splits_mode="fallback_cluster",
        splits_filename="splits.v2-fallback.json",
        cb_version_tag="v2-fallback",
        variant_label="备选划分对照",
        verdicts_descriptive=True,
    )
    assert (out_root / "splits.v2-fallback.json").is_file()
    sp = json.loads((out_root / "splits.v2-fallback.json").read_text(encoding="utf-8"))
    assert sp["method"] == "fallback_hash_cluster_constrained"
    by_id = {e["image_id"]: e["split"] for e in sp["images"]}
    # 簇约束生效验证：img_0005（img_0001 近重复）整簇随代表 img_0001 同侧（train），
    # 纯分桶下 img_0005 本应 test —— §3.4.1 近重复不跨划分在备选划分上自动满足
    assert by_id["img_0001.png"] == "train"
    assert by_id["img_0005.png"] == "train"
    assert by_id["img_0004.png"] == "test"
    assert [i["image_id"] for i in sp["images"] if i["split"] == "test"] == ["img_0004.png"]
    # CB npz 用 v2 版本标签（与官方 v1 文件不混）
    for w in ("1s", "3s", "7s"):
        assert Path(outputs.cb_paths[w]).name == f"cb_synthetic-ueyes_{w}_train.v2-fallback.npz"
    # 表 A 含候选模型行（test 集非空的证明）
    models_in_a = {r["model"] for r in outputs.tables["A"]["rows"]}
    assert "fake-eval-v1" in models_in_a
    # 表标注 + summary variant
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["variant"] == "备选划分对照"
    assert summary["splits_mode"] == "fallback_cluster"
    assert any("备选划分对照" in x for x in summary["limitations"])
    # S2 判定仅描述性
    assert outputs.verdicts["S2"].get("descriptive_only") is True
    assert "描述性" in outputs.verdicts["descriptive_note"]
    # 表 md 带对照横幅
    md = (out_root / "tables" / "table_A.md").read_text(encoding="utf-8")
    assert md.startswith("> **备选划分对照**")


def test_build_split_comparison(tmp_path):
    """对照报告⑤：官方 vs 备选（描述性组间差值，两运行分表不混）。"""
    from ui_attention.metrics.eval.config import EvalConfig

    ds_root = tmp_path / "extracted"
    make_ueyes_dataset(ds_root, n_participants=4, per_image_fixations=30, with_near_duplicate=False)
    off = drv.run_full_evaluation(
        dataset_root=ds_root,
        out_root=tmp_path / "off",
        config=EvalConfig(dataset="synthetic-ueyes", bootstrap_b=100, windows=("3s",)),
        profiles=("fake-eval-v1",),
        predictor_factory=_fake_predictor_factory,
        dataset_name="synthetic-ueyes",
        windows=("3s",),
    )
    fbk = drv.run_full_evaluation(
        dataset_root=ds_root,
        out_root=tmp_path / "fbk",
        config=EvalConfig(
            dataset="synthetic-ueyes", bootstrap_b=100, windows=("3s",), protocol_version="synthetic-fb-4"
        ),
        profiles=("fake-eval-v1",),
        predictor_factory=_fake_predictor_factory,
        dataset_name="synthetic-ueyes",
        windows=("3s",),
        splits_mode="fallback_cluster",
        splits_filename="splits.v2-fallback.json",
        cb_version_tag="v2-fallback",
        variant_label="备选划分对照",
        verdicts_descriptive=True,
    )
    result = drv.build_split_comparison(
        off.out_root / "tables" / "tables.json",
        fbk.out_root / "tables" / "tables.json",
        tmp_path / "fbk",
    )
    assert result["rows"], "对照行不应为空"
    models = {r["model"] for r in result["rows"]}
    assert {"uniform", "center_bias", "fake-eval-v1"} <= models
    assert all(r["metric"] in ("IG_CB", "NSS") for r in result["rows"])
    assert "描述性" in result["note"] and "§8.1" in result["note"]
    assert (tmp_path / "fbk" / "comparison_official_vs_fallback.md").is_file()
    assert (tmp_path / "fbk" / "comparison_official_vs_fallback.json").is_file()
    del off, fbk


# ---------------------------------------------------------------------------
# S2 口径裁决落地（用户批准 2026-09-12）：window_matched / all_window 双口径 + 附录生成
# ---------------------------------------------------------------------------


def test_nominal_window_parsing():
    windows = ("1s", "3s", "7s")
    assert drv.nominal_window("foveacast-onnx-3s-v1", windows) == "3s"
    assert drv.nominal_window("foveacast-onnx-1s-v1", windows) == "1s"
    assert drv.nominal_window("foveacast-onnx-7s-v1", windows) == "7s"
    assert drv.nominal_window("deepgaze-iie-v1", windows) is None  # 通用域无标称窗口
    assert drv.nominal_window("fake-eval-v1", windows) is None


def _b_entry(model, window, metric, ci_low, mean_diff=0.1):
    return {
        "model": model,
        "baseline": "center_bias",
        "window": window,
        "metric": metric,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_low + 0.2,
        "win_rate": 0.8,
        "n_paired": 100,
    }


def test_judge_criteria_s2_dual_mode():
    """window_matched：3s 模型只看 3s 窗（1s 交叉窗 fail 不影响门槛，标 sensitivity_only）；
    all_window：同一数据 fail。无标称窗口模型两口径一致（全窗口）。"""
    from ui_attention.metrics.eval.config import EvalConfig, SuccessCriteria

    table_b = [
        # foveacast-3s：标称窗口 3s 双指标过；交叉窗口 1s IG_CB 边际 fail
        _b_entry("foveacast-onnx-3s-v1", "3s", "IG_CB", 0.47),
        _b_entry("foveacast-onnx-3s-v1", "3s", "NSS", 0.54),
        _b_entry("foveacast-onnx-3s-v1", "1s", "IG_CB", -0.05),
        _b_entry("foveacast-onnx-3s-v1", "1s", "NSS", 0.19),
        _b_entry("foveacast-onnx-3s-v1", "7s", "IG_CB", 0.28),
        _b_entry("foveacast-onnx-3s-v1", "7s", "NSS", 0.43),
        # deepgaze：无标称窗口，多窗口 fail
        _b_entry("deepgaze-iie-v1", "1s", "IG_CB", -1.1),
        _b_entry("deepgaze-iie-v1", "1s", "NSS", -0.58),
        _b_entry("deepgaze-iie-v1", "3s", "IG_CB", -0.64),
        _b_entry("deepgaze-iie-v1", "3s", "NSS", -0.32),
        _b_entry("deepgaze-iie-v1", "7s", "IG_CB", -0.31),
        _b_entry("deepgaze-iie-v1", "7s", "NSS", 0.003),
    ]
    cands = ["foveacast-onnx-3s-v1", "deepgaze-iie-v1"]
    base = EvalConfig()

    wm = drv.judge_criteria(config=base, table_b=table_b, candidates=cands, s0_report={"skipped": True})["S2"]
    assert wm["s2_mode"] == "window_matched"
    assert wm["models"]["foveacast-onnx-3s-v1"]["status"] == "pass"
    assert wm["models"]["foveacast-onnx-3s-v1"]["gate_windows"] == ["3s"]
    d = wm["models"]["foveacast-onnx-3s-v1"]["detail"]
    assert d["1s/IG_CB"]["counts_toward_gate"] is False and d["1s/IG_CB"]["sensitivity_only"] is True
    assert d["3s/IG_CB"]["counts_toward_gate"] is True and "sensitivity_only" not in d["3s/IG_CB"]
    assert wm["models"]["deepgaze-iie-v1"]["status"] == "fail"  # 无标称窗口 → 全窗口判定
    assert wm["models"]["deepgaze-iie-v1"]["gate_windows"] == ["1s", "3s", "7s"]

    aw_cfg = EvalConfig(criteria=SuccessCriteria(s2_mode="all_window"))
    aw = drv.judge_criteria(config=aw_cfg, table_b=table_b, candidates=cands, s0_report={"skipped": True})["S2"]
    assert aw["s2_mode"] == "all_window"
    assert aw["models"]["foveacast-onnx-3s-v1"]["status"] == "fail"  # 1s/IG_CB 拖垮旧口径
    assert aw["models"]["foveacast-onnx-3s-v1"]["gate_windows"] == ["1s", "3s", "7s"]
    assert aw["models"]["deepgaze-iie-v1"]["status"] == "fail"


def test_load_per_image_csv_roundtrip(tmp_path):
    from ui_attention.metrics.eval.config import EvalConfig

    ds_root = tmp_path / "extracted"
    make_ueyes_dataset(ds_root, n_participants=4, per_image_fixations=30, with_near_duplicate=False)
    outputs = drv.run_full_evaluation(
        dataset_root=ds_root,
        out_root=tmp_path / "out",
        config=EvalConfig(dataset="synthetic-ueyes", bootstrap_b=50, windows=("3s",)),
        profiles=("fake-eval-v1",),
        predictor_factory=_fake_predictor_factory,
        dataset_name="synthetic-ueyes",
        windows=("3s",),
    )
    csv_path = Path(outputs.csv_paths["3s"].split(" ")[0])
    rows = drv.load_per_image_csv(csv_path)
    assert len(rows) == 63  # 3 模型 × 3 test 图 × 7 指标
    assert all(r.window == "3s" for r in rows)
    assert {r.model for r in rows} == {"uniform", "center_bias", "fake-eval-v1"}


def test_build_s2_addendum_end_to_end(tmp_path):
    """附录生成：不重推理、只读 CSV、双口径并列、输入哈希与批准注记齐备。"""
    from ui_attention.metrics.eval.config import EvalConfig

    ds_root = tmp_path / "extracted"
    make_ueyes_dataset(ds_root, n_participants=4, per_image_fixations=30, with_near_duplicate=False)
    cfg = EvalConfig(dataset="synthetic-ueyes", bootstrap_b=100, windows=("1s", "3s", "7s"))
    outputs = drv.run_full_evaluation(
        dataset_root=ds_root,
        out_root=tmp_path / "out",
        config=cfg,
        profiles=("fake-eval-v1",),
        predictor_factory=_fake_predictor_factory,
        dataset_name="synthetic-ueyes",
    )
    csvs = [Path(outputs.csv_paths[w].split(" ")[0]) for w in ("1s", "3s", "7s")]
    out_file = tmp_path / "s2_window_matched_addendum.json"
    add = drv.build_s2_addendum(official_csvs=csvs, out_path=out_file, fallback_csvs=csvs[:1], config=cfg)
    assert out_file.is_file()
    assert add["no_reinference"] is True
    assert "用户批准 2026-09-12" in add["approval_note"]
    assert add["bootstrap"] == {"B": 100, "seed": 20260911, "stratified": True, "ci": 0.95}
    # 输入清单 + 哈希
    assert len(add["inputs_official"]) == 3
    assert all(len(i["sha256"]) == 64 and i["size_bytes"] > 0 for i in add["inputs_official"])
    assert len(add["inputs_fallback"]) == 1
    # 双口径并列（fake-eval-v1 无标称窗口 → 两口径判定集合一致）
    s2 = add["official"]["s2"]
    assert set(s2) == {"window_matched", "all_window"}
    assert (
        s2["window_matched"]["models"]["fake-eval-v1"]["status"] == s2["all_window"]["models"]["fake-eval-v1"]["status"]
    )
    assert s2["window_matched"]["models"]["fake-eval-v1"]["gate_windows"] == ["1s", "3s", "7s"]
    assert add["fallback"]["descriptive_only"] is True
    # 落盘内容可解析且与返回一致
    on_disk = json.loads(out_file.read_text(encoding="utf-8"))
    assert on_disk["schema"] == "game-ui-attention-s2-addendum/v1"
    assert on_disk["official"]["n_rows"] == add["official"]["n_rows"] == 3 * 3 * 7 * 3
