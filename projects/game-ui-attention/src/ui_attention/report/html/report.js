/* ui-attention report.js — 单图报告交互（C3 原创，原生 JS，无外部依赖）。
 *
 * 安全约定：
 * - 所有用户可控文本（区域 id/label/role、玩家目标等）注入 DOM 前一律
 *   经 esc() HTML 转义；数据仅来自页内 JSON 数据岛（#report-data），
 *   不发起任何网络请求（无 fetch/XHR/动态 script）。
 * - 编辑只发生在浏览器内存；导出经 Blob + a[download] 触发浏览器下载，
 *   绝不隐式写回本地文件（technical-design §6）。
 * - SVG 片段通过宿主 div 的 innerHTML 由 HTML 解析器构建（svg 标签自动
 *   进入 SVG 命名空间），不使用命名空间 URL 字面量，产物中无 http(s):// 引用。
 */
(function () {
  "use strict";

  var dataEl = document.getElementById("report-data");
  if (!dataEl) { return; }
  var DATA = JSON.parse(dataEl.textContent);
  var IMG = DATA.image; /* {sha256, width, height, regions_schema_version} */

  var state = {
    regions: JSON.parse(JSON.stringify(DATA.regions || [])),
    mode: "off",          /* off | rect | poly */
    selectedId: null,
    editingId: null,      /* 非 null 表示正在编辑既有区域 */
    hasDraftGeometry: false,
    dragStart: null,
    dragCurrent: null,
    polyPoints: [],
    baselineCount: (DATA.regions || []).length,
    changeCount: 0
  };

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function isInt(v) { return typeof v === "number" && isFinite(v) && Math.floor(v) === v; }

  var stage = $("main-stage");
  var baseImg = $("base-layer");
  var svgHost = $("region-svg-host");
  var editNote = $("edit-note");
  var dirtyNote = $("dirty-note");
  var exportNote = $("export-note");

  function note(el, msg, cls) {
    if (!el) { return; }
    el.textContent = msg || "";
    el.className = cls || "";
  }

  /* ---------------- 视图切换 ---------------- */
  var radios = document.querySelectorAll('input[name="view-mode"]');
  function applyViewMode() {
    var mode = "original";
    for (var i = 0; i < radios.length; i++) {
      var seg = document.querySelector('label[for="' + radios[i].id + '"]');
      if (radios[i].checked) { mode = radios[i].value; }
      if (seg) { seg.classList.toggle("active", radios[i].checked); }
    }
    stage.setAttribute("data-mode", mode);
  }
  for (var i = 0; i < radios.length; i++) {
    radios[i].addEventListener("change", applyViewMode);
  }
  var toggleRegions = $("toggle-regions");
  if (toggleRegions) {
    toggleRegions.addEventListener("change", function () {
      stage.classList.toggle("regions-off", !toggleRegions.checked);
    });
  }
  applyViewMode();

  /* ---------------- 几何/校验（镜像 export_regions.py 规则） ---------------- */
  function geometrySummary(g) {
    if (!g) { return "(无)"; }
    if (g.type === "rect") {
      return "rect x=" + g.x + " y=" + g.y + " w=" + g.width + " h=" + g.height;
    }
    if (g.type === "polygon") {
      return "polygon " + g.points.length + " 点";
    }
    return "(未知)";
  }

  function orient(a, b, c) {
    var v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    return v > 0 ? 1 : (v < 0 ? -1 : 0);
  }
  function onSeg(a, b, p) {
    return Math.min(a[0], b[0]) <= p[0] && p[0] <= Math.max(a[0], b[0]) &&
           Math.min(a[1], b[1]) <= p[1] && p[1] <= Math.max(a[1], b[1]);
  }
  function segIntersect(p1, p2, p3, p4) {
    var o1 = orient(p1, p2, p3), o2 = orient(p1, p2, p4);
    var o3 = orient(p3, p4, p1), o4 = orient(p3, p4, p2);
    if (o1 !== o2 && o3 !== o4) { return true; }
    if (o1 === 0 && onSeg(p1, p2, p3)) { return true; }
    if (o2 === 0 && onSeg(p1, p2, p4)) { return true; }
    if (o3 === 0 && onSeg(p3, p4, p1)) { return true; }
    if (o4 === 0 && onSeg(p3, p4, p2)) { return true; }
    return false;
  }
  function polyArea2(pts) {
    var s = 0, n = pts.length;
    for (var k = 0; k < n; k++) {
      var a = pts[k], b = pts[(k + 1) % n];
      s += a[0] * b[1] - b[0] * a[1];
    }
    return s;
  }

  function validateRegion(r, allRegions) {
    var errs = [];
    if (typeof r.id !== "string" || !r.id.trim()) { errs.push("id 必须为非空字符串"); }
    for (var k = 0; k < allRegions.length; k++) {
      if (allRegions[k] !== r && allRegions[k].id === r.id) {
        errs.push("id 重复: " + r.id);
      }
    }
    if (["manual", "agent", "imported"].indexOf(r.source) < 0) {
      errs.push("source 非法: " + r.source);
    }
    if (r.source === "imported" && !(r.source_reference && Object.keys(r.source_reference).length)) {
      errs.push("source=imported 必须携带非空 source_reference（请经 CLI 导入管线附加）");
    }
    if (["candidate", "confirmed"].indexOf(r.status) < 0) {
      errs.push("status 非法: " + r.status);
    }
    var g = r.geometry;
    if (!g) { errs.push("缺少 geometry"); return errs; }
    var W = IMG.width, H = IMG.height;
    if (g.type === "rect") {
      if (!isInt(g.x) || !isInt(g.y) || !isInt(g.width) || !isInt(g.height)) {
        errs.push("rect 坐标必须为整数");
      } else {
        if (g.width <= 0 || g.height <= 0) { errs.push("rect 零面积"); }
        if (g.x < 0 || g.y < 0 || g.x + g.width > W || g.y + g.height > H) {
          errs.push("rect 越界（画布 " + W + "×" + H + "）");
        }
      }
    } else if (g.type === "polygon") {
      var pts = g.points;
      if (!Array.isArray(pts) || pts.length < 3) { errs.push("polygon 至少 3 个顶点"); return errs; }
      var ok = true, m;
      for (m = 0; m < pts.length; m++) {
        if (!Array.isArray(pts[m]) || pts[m].length !== 2 || !isInt(pts[m][0]) || !isInt(pts[m][1])) {
          errs.push("polygon 顶点必须为 [int,int]"); ok = false; break;
        }
        if (pts[m][0] < 0 || pts[m][1] < 0 || pts[m][0] >= W || pts[m][1] >= H) {
          errs.push("polygon 顶点越界: (" + pts[m][0] + "," + pts[m][1] + ")，须在 [0," + W + ")×[0," + H + ")"); ok = false; break;
        }
      }
      if (ok) {
        var seen = {}, dup = false;
        for (m = 0; m < pts.length; m++) {
          var key = pts[m][0] + "," + pts[m][1];
          if (seen[key]) { dup = true; break; }
          seen[key] = true;
        }
        if (dup) { errs.push("polygon 存在重复顶点（含首尾闭合写法）"); }
        else if (polyArea2(pts) === 0) { errs.push("polygon 零面积（共线或退化）"); }
        else {
          var n = pts.length;
          if (n >= 4) {
            outer:
            for (var a = 0; a < n; a++) {
              for (var b = a + 1; b < n; b++) {
                var adj = (b === a + 1) || (a === 0 && b === n - 1);
                if (adj) { continue; }
                if (segIntersect(pts[a], pts[(a + 1) % n], pts[b], pts[(b + 1) % n])) {
                  errs.push("polygon 自交");
                  break outer;
                }
              }
            }
          }
        }
      }
    } else {
      errs.push("未知 geometry.type: " + g.type);
    }
    return errs;
  }

  /* ---------------- SVG / 表格渲染 ---------------- */
  function svgMarkup() {
    var parts = [];
    parts.push('<svg viewBox="0 0 ' + IMG.width + " " + IMG.height + '" preserveAspectRatio="none">');
    parts.push("<g>");
    for (var i = 0; i < state.regions.length; i++) {
      var r = state.regions[i];
      var g = r.geometry;
      if (!g) { continue; }
      var cls = "region-box " + (r.status === "candidate" ? "candidate" : "confirmed");
      if (state.selectedId === r.id) { cls += " selected"; }
      if (g.type === "rect") {
        parts.push('<rect class="' + cls + '" x="' + g.x + '" y="' + g.y +
          '" width="' + g.width + '" height="' + g.height + '"></rect>');
        parts.push('<text class="region-label" x="' + (g.x + 4) + '" y="' + (g.y + 16) + '">' +
          esc(r.label || r.id) + (r.status === "candidate" ? "（候选）" : "") + "</text>");
      } else if (g.type === "polygon") {
        var ptStr = g.points.map(function (p) { return p[0] + "," + p[1]; }).join(" ");
        parts.push('<polygon class="' + cls + '" points="' + ptStr + '"></polygon>');
        parts.push('<text class="region-label" x="' + (g.points[0][0] + 4) + '" y="' + (g.points[0][1] + 16) + '">' +
          esc(r.label || r.id) + (r.status === "candidate" ? "（候选）" : "") + "</text>");
      }
    }
    /* 草稿层 */
    if (state.mode === "rect" && state.dragStart && state.dragCurrent) {
      var x0 = Math.min(state.dragStart.x, state.dragCurrent.x);
      var y0 = Math.min(state.dragStart.y, state.dragCurrent.y);
      var w0 = Math.abs(state.dragCurrent.x - state.dragStart.x);
      var h0 = Math.abs(state.dragCurrent.y - state.dragStart.y);
      parts.push('<rect class="draft-box" x="' + x0 + '" y="' + y0 + '" width="' + w0 + '" height="' + h0 + '"></rect>');
    }
    if (state.mode === "poly" && state.polyPoints.length) {
      var pp = state.polyPoints.map(function (p) { return p[0] + "," + p[1]; }).join(" ");
      parts.push('<polyline class="draft-box" points="' + pp + '"></polyline>');
      for (var v = 0; v < state.polyPoints.length; v++) {
        parts.push('<circle class="vertex-dot" cx="' + state.polyPoints[v][0] + '" cy="' +
          state.polyPoints[v][1] + '" r="3"></circle>');
      }
    }
    parts.push("</g></svg>");
    return parts.join("");
  }

  function renderSvg() {
    if (svgHost) { svgHost.innerHTML = svgMarkup(); }
  }

  function renderTable() {
    var tbody = $("regions-tbody");
    if (!tbody) { return; }
    var rows = [];
    for (var i = 0; i < state.regions.length; i++) {
      var r = state.regions[i];
      var trCls = (r.status === "candidate" ? "row-candidate" : "") +
        (state.selectedId === r.id ? " row-selected" : "");
      rows.push('<tr class="' + trCls.trim() + '" data-rid="' + esc(r.id) + '">' +
        "<td>" + esc(r.id) + "</td>" +
        "<td>" + esc(r.label === undefined || r.label === null ? "" : r.label) + "</td>" +
        "<td>" + esc(r.role === undefined || r.role === null ? "" : r.role) + "</td>" +
        "<td>" + esc(r.source) + "</td>" +
        '<td><span class="badge ' + (r.status === "candidate" ? "candidate" : "confirmed") + '">' +
          esc(r.status) + "</span></td>" +
        '<td class="mono">' + esc(geometrySummary(r.geometry)) + "</td>" +
        '<td><button class="btn" data-act="edit" data-rid="' + esc(r.id) + '">编辑</button> ' +
        '<button class="btn danger" data-act="del" data-rid="' + esc(r.id) + '">删除</button></td>' +
        "</tr>");
    }
    tbody.innerHTML = rows.join("");
  }

  function renderDirty() {
    if (!dirtyNote) { return; }
    if (state.changeCount > 0) {
      dirtyNote.textContent = "未导出变更 " + state.changeCount +
        " 处：编辑仅存在于浏览器内存，不会写回本地文件；导出 JSON 后由 CLI summarize 重算统计（不重新推理）。";
      dirtyNote.className = "dirty-note";
    } else {
      dirtyNote.textContent = "";
    }
  }

  function bumpChange() { state.changeCount += 1; renderDirty(); }
  function renderAll() { renderSvg(); renderTable(); }

  var tbodyEl = $("regions-tbody");
  if (tbodyEl) {
    tbodyEl.addEventListener("click", function (ev) {
      var btn = ev.target.closest ? ev.target.closest("button[data-act]") : null;
      if (!btn) { return; }
      var rid = btn.getAttribute("data-rid");
      if (btn.getAttribute("data-act") === "del") {
        state.regions = state.regions.filter(function (r) { return r.id !== rid; });
        if (state.selectedId === rid) { state.selectedId = null; }
        if (state.editingId === rid) { cancelEdit(); }
        bumpChange();
        renderAll();
      } else if (btn.getAttribute("data-act") === "edit") {
        startEdit(rid);
      }
    });
  }

  /* ---------------- 编辑表单 ---------------- */
  var fId = $("f-id"), fLabel = $("f-label"), fRole = $("f-role");
  var fSource = $("f-source"), fStatus = $("f-status");
  var btnAdd = $("btn-add"), btnCancel = $("btn-cancel-edit");

  function cancelEdit() {
    state.editingId = null;
    if (fId) { fId.readOnly = false; }
    if (fSource) { fSource.disabled = false; }
    if (btnAdd) { btnAdd.textContent = "添加区域"; }
    if (btnCancel) { btnCancel.disabled = true; }
    note(editNote, "", "");
  }

  function startEdit(rid) {
    var r = null;
    for (var i = 0; i < state.regions.length; i++) {
      if (state.regions[i].id === rid) { r = state.regions[i]; }
    }
    if (!r) { return; }
    state.editingId = rid;
    state.selectedId = rid;
    fId.value = r.id; fId.readOnly = true;
    fLabel.value = r.label || "";
    fRole.value = r.role || "";
    fSource.value = r.source; fSource.disabled = true;
    fStatus.value = r.status;
    if (btnAdd) { btnAdd.textContent = "保存修改"; }
    if (btnCancel) { btnCancel.disabled = false; }
    note(editNote, "正在编辑 " + rid + "：可改 label/role/status；几何不可改（删除后重新圈选）。", "");
    renderAll();
  }

  if (btnCancel) { btnCancel.addEventListener("click", cancelEdit); }

  if (btnAdd) {
    btnAdd.addEventListener("click", function () {
      var errs = [];
      var id = fId.value.trim();
      if (!id) { errs.push("id 不能为空"); }
      var label = fLabel.value;
      var role = fRole.value.trim();
      var status = fStatus.value;
      if (state.editingId !== null) {
        for (var i = 0; i < state.regions.length; i++) {
          if (state.regions[i].id === state.editingId) {
            state.regions[i].label = label;
            if (role) { state.regions[i].role = role; } else { delete state.regions[i].role; }
            state.regions[i].status = status;
          }
        }
        bumpChange();
        cancelEdit();
        renderAll();
        return;
      }
      /* 新增：需要已完成的草稿几何 */
      var geometry = commitDraftGeometry();
      if (!geometry) {
        note(editNote, "请先在图上圈选几何（矩形拖选或多边形点选并完成）。", "error-note");
        return;
      }
      var candidateRegion = {
        id: id, label: label, geometry: geometry,
        source: fSource.value, status: status
      };
      if (role) { candidateRegion.role = role; }
      var dup = state.regions.some(function (r) { return r.id === id; });
      if (dup) { errs.push("id 重复: " + id); }
      errs = errs.concat(validateRegion(candidateRegion, state.regions));
      if (errs.length) {
        note(editNote, "校验失败：\n" + errs.join("\n"), "error-note");
        return;
      }
      state.regions.push(candidateRegion);
      state.selectedId = id;
      bumpChange();
      clearDraft();
      fId.value = ""; fLabel.value = ""; fRole.value = "";
      note(editNote, "已添加区域 " + id + "。", "ok-note");
      renderAll();
    });
  }

  /* ---------------- 圈选交互 ---------------- */
  function toImageCoords(ev) {
    var rect = baseImg.getBoundingClientRect();
    var sx = IMG.width / rect.width;
    var sy = IMG.height / rect.height;
    return {
      x: clamp(Math.round((ev.clientX - rect.left) * sx), 0, IMG.width),
      y: clamp(Math.round((ev.clientY - rect.top) * sy), 0, IMG.height)
    };
  }

  function setMode(m) {
    state.mode = m;
    clearDraft();
    stage.classList.toggle("selecting", m !== "off");
    var defs = { "mode-off": "off", "mode-rect": "rect", "mode-poly": "poly" };
    Object.keys(defs).forEach(function (btnId) {
      var b = $(btnId);
      if (b) { b.classList.toggle("active", defs[btnId] === m); }
    });
    var polyBtn = $("btn-finish-poly");
    if (polyBtn) { polyBtn.disabled = (m !== "poly"); }
    note(editNote, m === "rect" ? "矩形模式：在图上按下并拖动圈选。" :
      (m === "poly" ? "多边形模式：单击加点，双击或点『完成多边形』闭合，Esc 取消。" : ""), "");
  }

  function clearDraft() {
    state.dragStart = null; state.dragCurrent = null;
    state.polyPoints = []; state.hasDraftGeometry = false;
    state.draftGeometry = null;
    renderSvg();
  }

  function commitDraftGeometry() {
    if (state.draftGeometry) {
      var g = state.draftGeometry;
      state.draftGeometry = null;
      return g;
    }
    if (state.mode === "rect" && state.dragStart && state.dragCurrent) {
      var x = Math.min(state.dragStart.x, state.dragCurrent.x);
      var y = Math.min(state.dragStart.y, state.dragCurrent.y);
      var w = Math.abs(state.dragCurrent.x - state.dragStart.x);
      var h = Math.abs(state.dragCurrent.y - state.dragStart.y);
      if (w < 1 || h < 1) { return null; }
      return { type: "rect", x: x, y: y, width: w, height: h };
    }
    if (state.mode === "poly" && state.polyPoints.length >= 3) {
      return { type: "polygon", points: state.polyPoints.map(function (p) { return [p[0], p[1]]; }) };
    }
    return null;
  }

  stage.addEventListener("pointerdown", function (ev) {
    if (state.mode !== "rect") { return; }
    ev.preventDefault();
    state.draftGeometry = null;
    state.dragStart = toImageCoords(ev);
    state.dragCurrent = state.dragStart;
    renderSvg();
  });
  stage.addEventListener("pointermove", function (ev) {
    if (state.mode !== "rect" || !state.dragStart) { return; }
    state.dragCurrent = toImageCoords(ev);
    renderSvg();
  });
  stage.addEventListener("pointerup", function (ev) {
    if (state.mode !== "rect" || !state.dragStart) { return; }
    state.dragCurrent = toImageCoords(ev);
    var g = commitDraftGeometry();
    if (g) {
      state.draftGeometry = g;
      state.hasDraftGeometry = true;
      note(editNote, "已圈选 " + geometrySummary(g) + "，填写字段后点『添加区域』。", "ok-note");
    } else {
      note(editNote, "选区为零面积，请重新拖动。", "error-note");
    }
    state.dragStart = null; state.dragCurrent = null;
    renderSvg();
  });
  stage.addEventListener("click", function (ev) {
    if (state.mode !== "poly") { return; }
    state.polyPoints.push([toImageCoords(ev).x, toImageCoords(ev).y]);
    renderSvg();
  });
  stage.addEventListener("dblclick", function (ev) {
    if (state.mode !== "poly") { return; }
    ev.preventDefault();
    finishPolygon();
  });
  function finishPolygon() {
    var pts = state.polyPoints;
    while (pts.length >= 2 && pts[pts.length - 1][0] === pts[pts.length - 2][0] &&
           pts[pts.length - 1][1] === pts[pts.length - 2][1]) {
      pts.pop(); /* dblclick 产生的重复点 */
    }
    if (pts.length < 3) {
      note(editNote, "多边形至少需要 3 个不重复顶点。", "error-note");
      return;
    }
    var g = { type: "polygon", points: pts.map(function (p) { return [p[0], p[1]]; }) };
    var errs = validateRegion({ id: "__draft__", source: "manual", status: "confirmed", geometry: g }, []);
    if (errs.length) {
      note(editNote, "多边形校验失败：\n" + errs.join("\n"), "error-note");
      return;
    }
    state.draftGeometry = g;
    state.hasDraftGeometry = true;
    state.polyPoints = [];
    note(editNote, "已圈选 " + geometrySummary(g) + "，填写字段后点『添加区域』。", "ok-note");
    renderSvg();
  }
  var finishBtn = $("btn-finish-poly");
  if (finishBtn) { finishBtn.addEventListener("click", finishPolygon); }
  var clearBtn = $("btn-clear-draft");
  if (clearBtn) { clearBtn.addEventListener("click", function () { clearDraft(); note(editNote, "草稿已清除。", ""); }); }
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") { clearDraft(); }
  });

  var modeButtons = [["mode-off", "off"], ["mode-rect", "rect"], ["mode-poly", "poly"]];
  modeButtons.forEach(function (pair) {
    var b = $(pair[0]);
    if (b) { b.addEventListener("click", function () { setMode(pair[1]); }); }
  });

  /* ---------------- 导出（浏览器下载，不写回本地文件） ---------------- */
  function exportRegion(r) {
    /* 对齐 C1 RegionSpec.to_dict 形态：label/role 恒在（空 → null），
       source_reference 原样透传（imported 区域必须保留）。 */
    var out = { id: r.id };
    out.label = (r.label === undefined || r.label === null || r.label === "") ? null : r.label;
    out.role = (r.role === undefined || r.role === null || r.role === "") ? null : r.role;
    out.geometry = JSON.parse(JSON.stringify(r.geometry));
    out.source = r.source;
    out.status = r.status;
    if (r.source_reference !== undefined && r.source_reference !== null) {
      out.source_reference = JSON.parse(JSON.stringify(r.source_reference));
    }
    return out;
  }

  var btnExport = $("btn-export");
  if (btnExport) {
    btnExport.addEventListener("click", function () {
      var payload = {
        schema_version: IMG.regions_schema_version,
        image_sha256: IMG.sha256,
        regions: state.regions.map(exportRegion)
      };
      var allErrs = [];
      payload.regions.forEach(function (r) {
        validateRegion(r, payload.regions).forEach(function (e) {
          allErrs.push("[" + r.id + "] " + e);
        });
      });
      if (allErrs.length) {
        note(exportNote, "导出被拒绝（校验失败）：\n" + allErrs.join("\n"), "error-note");
        return;
      }
      var text = JSON.stringify(payload, null, 2) + "\n";
      var blob = new Blob([text], { type: "application/json" });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = "regions-" + IMG.sha256.slice(0, 8) + ".json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
      note(exportNote, "已触发浏览器下载 " + a.download +
        "（含图片 SHA-256 绑定）。本页面不会写回任何本地文件；把该文件交给 CLI summarize 重算统计。", "ok-note");
    });
  }

  /* ---------------- 初始渲染 ---------------- */
  renderAll();
  renderDirty();
  if (fSource) { fSource.disabled = false; }
})();
