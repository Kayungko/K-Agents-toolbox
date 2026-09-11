/* ui-attention compare.js — A/B 对比页交互（C3 原创，原生 JS，无外部依赖）。
 *
 * 只做视图切换与区域层显隐；数值全部来自服务端静态渲染（comparison.json
 * 内容），本脚本不计算、不改写任何统计值；无 fetch/XHR/动态 script，
 * 无 http(s):// 引用。
 */
(function () {
  "use strict";

  function bindStage(stageId, radioName, toggleId) {
    var stage = document.getElementById(stageId);
    if (!stage) { return; }
    var radios = document.querySelectorAll('input[name="' + radioName + '"]');
    function apply() {
      var mode = "overlay";
      for (var i = 0; i < radios.length; i++) {
        var seg = document.querySelector('label[for="' + radios[i].id + '"]');
        if (radios[i].checked) { mode = radios[i].value; }
        if (seg) { seg.classList.toggle("active", radios[i].checked); }
      }
      stage.setAttribute("data-mode", mode);
    }
    for (var i = 0; i < radios.length; i++) {
      radios[i].addEventListener("change", apply);
    }
    var toggle = document.getElementById(toggleId);
    if (toggle) {
      toggle.addEventListener("change", function () {
        stage.classList.toggle("regions-off", !toggle.checked);
      });
    }
    apply();
  }

  bindStage("stage-a", "view-mode-a", "toggle-regions-a");
  bindStage("stage-b", "view-mode-b", "toggle-regions-b");
})();
