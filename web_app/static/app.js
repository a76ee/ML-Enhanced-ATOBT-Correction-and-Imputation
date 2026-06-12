const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const submitButton = document.getElementById("submitButton");
const toast = document.getElementById("toast");
const downloads = document.getElementById("downloads");
const downloadXlsx = document.getElementById("downloadXlsx");
const downloadCsv = document.getElementById("downloadCsv");
const airlineTable = document.getElementById("airlineTable");
const previewTable = document.getElementById("previewTable");
const chart = document.getElementById("airlineChart");

const metricIds = {
  rows: document.getElementById("mRows"),
  history: document.getElementById("mHistory"),
  mae: document.getElementById("mMae"),
  m3: document.getElementById("m3"),
  m5: document.getElementById("m5"),
  baseline: document.getElementById("mBaseline"),
};

const HEADER_LABELS = {
  _source_row: "来源行",
  _source_file: "来源文件",
  IFC: "航司",
  n: "样本量",
  "A-TOBT": "原始A-TOBT",
  "A-DOBT": "A-DOBT",
  target_time: "历史实际移交机坪管制",
  anchor_type: "锚点类型",
  anchor_time: "锚点时间",
  selected_model: "最优算法",
  predicted_handover: "预测实际移交机坪时间",
  predicted_delta_min: "模型修正量_分钟",
  error_min: "预测误差_分钟",
  abs_error_min: "绝对误差_分钟",
  within_3min: "是否<=3分钟",
  within_5min: "是否<=5分钟",
  original_ATOBT_error_min: "原始A-TOBT误差_分钟",
  adobt_baseline_error_min: "A-DOBT基准误差_分钟",
  status: "状态",
  MAE_min: "平均绝对误差_分钟",
  MedianAE_min: "中位绝对误差_分钟",
  RMSE_min: "均方根误差_分钟",
  Within_le_3min_pct: "<=3分钟比例",
  Within_le_5min_pct: "<=5分钟比例",
  original_ATOBT_MAE_min: "原始A-TOBT MAE_分钟",
  adobt_baseline_MAE_min: "A-DOBT基准MAE_分钟",
};

const STATUS_LABELS = {
  ok: "已计算",
  missing_anchor: "A-TOBT和A-DOBT均缺失或无法解析",
};

let currentAirlineRows = [];

function number(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(1)}%`;
}

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.hidden = true;
  }, 5200);
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "计算中..." : "开始计算";
}

function displayValue(key, value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (key === "status") return STATUS_LABELS[value] || value;
  if (key.endsWith("_pct")) return percent(value);
  if (["MAE_min", "MedianAE_min", "RMSE_min", "original_ATOBT_MAE_min", "adobt_baseline_MAE_min", "error_min", "abs_error_min", "predicted_delta_min", "original_ATOBT_error_min", "adobt_baseline_error_min"].includes(key)) {
    return number(value);
  }
  return value;
}

function renderTable(table, rows, columns) {
  table.innerHTML = "";
  if (!rows || rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.textContent = "暂无数据";
    tr.appendChild(td);
    table.appendChild(tr);
    return;
  }
  const keys = columns || Object.keys(rows[0]);
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  keys.forEach((key) => {
    const th = document.createElement("th");
    th.textContent = HEADER_LABELS[key] || key;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    keys.forEach((key) => {
      const td = document.createElement("td");
      const value = row[key];
      td.textContent = displayValue(key, value);
      if (key.includes("within") && value === true) td.className = "good";
      if (key.includes("within") && value === false) td.className = "bad";
      if (key === "abs_error_min" && Number(value) > 5) td.className = "bad";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function prepareCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = chart.getBoundingClientRect();
  const width = Math.max(360, Math.floor(rect.width || chart.parentElement.clientWidth || 980));
  const height = 320;
  const backingWidth = Math.floor(width * dpr);
  const backingHeight = Math.floor(height * dpr);
  if (chart.width !== backingWidth || chart.height !== backingHeight) {
    chart.width = backingWidth;
    chart.height = backingHeight;
  }
  const ctx = chart.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function drawChart(rows) {
  const { ctx, width, height } = prepareCanvas();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!rows || rows.length === 0) {
    ctx.fillStyle = "#63706a";
    ctx.font = "18px Microsoft YaHei, Segoe UI, Arial, sans-serif";
    ctx.fillText("上传包含实际移交机坪管制的文件后显示对比", 28, 54);
    return;
  }

  const data = rows
    .filter((row) => row.MAE_min !== null && row.MAE_min !== undefined)
    .slice(0, 16);
  if (data.length === 0) return;

  const left = 58;
  const right = 24;
  const top = 28;
  const bottom = 58;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const max = Math.max(...data.map((row) => Number(row.MAE_min)), 1);
  const barGap = 8;
  const barW = Math.max(12, (plotW - barGap * (data.length - 1)) / data.length);

  ctx.strokeStyle = "#dbe2de";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = top + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - right, y);
    ctx.stroke();
    ctx.fillStyle = "#63706a";
    ctx.font = "12px Microsoft YaHei, Segoe UI, Arial, sans-serif";
    ctx.fillText(number((max * (4 - i)) / 4, 1), 12, y + 4);
  }

  data.forEach((row, index) => {
    const value = Number(row.MAE_min);
    const x = left + index * (barW + barGap);
    const h = (value / max) * plotH;
    const y = top + plotH - h;
    ctx.fillStyle = value <= 3 ? "#19786f" : value <= 5 ? "#b56c15" : "#b73737";
    ctx.fillRect(x, y, barW, h);
    ctx.fillStyle = "#16211d";
    ctx.font = "11px Microsoft YaHei, Segoe UI, Arial, sans-serif";
    ctx.save();
    ctx.translate(x + barW / 2, height - 14);
    ctx.rotate(-Math.PI / 5);
    ctx.textAlign = "right";
    ctx.fillText(String(row.IFC || "-"), 0, 0);
    ctx.restore();
  });
}

function renderResult(data) {
  const summary = data.summary;
  metricIds.rows.textContent = summary.predictedRows ?? "-";
  metricIds.history.textContent = summary.historicalRows ?? "-";
  metricIds.mae.textContent = number(summary.MAE_min);
  metricIds.m3.textContent = percent(summary.Within_le_3min_pct);
  metricIds.m5.textContent = percent(summary.Within_le_5min_pct);
  metricIds.baseline.textContent = number(summary.original_ATOBT_MAE_min);

  document.getElementById("chartNote").textContent = `${summary.scope} · ${summary.filename}`;
  currentAirlineRows = data.byAirline || [];
  drawChart(currentAirlineRows);
  renderTable(airlineTable, data.byAirline, [
    "IFC",
    "n",
    "MAE_min",
    "MedianAE_min",
    "RMSE_min",
    "Within_le_3min_pct",
    "Within_le_5min_pct",
    "original_ATOBT_MAE_min",
    "adobt_baseline_MAE_min",
  ]);
  renderTable(previewTable, data.preview, [
    "_source_row",
    "IFC",
    "A-TOBT",
    "A-DOBT",
    "target_time",
    "anchor_type",
    "anchor_time",
    "selected_model",
    "predicted_handover",
    "predicted_delta_min",
    "error_min",
    "abs_error_min",
    "within_3min",
    "within_5min",
    "original_ATOBT_error_min",
    "status",
  ]);

  downloadXlsx.href = data.downloads.xlsx;
  downloadCsv.href = data.downloads.csv;
  downloads.hidden = false;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    const pill = document.getElementById("readyPill");
    const status = document.getElementById("modelStatus");
    if (data.ready) {
      pill.textContent = "已就绪";
      status.textContent = `目标：${data.training.target}；A-TOBT最优 ${data.training.atobtGlobalBest}；A-DOBT兜底 ${data.training.adobtGlobalBest}`;
    } else {
      pill.textContent = "启动中";
      status.textContent = "模型准备中";
    }
  } catch (error) {
    document.getElementById("readyPill").textContent = "未连接";
    document.getElementById("modelStatus").textContent = "服务未响应";
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileName.textContent = file ? file.name : "选择 Excel 或 CSV 文件";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) return;
  setBusy(true);
  downloads.hidden = true;
  const body = new FormData();
  body.append("file", fileInput.files[0]);
  body.append("scope", document.getElementById("scopeSelect").value);
  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "计算失败");
    }
    renderResult(data);
    showToast("计算完成，结果已生成");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
});

drawChart([]);
loadStatus();

window.addEventListener("resize", () => {
  drawChart(currentAirlineRows);
});
