const FEATURES = [
  "age",
  "bmi",
  "glucose",
  "systolic",
  "cholesterol",
  "activity",
  "smoking",
  "priorAdmissions",
  "oxygen"
];

function createRandom(seed) {
  let state = seed >>> 0;
  return function random() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function round(value, digits = 1) {
  return Number(value.toFixed(digits));
}

function generateDataset(size = 240, seed = 42) {
  const random = createRandom(seed);
  const records = [];

  for (let index = 0; index < size; index += 1) {
    const age = Math.round(22 + random() * 58);
    const bmi = round(18 + random() * 20, 1);
    const glucose = Math.round(78 + random() * 130);
    const systolic = Math.round(96 + random() * 76);
    const cholesterol = Math.round(145 + random() * 150);
    const activity = round(random() * 8, 1);
    const smoking = random() > 0.73 ? 1 : 0;
    const priorAdmissions = Math.round(random() * 4);
    const oxygen = Math.round(89 + random() * 10);

    const riskSignal =
      -9.5 +
      age * 0.05 +
      bmi * 0.1 +
      glucose * 0.033 +
      systolic * 0.02 +
      cholesterol * 0.009 +
      smoking * 1.1 +
      priorAdmissions * 0.55 -
      activity * 0.25 -
      oxygen * 0.08;

    const probability = sigmoid(riskSignal);
    const label = probability > 0.5 ? 1 : 0;

    records.push({
      id: `PT-${String(index + 1).padStart(3, "0")}`,
      age,
      bmi,
      glucose,
      systolic,
      cholesterol,
      activity,
      smoking,
      priorAdmissions,
      oxygen,
      probability,
      label
    });
  }

  return records;
}

function shuffleRecords(records, seed = 7) {
  const shuffled = [...records];
  const random = createRandom(seed);

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}

function splitDataset(records, ratio = 0.8) {
  const shuffled = shuffleRecords(records);
  const cutoff = Math.floor(shuffled.length * ratio);
  return {
    train: shuffled.slice(0, cutoff),
    test: shuffled.slice(cutoff)
  };
}

function getFeatureStats(records) {
  const stats = {};
  FEATURES.forEach((feature) => {
    const values = records.map((record) => record[feature]);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance =
      values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    stats[feature] = { mean, std: Math.sqrt(variance) || 1 };
  });
  return stats;
}

function normalizeRecord(record, stats) {
  return FEATURES.map((feature) => (record[feature] - stats[feature].mean) / stats[feature].std);
}

function trainLogisticRegression(trainRecords, iterations = 1800, learningRate = 0.09) {
  const stats = getFeatureStats(trainRecords);
  const weights = new Array(FEATURES.length).fill(0);
  let bias = 0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const gradients = new Array(FEATURES.length).fill(0);
    let biasGradient = 0;

    trainRecords.forEach((record) => {
      const vector = normalizeRecord(record, stats);
      const linear = vector.reduce((sum, value, index) => sum + value * weights[index], bias);
      const predicted = sigmoid(linear);
      const error = predicted - record.label;

      vector.forEach((value, index) => {
        gradients[index] += error * value;
      });
      biasGradient += error;
    });

    const scale = 1 / trainRecords.length;
    weights.forEach((weight, index) => {
      weights[index] = weight - learningRate * gradients[index] * scale;
    });
    bias -= learningRate * biasGradient * scale;
  }

  return { weights, bias, stats };
}

function predictProbability(record, model) {
  const vector = normalizeRecord(record, model.stats);
  const linear = vector.reduce(
    (sum, value, index) => sum + value * model.weights[index],
    model.bias
  );
  return sigmoid(linear);
}

function evaluateModel(records, model) {
  let truePositive = 0;
  let trueNegative = 0;
  let falsePositive = 0;
  let falseNegative = 0;

  records.forEach((record) => {
    const predicted = predictProbability(record, model) >= 0.5 ? 1 : 0;
    if (predicted === 1 && record.label === 1) truePositive += 1;
    if (predicted === 0 && record.label === 0) trueNegative += 1;
    if (predicted === 1 && record.label === 0) falsePositive += 1;
    if (predicted === 0 && record.label === 1) falseNegative += 1;
  });

  const total = records.length;
  const accuracy = (truePositive + trueNegative) / total;
  const precision = truePositive / Math.max(truePositive + falsePositive, 1);
  const recall = truePositive / Math.max(truePositive + falseNegative, 1);
  const f1 = (2 * precision * recall) / Math.max(precision + recall, 0.0001);

  return {
    accuracy,
    precision,
    recall,
    f1,
    matrix: { truePositive, trueNegative, falsePositive, falseNegative }
  };
}

function pearsonCorrelation(records, feature) {
  const featureMean =
    records.reduce((sum, record) => sum + record[feature], 0) / records.length;
  const labelMean = records.reduce((sum, record) => sum + record.label, 0) / records.length;

  let numerator = 0;
  let featureVariance = 0;
  let labelVariance = 0;

  records.forEach((record) => {
    const featureDelta = record[feature] - featureMean;
    const labelDelta = record.label - labelMean;
    numerator += featureDelta * labelDelta;
    featureVariance += featureDelta ** 2;
    labelVariance += labelDelta ** 2;
  });

  return numerator / Math.sqrt(featureVariance * labelVariance || 1);
}

function getRiskBands(records, model) {
  const bands = [
    { label: "Low", min: 0, max: 0.35, count: 0 },
    { label: "Moderate", min: 0.35, max: 0.65, count: 0 },
    { label: "High", min: 0.65, max: 1.01, count: 0 }
  ];

  records.forEach((record) => {
    const probability = predictProbability(record, model);
    const band = bands.find((item) => probability >= item.min && probability < item.max);
    band.count += 1;
  });

  return bands;
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function scrollToSection(targetId) {
  const target = document.getElementById(targetId);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    target.classList.remove("section-focus");
    window.setTimeout(() => target.classList.add("section-focus"), 40);
    window.setTimeout(() => target.classList.remove("section-focus"), 1100);
  }
}

function scrollToElement(element) {
  if (element) {
    element.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function highlightElement(element) {
  if (element) {
    element.classList.remove("result-focus");
    window.setTimeout(() => element.classList.add("result-focus"), 40);
    window.setTimeout(() => element.classList.remove("result-focus"), 1100);
  }
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function renderRiskBars(bands, total) {
  const container = document.getElementById("riskBars");
  container.innerHTML = bands
    .map(
      (band) => `
        <div class="bar-row">
          <span>${band.label}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:${(band.count / total) * 100}%"></div>
          </div>
          <strong>${band.count}</strong>
        </div>
      `
    )
    .join("");
}

function renderFeatureImportance(model) {
  const container = document.getElementById("featureImportance");
  const maxWeight = Math.max(...model.weights.map((weight) => Math.abs(weight)));

  container.innerHTML = FEATURES.map((feature, index) => {
    const value = Math.abs(model.weights[index]);
    const width = (value / maxWeight) * 100;
    return `
      <div class="importance-item">
        <span>${feature}</span>
        <div class="importance-track">
          <div class="importance-fill" style="width:${width}%"></div>
        </div>
        <strong>${round(value, 2)}</strong>
      </div>
    `;
  }).join("");
}

function renderCorrelations(records) {
  const container = document.getElementById("correlationGrid");
  container.innerHTML = FEATURES.map((feature) => {
    const value = pearsonCorrelation(records, feature);
    const tone = value > 0 ? "var(--danger)" : "var(--success)";
    return { feature, value, tone };
  })
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
    .slice(0, 6)
    .map(
      ({ feature, value, tone }) => `
      <div class="correlation-card">
        <span>${feature}</span>
        <strong style="color:${tone}">${round(value, 2)}</strong>
      </div>
    `
    )
    .join("");
}

function renderConfusionMatrix(metrics) {
  const { truePositive, trueNegative, falsePositive, falseNegative } = metrics.matrix;
  document.getElementById("confusionMatrix").innerHTML = `
    <div class="confusion-cell">
      <span>True positive</span>
      <strong>${truePositive}</strong>
    </div>
    <div class="confusion-cell">
      <span>True negative</span>
      <strong>${trueNegative}</strong>
    </div>
    <div class="confusion-cell">
      <span>False positive</span>
      <strong>${falsePositive}</strong>
    </div>
    <div class="confusion-cell">
      <span>False negative</span>
      <strong>${falseNegative}</strong>
    </div>
  `;
}

function buildRecommendations(record, probability) {
  const recommendations = [];

  if (record.glucose >= 140) recommendations.push("Escalate glucose monitoring and diabetes screening.");
  if (record.systolic >= 140) recommendations.push("Review blood-pressure management and cardiovascular risk.");
  if (record.oxygen <= 93) recommendations.push("Schedule respiratory evaluation and pulse-ox follow-up.");
  if (record.activity < 3) recommendations.push("Recommend an activity and lifestyle coaching plan.");
  if (record.priorAdmissions >= 2) recommendations.push("Flag for readmission-prevention outreach.");

  if (recommendations.length === 0) {
    recommendations.push("Maintain preventive follow-up and continue baseline monitoring.");
  }

  recommendations.push(
    probability >= 0.65
      ? "High priority: route this case to clinician review within 24 hours."
      : "Moderate priority: include this patient in the next risk review cycle."
  );

  return recommendations;
}

function renderPrediction(record, model) {
  const probability = predictProbability(record, model);
  const badge = document.getElementById("riskBadge");
  const meter = document.getElementById("riskMeterFill");
  const recommendations = buildRecommendations(record, probability);
  const resultNumber = document.getElementById("resultRiskNumber");

  let level = "Low risk";
  let color = "var(--success)";
  if (probability >= 0.65) {
    level = "High risk";
    color = "var(--danger)";
  } else if (probability >= 0.35) {
    level = "Moderate risk";
    color = "var(--orange)";
  }

  badge.textContent = level;
  badge.style.background = color;
  const headline = `${level} patient profile detected`;
  const probabilityText = `${formatPercent(probability)} probability of elevated clinical risk.`;
  setText("riskHeadline", headline);
  setText("riskProbability", probabilityText);
  if (resultNumber) {
    resultNumber.textContent = formatPercent(probability);
    resultNumber.style.color = color;
  }
  meter.style.width = `${Math.round(probability * 100)}%`;
  meter.style.background = `linear-gradient(90deg, ${color}, var(--teal-deep))`;
  document.getElementById("recommendations").innerHTML = recommendations
    .map((item) => `<div class="recommendation-item">${item}</div>`)
    .join("");
}

function getRecordFromForm(form) {
  return {
    age: Number(form.elements.age.value),
    bmi: Number(form.elements.bmi.value),
    glucose: Number(form.elements.glucose.value),
    systolic: Number(form.elements.systolic.value),
    cholesterol: Number(form.elements.cholesterol.value),
    activity: Number(form.elements.activity.value),
    smoking: Number(form.elements.smoking.value),
    priorAdmissions: Number(form.elements.priorAdmissions.value),
    oxygen: Number(form.elements.oxygen.value)
  };
}

function initializeDashboard() {
  const dataset = generateDataset();
  const { train, test } = splitDataset(dataset);
  const model = trainLogisticRegression(train);
  const metrics = evaluateModel(test, model);
  const bands = getRiskBands(dataset, model);
  const highRiskPatients = bands.find((band) => band.label === "High").count;

  setText("patientCount", dataset.length.toString());
  setText("highRiskRate", formatPercent(highRiskPatients / dataset.length));
  setText("modelAccuracy", formatPercent(metrics.accuracy));
  setText(
    "avgGlucose",
    `${round(dataset.reduce((sum, item) => sum + item.glucose, 0) / dataset.length)} mg/dL`
  );
  setText(
    "avgSystolic",
    `${round(dataset.reduce((sum, item) => sum + item.systolic, 0) / dataset.length)} mmHg`
  );

  const expectedAdmissions = Math.max(4, Math.round(highRiskPatients * 0.27));
  setText("admissionForecast", `${expectedAdmissions} next 7 days`);
  setText("staffingPlan", `${Math.max(1, Math.ceil(expectedAdmissions / 6))} extra FTE`);

  setText("qualityAccuracy", formatPercent(metrics.accuracy));
  setText("qualityPrecision", formatPercent(metrics.precision));
  setText("qualityRecall", formatPercent(metrics.recall));
  setText("qualityF1", formatPercent(metrics.f1));

  renderRiskBars(bands, dataset.length);
  renderFeatureImportance(model);
  renderCorrelations(dataset);
  renderConfusionMatrix(metrics);

  document.querySelectorAll(".nav-action").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      scrollToSection(button.dataset.target);
    });
  });

  const form = document.getElementById("riskForm");
  const resultPanel = document.getElementById("resultPanel");
  const predictButton = document.getElementById("predictButton");
  const resetButton = document.getElementById("resetForm");
  const highRiskButton = document.getElementById("loadHighRisk");
  if (!form || !resultPanel || !predictButton || !resetButton || !highRiskButton) return;
  const highRiskSample = {
    age: 68,
    bmi: 34,
    glucose: 190,
    systolic: 165,
    cholesterol: 260,
    activity: 1,
    smoking: 1,
    priorAdmissions: 3,
    oxygen: 91
  };

  highRiskButton.addEventListener("click", () => {
    Object.entries(highRiskSample).forEach(([key, value]) => {
      form.elements[key].value = value;
    });
    renderPrediction(highRiskSample, model);
    scrollToElement(resultPanel);
    highlightElement(resultPanel);
  });

  const defaultSample = {
    age: 58,
    bmi: 31.4,
    glucose: 162,
    systolic: 148,
    cholesterol: 228,
    activity: 2.5,
    smoking: 0,
    priorAdmissions: 1,
    oxygen: 95
  };

  resetButton.addEventListener("click", () => {
    Object.entries(defaultSample).forEach(([key, value]) => {
      form.elements[key].value = value;
    });
    renderPrediction(defaultSample, model);
    scrollToElement(form);
  });

  function runPrediction() {
    if (!form.reportValidity()) return;
    const record = getRecordFromForm(form);
    renderPrediction(record, model);
    scrollToElement(resultPanel);
    highlightElement(resultPanel);
  }

  window.runPrediction = runPrediction;
  predictButton.addEventListener("click", runPrediction);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runPrediction();
  });

  renderPrediction(defaultSample, model);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeDashboard);
} else {
  initializeDashboard();
}
