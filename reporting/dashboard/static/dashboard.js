// Evaluation Dashboard — read-only view over /api/results.
// All report text goes through textContent (never innerHTML): answers,
// contexts and Judge reasons are model output and may contain markup.

const FRAMEWORK_LABELS = { custom: "Custom", ragas: "RAGAS", deepeval: "DeepEval" };
const NO_DIMENSION = "(unmapped)";
const $ = (id) => document.getElementById(id);

let data = null;
let rowsById = new Map();

function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null || value === false) continue;
        if (key === "class") node.className = value;
        else if (key === "text") node.textContent = value;
        else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value);
    }
    for (const child of children.flat(Infinity)) {
        if (child === null || child === undefined) continue;
        node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
}

const fwLabel = (fw) => FRAMEWORK_LABELS[fw] || fw;
const dimLabel = (dim) => dim || NO_DIMENSION;
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const fmt = (v) => (isNum(v) ? (Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")) : "—");
const pct = (v) => (isNum(v) ? `${Math.round(v * 100)}%` : "—");
const statusClass = (s) => (s === "PASS" || s === "FAIL" ? s : "other");

function statusEl(status) {
    const icon = status === "PASS" ? "✓ " : status === "FAIL" ? "✗ " : "";
    return el("span", { class: `status ${statusClass(status)}`, text: icon + status });
}

function scoreBar(score, threshold, status) {
    // Only on the shared 0-1 scale; threshold direction is not stored in the
    // schema, so the tick marks its position without implying min or max.
    if (!isNum(score) || score < 0 || score > 1) return null;
    const bar = el("div", { class: "bar" }, el("div", { class: `fill ${statusClass(status)}`, style: `width:${score * 100}%` }));
    if (isNum(threshold) && threshold >= 0 && threshold <= 1) {
        bar.append(el("div", { class: "tick", style: `left:calc(${threshold * 100}% - 1px)`, title: `threshold ${fmt(threshold)}` }));
    }
    return bar;
}

function scoreCell(row) {
    const text = `${fmt(row.score)}${isNum(row.threshold) ? ` / thr ${fmt(row.threshold)}` : ""}`;
    const native = row.native_score ? el("span", { class: "tag", title: row.native_score.name, text: `raw ${fmt(row.native_score.value)}` }) : null;
    return el("td", {}, el("span", { class: "score", text }), native, scoreBar(row.score, row.threshold, row.status));
}

function rateCell(rate) {
    return el("div", { class: "rate" }, el("span", { class: "num", text: pct(rate) }),
        isNum(rate) ? el("div", { class: "bar" }, el("div", { class: "fill", style: `width:${rate * 100}%;background:var(--accent)` })) : null);
}

// ---------- overview ----------

function renderCards(summary) {
    const o = summary.overall;
    const phases = o.by_phase || {};
    const card = (label, value, sub) => el("div", { class: "card" },
        el("div", { class: "label", text: label }), el("div", { class: "value", text: value }), sub ? el("div", { class: "sub", text: sub }) : null);
    const other = Object.entries(o.by_status).filter(([s]) => s !== "PASS" && s !== "FAIL").map(([s, n]) => `${n} ${s}`).join(", ");
    $("cards").replaceChildren(
        card("Results", o.total, other ? `incl. ${other}` : null),
        card("PASS", o.pass),
        card("FAIL", o.fail),
        card("Pass rate", pct(o.pass_rate), `of ${o.pass + o.fail} gated`),
        card("Frameworks", o.frameworks.length, o.frameworks.map(fwLabel).join(", ")),
        card("Generation / Retrieval", `${phases.generation || 0} / ${phases.retrieval || 0}`),
    );
}

function renderFrameworks(summary) {
    const head = el("tr", {}, ["Framework", "Results", "PASS", "FAIL", "Other", "Pass rate"].map((h, i) => el("th", { class: i >= 1 && i <= 3 ? "num" : null, text: h })));
    const rows = Object.entries(summary.frameworks).map(([fw, c]) => {
        const other = Object.entries(c.by_status).filter(([s]) => s !== "PASS" && s !== "FAIL").map(([s, n]) => `${n} ${s}`).join(", ");
        return el("tr", {},
            el("td", { text: fwLabel(fw) }), el("td", { class: "num", text: c.total }),
            el("td", { class: "num", text: c.pass }), el("td", { class: "num", text: c.fail }),
            el("td", { class: "muted", text: other || "—" }), el("td", {}, rateCell(c.pass_rate)));
    });
    $("framework-table").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}

function renderDimensions(groups, frameworks) {
    const head = el("tr", {}, el("th", { text: "Phase" }), el("th", { text: "Quality dimension" }),
        frameworks.map((fw) => el("th", { text: fwLabel(fw) })));
    const rows = groups.map((g) => el("tr", {},
        el("td", { class: "muted", text: g.phase }),
        el("td", { text: dimLabel(g.quality_dimension) }),
        frameworks.map((fw) => {
            const c = g.frameworks[fw];
            if (!c) return el("td", { class: "muted", text: "—" });
            const gated = c.pass + c.fail;
            const verdict = gated ? `${c.pass}/${gated} PASS` : `${c.total} ${Object.keys(c.by_status).join("/")}`;
            return el("td", {
                class: "cell",
                title: "Filter results to this framework + dimension",
                onclick: () => applyFilters({ framework: fw, phase: g.phase, dimension: dimLabel(g.quality_dimension) }),
            },
            el("div", {}, el("span", { class: c.fail ? "status FAIL" : "", text: verdict })),
            el("div", { class: "metrics", text: c.metrics.join(", ") }));
        })));
    $("dimension-table").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}

function renderAggregates(rows) {
    const aggregates = rows.filter((r) => r.is_aggregate);
    if (!aggregates.length) {
        $("aggregate-table").replaceChildren(el("tbody", {}, el("tr", {}, el("td", { class: "muted", text: "No aggregate metrics in this scope." }))));
        return;
    }
    const head = el("tr", {}, ["Metric", "Suite", "Score / threshold", "Status"].map((h) => el("th", { text: h })));
    const body = aggregates.map((r) => el("tr", { class: "clickable", onclick: () => openDetails(r.evaluation_id) },
        el("td", { text: r.metric }), el("td", { class: "muted", text: `${r.suite} — ${r.query}` }),
        scoreCell(r), el("td", {}, statusEl(r.status))));
    $("aggregate-table").replaceChildren(el("thead", {}, head), el("tbody", {}, body));
}

function renderDisagreements(items) {
    if (!items.length) {
        $("disagreements").replaceChildren(el("p", { class: "muted", text: "No PASS/FAIL disagreements between frameworks in this scope." }));
        return;
    }
    $("disagreements").replaceChildren(...items.map((d) => el("div", {
        class: "disagreement",
        title: "Show these results in the table",
        onclick: () => applyFilters({ text: d.case_id, dimension: d.quality_dimension, disagree: true }),
    },
    el("div", {}, el("strong", { text: d.case_id.split("::").slice(1).join("::") || d.case_id }),
        el("span", { class: "tag", text: d.quality_dimension }), el("span", { class: "tag", text: d.case_id.split("::")[0] })),
    el("div", { class: "verdicts" }, Object.entries(d.frameworks).map(([fw, results]) =>
        el("span", {}, `${fwLabel(fw)}: `, results.map((m, i) => [i ? ", " : "", statusEl(m.status), el("span", { class: "muted", text: ` (${m.metric})` })])))))));
}

// ---------- results table + filters ----------

const FILTERS = {
    framework: { id: "f-framework", value: (r) => r.framework, label: fwLabel },
    phase: { id: "f-phase", value: (r) => r.phase },
    dimension: { id: "f-dimension", value: (r) => dimLabel(r.quality_dimension) },
    metric: { id: "f-metric", value: (r) => r.metric },
    status: { id: "f-status", value: (r) => r.status },
};

function populateFilters(rows) {
    for (const f of Object.values(FILTERS)) {
        const select = $(f.id);
        const previous = select.value;
        const values = [...new Set(rows.map(f.value))].sort();
        select.replaceChildren(select.options[0], ...values.map((v) => el("option", { value: v, text: f.label ? f.label(v) : v })));
        select.value = values.includes(previous) ? previous : "";
    }
}

function applyFilters(values) {
    for (const [key, f] of Object.entries(FILTERS)) $(f.id).value = values[key] || "";
    $("f-text").value = values.text || "";
    $("f-disagree").checked = !!values.disagree;
    renderResults();
    $("results-table").scrollIntoView({ behavior: "smooth", block: "start" });
}

function reasonText(r) {
    // Custom deterministic guardrail failures carry no Judge reason; their
    // stored metadata.failure_stage says which check stopped the case.
    if (r.judge_reason || r.error_message) return r.judge_reason || r.error_message;
    return r.metadata.failure_stage ? `Guardrail failed: ${r.metadata.failure_stage}` : "";
}

function renderResults() {
    const text = $("f-text").value.trim().toLowerCase();
    const onlyDisagree = $("f-disagree").checked;
    const rows = data.rows.filter((r) =>
        Object.values(FILTERS).every((f) => !$(f.id).value || f.value(r) === $(f.id).value)
        && (!onlyDisagree || r.disagreement)
        && (!text || [r.case_id, r.case_name, r.query].some((v) => String(v || "").toLowerCase().includes(text))));

    $("results-table").querySelector("tbody").replaceChildren(...rows.map((r) => el("tr", { onclick: () => openDetails(r.evaluation_id) },
        el("td", { text: fwLabel(r.framework) }),
        el("td", { class: "muted", text: r.phase }),
        el("td", { text: dimLabel(r.quality_dimension) }),
        el("td", { text: r.metric }),
        el("td", { class: "case" },
            el("div", { class: "clip", title: r.case_id, text: r.case_name || r.case_id }),
            r.is_aggregate ? el("span", { class: "tag", text: "aggregate" }) : null,
            r.disagreement ? el("span", { class: "tag", text: "disagreement" }) : null),
        scoreCell(r),
        el("td", {}, statusEl(r.status)),
        el("td", { class: "reason" }, el("div", { class: "clip", text: reasonText(r) })))));

    $("result-count").textContent = `${rows.length} of ${data.rows.length}`;
    $("no-results").hidden = rows.length > 0;
}

// ---------- details ----------

const DETAIL_FIELDS = [
    ["Case", (r) => r.case_name], ["case_id", (r) => r.case_id], ["Framework / phase", (r) => `${fwLabel(r.framework)} / ${r.phase}`],
    ["Suite", (r) => r.suite], ["Metric", (r) => r.metric], ["Quality dimension", (r) => dimLabel(r.quality_dimension)],
    ["Status", (r) => statusEl(r.status)], ["Score", (r) => fmt(r.score)], ["Threshold", (r) => (isNum(r.threshold) ? fmt(r.threshold) : null)],
    ["Native score", (r) => (r.native_score ? `${r.native_score.name} = ${fmt(r.native_score.value)}` : null)],
    ["Previous valid result", (r) => (r.previous_valid ? `${r.previous_valid.status} ${fmt(r.previous_valid.score)} (run ${String(r.previous_valid.run_id).slice(0, 8)}, ${r.previous_valid.completed_at})` : null)],
    ["Query", (r) => r.query], ["Reference answer", (r) => r.reference_answer], ["Assistant answer", (r) => r.assistant_answer],
    ["Judge reason", (r) => r.judge_reason], ["Error", (r) => r.error_message],
    ["Expected / actual confidence", (r) => (r.expected_confidence == null && r.actual_confidence == null ? null : `${r.expected_confidence ?? "—"} / ${r.actual_confidence ?? "—"}`)],
    ["Expected IDs", (r) => listText(r.expected_ids)], ["Retrieved IDs", (r) => listText(r.retrieved_ids)],
    ["Filter", (r) => jsonBlock(r.filter)], ["Context", (r) => listBlock(r.context)], ["Retrieved contexts", (r) => listBlock(r.retrieved_contexts)],
    ["Metadata", (r) => jsonBlock(r.metadata)],
    ["Models", (r) => [["assistant", r.assistant_model], ["judge", r.judge_model], ["embedding", r.embedding_model]].filter(([, m]) => m).map(([k, m]) => `${k}: ${m}`).join("\n") || null],
    ["Run", (r) => `${r.run_id} · ${r.completed_at || "unknown time"}`], ["evaluation_id", (r) => r.evaluation_id],
];

function listText(v) { return Array.isArray(v) && v.length ? v.join(", ") : null; }
function listBlock(v) { return Array.isArray(v) && v.length ? el("pre", { text: v.map((x, i) => `[${i + 1}] ${typeof x === "string" ? x : JSON.stringify(x)}`).join("\n\n") }) : null; }
function jsonBlock(v) { return v && typeof v === "object" && Object.keys(v).length ? el("pre", { text: JSON.stringify(v, null, 2) }) : null; }

function openDetails(evaluationId) {
    const row = rowsById.get(evaluationId);
    if (!row) return;
    $("details-title").textContent = `${fwLabel(row.framework)} · ${row.metric}`;
    const list = el("dl");
    for (const [label, get] of DETAIL_FIELDS) {
        let value;
        try { value = get(row); } catch { value = null; }
        if (value === null || value === undefined || value === "") continue;
        list.append(el("dt", { text: label }), el("dd", {}, value));
    }
    $("details-body").replaceChildren(list);
    $("details").showModal();
}

// ---------- load ----------

async function load() {
    const scope = $("scope").value;
    $("error").hidden = true;
    let payload;
    try {
        const response = await fetch(`/api/results?scope=${encodeURIComponent(scope)}`);
        payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    } catch (err) {
        $("error").textContent = `Could not load evaluation results: ${err.message}`;
        $("error").hidden = false;
        $("content").hidden = true;
        $("source-info").textContent = "";
        return;
    }
    data = payload;
    rowsById = new Map(data.rows.map((r) => [r.evaluation_id, r]));

    const runCount = new Set(data.runs.map((r) => r.run_id)).size;
    $("source-info").textContent = data.source
        ? `Source: ${data.source} · generated ${data.generated_at || "unknown"}${data.scope === "current" ? ` · ${runCount} run(s) across ${data.runs.length} suite(s)` : ""}`
        : "No report files found.";

    $("warnings").hidden = !data.warnings.length;
    $("warnings").replaceChildren(...data.warnings.map((w) => el("div", { text: w })));

    if (!data.rows.length) {
        $("content").hidden = true;
        $("empty").hidden = false;
        $("empty").textContent = data.source
            ? "The report contains no evaluation results for this scope."
            : "No evaluation results yet. Run an evaluation suite to create reports/results/latest.json.";
        return;
    }
    $("empty").hidden = true;
    $("content").hidden = false;

    const frameworks = data.summary.overall.frameworks;
    renderCards(data.summary);
    renderFrameworks(data.summary);
    renderDimensions(data.dimensions, frameworks);
    renderAggregates(data.rows);
    renderDisagreements(data.disagreements);
    populateFilters(data.rows);
    renderResults();
}

for (const f of Object.values(FILTERS)) $(f.id).addEventListener("change", renderResults);
$("f-text").addEventListener("input", renderResults);
$("f-disagree").addEventListener("change", renderResults);
$("f-reset").addEventListener("click", () => applyFilters({}));
$("scope").addEventListener("change", load);
$("details-close").addEventListener("click", () => $("details").close());
$("details").addEventListener("click", (e) => { if (e.target === $("details")) $("details").close(); });
load();
