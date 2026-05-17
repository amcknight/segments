"""Interactive PGM inspector for the learning-curve segment model.

Each node is shown as a card with the actual prior distribution rendered:
  - Latent priors: Normal PDF in unconstrained space, with NATURAL-unit
    tick labels so you can read "bpt is plausibly 1.2s to 500s" directly.
    Vertical lines mark scenario values (truth, MAP, current).
  - Per-attempt curves: shaded prior-predictive band (5-95% + 25-75% + median)
    computed from 500 prior samples. Then truth/current/MAP overlaid on top.
    The band IS the prior; the colored lines are specific points.
  - Observation panels: spaghetti plot of 30 prior-sample clean_dist or
    survival curves (each Lognormal density normalized to unit peak so
    they're visually comparable). Then truth/current/MAP overlaid.

The 'current' scenario is editable via the sliders at the top. Snap buttons
let you jump current to prior-median / truth / MAP as a starting point.

Usage:
    python pgm_inspect.py                       # prior bands + truth
    python pgm_inspect.py --fit-n 500 --seed 1  # also adds MAP scenario
"""
import argparse
import html
import json
import time

import numpy as np

import config
import learning_model
import model
from generate_synthetic_learning import TRUTH_DEFAULTS, generate


# Single-line scenarios (point estimates overlaid on top of the prior bands).
# Note: 'prior median' is NOT a single-line scenario -- it's part of the band.
SCENARIO_COLORS = {
    'truth': '#e67e22',
    'MAP':   '#2980b9',
}
CURRENT_COLOR = '#7b1fa2'


def truth_theta():
    t = TRUTH_DEFAULTS
    return [
        float(np.log(t['bpt'])),
        float(np.log(t['sf_inf'])),
        float(np.log(t['ssp_inf'])),
        float(np.log(t['alpha_inf'])),
        float(t['sf_delta']),
        float(t['ssp_delta']),
        float(t['alpha_delta']),
        float(np.log(t['sf_tau'])),
        float(np.log(t['ssp_tau'])),
        float(np.log(t['alpha_tau'])),
    ]


def maybe_fit(n_attempts, seed):
    from phase1_check import find_map_learning
    data = generate(n_attempts, TRUTH_DEFAULTS, seed=seed)
    t0 = time.perf_counter()
    theta_map, result = find_map_learning(data, learning_model.haz1())
    dt = time.perf_counter() - t0
    info = dict(n_attempts=n_attempts, seed=seed, dt=dt,
                nit=result.nit, converged=bool(result.success),
                n_died=sum(1 for o, _ in data if o == 'died'))
    return [float(x) for x in theta_map], info


LATENT_SPECS = [
    ('log bpt',     0, config.PRIOR_LOG_BPT_MEAN,         config.PRIOR_LOG_BPT_SD,         'log', 'bpt (ms)'),
    ('log sf_inf',  1, config.PRIOR_LOG_SLOP_FRAC_MEAN,   config.PRIOR_LOG_SLOP_FRAC_SD,   'log', 'sf_inf'),
    ('log ssp_inf', 2, config.PRIOR_LOG_SLOP_SPREAD_MEAN, config.PRIOR_LOG_SLOP_SPREAD_SD, 'log', 'ssp_inf'),
    ('log α_inf',   3, config.PRIOR_LOG_HAZ_MEAN,         config.PRIOR_LOG_HAZ_SD,         'log', 'α_haz_inf'),
    ('Δ_sf',        4, learning_model.PRIOR_DELTA_MEAN,   learning_model.PRIOR_DELTA_SD,   'id',  'Δ_sf'),
    ('Δ_ssp',       5, learning_model.PRIOR_DELTA_MEAN,   learning_model.PRIOR_DELTA_SD,   'id',  'Δ_ssp'),
    ('Δ_α',         6, learning_model.PRIOR_DELTA_MEAN,   learning_model.PRIOR_DELTA_SD,   'id',  'Δ_α'),
    ('log τ_sf',    7, learning_model.PRIOR_LOG_TAU_MEAN, learning_model.PRIOR_LOG_TAU_SD, 'log', 'τ_sf'),
    ('log τ_ssp',   8, learning_model.PRIOR_LOG_TAU_MEAN, learning_model.PRIOR_LOG_TAU_SD, 'log', 'τ_ssp'),
    ('log τ_α',     9, learning_model.PRIOR_LOG_TAU_MEAN, learning_model.PRIOR_LOG_TAU_SD, 'log', 'τ_α'),
]

# Plain-language titles for per-attempt nodes
PER_N_SPECS = [
    ('slop_frac (overshoot fraction)',
     'sf(n) = sf_inf · exp(Δ_sf · exp(-n/τ_sf))',
     'sf',
     'log sf_inf, Δ_sf, log τ_sf'),
    ('slop_spread (overshoot CV)',
     'ssp(n) = ssp_inf · exp(Δ_ssp · exp(-n/τ_ssp))',
     'ssp',
     'log ssp_inf, Δ_ssp, log τ_ssp'),
    ('α_haz (cumulative hazard per attempt)',
     'α(n) = α_inf · exp(Δ_α · exp(-n/τ_α))',
     'alphaHaz',
     'log α_inf, Δ_α, log τ_α'),
    ('P(die) per attempt',
     'P(die) = 1 − exp(−α(n))',
     'pDie',
     'α(n)'),
    ('mean clean run overshoot (ms)',
     'mean_slop(n) = sf(n) · bpt',
     'meanSlop',
     'sf(n), log bpt'),
    ('std clean run overshoot (ms)',
     'std_slop(n) = ssp(n) · mean_slop(n)',
     'stdSlop',
     'ssp(n), mean_slop(n)'),
    ('M_clear: expected wall-clock to first clear (s)',
     '(1-p)/p · E[died wall-clock] + E[T_clean]',
     'mClear',
     'all the above'),
]

OBS_NS = [0, 1, 50, 150]


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1340px; margin: 16px auto; padding: 0 14px;
       color: #222; line-height: 1.45; background: #fff; }
h1 { border-bottom: 2px solid #333; padding-bottom: 6px; margin: 6px 0 12px; }
h2 { margin-top: 24px; border-bottom: 1px solid #aaa; padding-bottom: 4px; }
.intro { background: #fafafe; border: 1px solid #ccd; border-radius: 6px;
         padding: 10px 14px; margin: 8px 0; font-size: 13px; }
.intro b { color: #333; }
.legend { font-size: 12px; color: #555; }
.legend-key { display: inline-block; margin-right: 14px; }
.legend-swatch { display: inline-block; width: 14px; height: 3px;
                 vertical-align: middle; margin-right: 4px; }
.legend-band { display: inline-block; width: 14px; height: 8px;
               vertical-align: middle; margin-right: 4px; border-radius: 2px; }
.controls { background: #f7f7fb; border: 1px solid #ccd; border-radius: 6px;
            padding: 8px 12px; margin: 12px 0;
            position: sticky; top: 0; z-index: 10; }
.controls .toolbar { margin-bottom: 6px; }
.controls button { font-size: 12px; padding: 3px 10px; margin-right: 6px;
                   cursor: pointer; border: 1px solid #888; border-radius: 3px;
                   background: #fff; }
.controls button:hover { background: #eef; }
.sliders { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
           gap: 4px 14px; }
.slider-row { display: flex; align-items: center; gap: 6px; font-size: 12px; }
.slider-row label { width: 70px; font-family: "SF Mono", Consolas, monospace;
                    color: #333; }
.slider-row input[type=range] { flex: 1; }
.slider-row .vals { width: 130px; font-family: "SF Mono", Consolas, monospace;
                    color: #444; font-size: 11px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
        gap: 12px; margin: 8px 0; }
.node { border: 1px solid #ddd; border-radius: 6px; padding: 8px 10px;
        background: #fcfcfc; }
.node h3 { margin: 0 0 3px 0; font-size: 13px; color: #333; }
.node h3 .nat { color: #888; font-weight: normal; font-size: 11px; }
.dist { font-family: "SF Mono", Consolas, monospace; color: #06a; font-size: 11px; }
.eq { font-family: "SF Mono", Consolas, monospace; color: #064; font-size: 11px;
      background: #f3faf3; padding: 1px 4px; border-radius: 3px;
      display: inline-block; }
.parents { font-size: 10.5px; color: #888; margin-top: 2px; font-style: italic; }
.node svg { display: block; margin: 4px 0; max-width: 100%; height: auto; }
.values { width: 100%; border-collapse: collapse; font-size: 10.5px;
          margin-top: 2px; }
.values th, .values td { border: 1px solid #e0e0e0; padding: 2px 4px;
                         text-align: right;
                         font-family: "SF Mono", Consolas, monospace; }
.values th { background: #f0f0f0; }
.values td:first-child { text-align: left; background: #fafafa; }
table.sensitivity { border-collapse: collapse; }
table.sensitivity th, table.sensitivity td { border: 1px solid #ccc;
                                              padding: 5px 8px;
                                              font-family: "SF Mono",
                                                Consolas, monospace;
                                              font-size: 12px; }
table.sensitivity th { background: #eef; }
.diff-bad { background: #fcc; }
.diff-warn { background: #ffd; }
.diff-ok { background: #cfc; }
"""


JS_RENDER = r"""
'use strict';

const PLOT_W = 270, PLOT_H = 120;
const RESPAWN_MS = __RESPAWN_MS__;
const PRIOR_MU = __PRIOR_MU__;
const PRIOR_SD = __PRIOR_SD__;
const LATENT_KIND = __LATENT_KIND__;
const LATENT_LABELS = __LATENT_LABELS__;
const NAT_LABELS = __NAT_LABELS__;

const N_DENSE = __N_DENSE__;
const N_GRID = __N_GRID__;
const OBS_NS = __OBS_NS__;
const PER_N_SPECS = __PER_N_SPECS_JS__;

const SCENARIOS = __SCENARIOS_JS__;
SCENARIOS['current'] = {
    theta: (SCENARIOS['truth'] || { theta: PRIOR_MU.slice() }).theta.slice(),
    color: '__CURRENT_COLOR__',
    editable: true,
};

// ----- deterministic PRNG (Mulberry32) -----
function mulberry32(seed) {
    let a = seed >>> 0;
    return function() {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
function makeNormalDraws(rng) {
    // Box-Muller; returns a function that produces standard normal draws.
    let cached = null;
    return function() {
        if (cached !== null) { const v = cached; cached = null; return v; }
        let u1 = rng(); let u2 = rng();
        if (u1 < 1e-300) u1 = 1e-300;
        const r = Math.sqrt(-2 * Math.log(u1));
        const t = 2 * Math.PI * u2;
        cached = r * Math.sin(t);
        return r * Math.cos(t);
    };
}

// ----- eval math (mirrors learning_model.py + model.py) -----
function expCurve(logInf, delta, logTau, n) {
    return Math.exp(logInf + delta * Math.exp(-n / Math.exp(logTau)));
}
function evalAtN(theta, n) {
    const bpt = Math.exp(theta[0]);
    const sf  = expCurve(theta[1], theta[4], theta[7], n);
    const ssp = expCurve(theta[2], theta[5], theta[8], n);
    const alphaHaz = expCurve(theta[3], theta[6], theta[9], n);
    const meanSlop = sf * bpt;
    const sigmaLogSq = Math.log(1.0 + ssp * ssp);
    const muLog = Math.log(meanSlop) - 0.5 * sigmaLogSq;
    const sigmaLog = Math.sqrt(sigmaLogSq);
    const stdSlop = meanSlop * Math.sqrt(Math.exp(sigmaLogSq) - 1.0);
    const pDie = 1.0 - Math.exp(-alphaHaz);
    const pSurv = Math.exp(-alphaHaz);
    const eTClean = bpt + meanSlop;
    let mClear;
    if (pSurv >= 1.0 - 1e-12) mClear = eTClean;
    else {
        let meanSDied;
        if (alphaHaz < 1e-6) meanSDied = 0.5;
        else meanSDied = 1.0 / alphaHaz - Math.exp(-alphaHaz) / (1.0 - Math.exp(-alphaHaz));
        const eDiedTime = meanSDied * eTClean + RESPAWN_MS;
        mClear = (1.0 - pSurv) / pSurv * eDiedTime + eTClean;
    }
    return { bpt, sf, ssp, alphaHaz, meanSlop, stdSlop, pDie,
             mClear: mClear / 1000.0, muLog, sigmaLog };
}
function curveOverN(theta, key, ns) {
    return ns.map(n => evalAtN(theta, n)[key]);
}
function normalPdf(x, mu, sigma) {
    return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2))
         / (sigma * Math.sqrt(2 * Math.PI));
}

// ----- prior sampling -----
const N_PRIOR_SAMPLES = 500;
const N_SPAGHETTI = 30;
const PRIOR_SAMPLES = [];
function samplePrior() {
    const rng = mulberry32(42);
    const randn = makeNormalDraws(rng);
    for (let s = 0; s < N_PRIOR_SAMPLES; s++) {
        const theta = new Array(10);
        for (let i = 0; i < 10; i++) theta[i] = PRIOR_MU[i] + PRIOR_SD[i] * randn();
        PRIOR_SAMPLES.push(theta);
    }
}

// ----- pre-compute prior bands for per-attempt curves -----
const PRIOR_BANDS = {};  // key -> { p5, p25, p50, p75, p95 } arrays at N_DENSE
function computePriorBands() {
    for (const spec of PER_N_SPECS) {
        const N = N_DENSE.length;
        const p5 = new Array(N), p25 = new Array(N), p50 = new Array(N);
        const p75 = new Array(N), p95 = new Array(N);
        for (let i = 0; i < N; i++) {
            const vals = new Array(PRIOR_SAMPLES.length);
            for (let s = 0; s < PRIOR_SAMPLES.length; s++)
                vals[s] = evalAtN(PRIOR_SAMPLES[s], N_DENSE[i])[spec.key];
            vals.sort((a, b) => a - b);
            const L = vals.length;
            p5[i]  = vals[Math.floor(L * 0.05)];
            p25[i] = vals[Math.floor(L * 0.25)];
            p50[i] = vals[Math.floor(L * 0.50)];
            p75[i] = vals[Math.floor(L * 0.75)];
            p95[i] = vals[Math.floor(L * 0.95)];
        }
        PRIOR_BANDS[spec.key] = { p5, p25, p50, p75, p95 };
    }
}

// ----- pre-compute prior spaghetti curves for observation panels -----
// For clean_dist: each sample produces a Lognormal-on-(T - bpt) PDF.
// X-axis: log(T_seconds) from log(0.3) to log(500).
// Y-axis: normalized so each curve peaks at 1 (visual comparability).
const OBS_X_LOG_MIN = Math.log(0.3);
const OBS_X_LOG_MAX = Math.log(500);
const PRIOR_SPAG_CLEAN = {};   // n -> array of {xs (log T_s), ys (normalized)}
const PRIOR_SPAG_SURV  = {};   // n -> array of {xs, ys}
function computePriorSpaghetti() {
    const samples = PRIOR_SAMPLES.slice(0, N_SPAGHETTI);
    for (const n of OBS_NS) {
        PRIOR_SPAG_CLEAN[n] = samples.map(theta => {
            const v = evalAtN(theta, n);
            const xs = [], ys = [];
            for (let i = 0; i <= 100; i++) {
                const logT = OBS_X_LOG_MIN + (OBS_X_LOG_MAX - OBS_X_LOG_MIN) * i / 100;
                const Tms = Math.exp(logT) * 1000;
                let y = 0;
                if (Tms > v.bpt) {
                    const z = (Math.log(Tms - v.bpt) - v.muLog) / v.sigmaLog;
                    y = Math.exp(-0.5 * z * z) /
                        ((Tms - v.bpt) * v.sigmaLog * Math.sqrt(2 * Math.PI));
                }
                xs.push(logT); ys.push(y);
            }
            // normalize to peak = 1 for visual comparability
            const ymax = Math.max(...ys);
            if (ymax > 0) for (let i = 0; i < ys.length; i++) ys[i] /= ymax;
            return { xs, ys };
        });
        PRIOR_SPAG_SURV[n] = samples.map(theta => {
            const v = evalAtN(theta, n);
            const xs = [], ys = [];
            for (let i = 0; i <= 100; i++) {
                const sX = i / 100;
                xs.push(sX); ys.push(Math.exp(-v.alphaHaz * sX));
            }
            return { xs, ys };
        });
    }
}

// ----- SVG helpers -----
function scaleLin(x, xMin, xMax, lo, hi) {
    return lo + (x - xMin) / (xMax - xMin) * (hi - lo);
}
function svgEl(tag, attrs, text) {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
}
function clearSvg(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
function pathFromPoints(pts) {
    if (pts.length === 0) return '';
    let d = `M ${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; i++)
        d += ` L ${pts[i][0].toFixed(1)},${pts[i][1].toFixed(1)}`;
    return d;
}

// ----- helper: format natural-unit value for tick labels -----
function fmtNat(v, kind, idx) {
    if (kind === 'id') return v.toFixed(2);
    // log-kind: convert to natural and format
    const n = Math.exp(v);
    // Heuristic: special handling for bpt (idx 0, in ms) -> show seconds
    if (idx === 0) {
        const sec = n / 1000;
        if (sec >= 100) return sec.toFixed(0) + 's';
        if (sec >= 10) return sec.toFixed(0) + 's';
        if (sec >= 1) return sec.toFixed(1) + 's';
        return (sec * 1000).toFixed(0) + 'ms';
    }
    // tau in attempts (idx 7-9)
    if (idx >= 7) return n < 10 ? n.toFixed(1) : n.toFixed(0);
    // sf, ssp, alpha
    if (n >= 100) return n.toFixed(0);
    if (n >= 1) return n.toFixed(2);
    if (n >= 0.01) return n.toFixed(3);
    return n.toExponential(1);
}

// ----- render: latent prior -----
function renderLatentPrior(idx) {
    const svg = document.getElementById('prior-svg-' + idx);
    if (!svg) return;
    clearSvg(svg);
    const mu = PRIOR_MU[idx], sd = PRIOR_SD[idx];
    const xMin = mu - 3 * sd, xMax = mu + 3 * sd;
    const padL = 4, padR = 4, padT = 6, padB = 24;
    const xs = [];
    for (let i = 0; i <= 80; i++) xs.push(xMin + (xMax - xMin) * i / 80);
    const ys = xs.map(x => normalPdf(x, mu, sd));
    const yMax = Math.max(...ys) * 1.15;
    const px = x => scaleLin(x, xMin, xMax, padL, PLOT_W - padR);
    const py = y => scaleLin(y, 0, yMax, PLOT_H - padB, padT);

    // PDF curve filled
    const pdfPts = xs.map((x, i) => [px(x), py(ys[i])]);
    pdfPts.push([px(xMax), py(0)]);
    pdfPts.unshift([px(xMin), py(0)]);
    svg.appendChild(svgEl('path', {
        d: pathFromPoints(pdfPts) + ' Z',
        fill: '#dddddd', stroke: '#888', 'stroke-width': 1.0
    }));

    // Tick labels in NATURAL units
    const kind = LATENT_KIND[idx];
    const tickThetas = [mu - 2 * sd, mu, mu + 2 * sd];
    for (const tv of tickThetas) {
        svg.appendChild(svgEl('line', {
            x1: px(tv), y1: py(0), x2: px(tv), y2: py(0) + 3,
            stroke: '#888', 'stroke-width': 0.6 }));
        svg.appendChild(svgEl('text', {
            x: px(tv).toFixed(1), y: (PLOT_H - 13).toFixed(0),
            'font-size': 9, 'text-anchor': 'middle', fill: '#444' },
            fmtNat(tv, kind, idx)));
    }

    // Vertical marks per (overlay) scenario
    for (const [name, s] of Object.entries(SCENARIOS)) {
        let v = s.theta[idx];
        let mx;
        if (v < xMin) mx = px(xMin) + 3;
        else if (v > xMax) mx = px(xMax) - 3;
        else mx = px(v);
        svg.appendChild(svgEl('line', {
            x1: mx.toFixed(1), y1: padT, x2: mx.toFixed(1), y2: py(0),
            stroke: s.color, 'stroke-width': s.editable ? 2.4 : 1.8 }));
    }
    svg.appendChild(svgEl('text', {
        x: PLOT_W / 2, y: PLOT_H - 2,
        'font-size': 10, 'text-anchor': 'middle', fill: '#444' },
        '← μ−2σ . μ . μ+2σ → (natural units)'));
}

// ----- render: curve over n with prior band -----
function renderCurveOverN(specIdx) {
    const spec = PER_N_SPECS[specIdx];
    const svg = document.getElementById('curve-svg-' + specIdx);
    if (!svg) return;
    clearSvg(svg);
    const padL = 42, padR = 8, padT = 8, padB = 22;
    const band = PRIOR_BANDS[spec.key];

    // Compute scenario curves
    const seriesData = {};
    for (const [name, s] of Object.entries(SCENARIOS)) {
        seriesData[name] = curveOverN(s.theta, spec.key, N_DENSE);
    }

    // Y-range and scale: switch to log-y if the prior band spans > 30x
    // (otherwise the absurd prior tails would crush scenarios flat).
    let allY = band.p5.concat(band.p95);
    for (const ys of Object.values(seriesData)) allY = allY.concat(ys);
    allY = allY.filter(v => Number.isFinite(v));
    let yLoRaw = Math.min(...allY), yHiRaw = Math.max(...allY);
    const allPositive = yLoRaw > 0;
    const useLog = allPositive && (yHiRaw / Math.max(yLoRaw, 1e-12) > 30);

    let yMin, yMax;
    if (useLog) {
        yMin = Math.log(Math.max(yLoRaw, 1e-9));
        yMax = Math.log(yHiRaw);
        const r = yMax - yMin;
        yMin -= 0.05 * r; yMax += 0.05 * r;
    } else {
        yMin = yLoRaw; yMax = yHiRaw;
        const r = Math.max(yMax - yMin, 1e-9);
        yMin -= 0.05 * r; yMax += 0.05 * r;
    }

    const xMin = N_DENSE[0], xMax = N_DENSE[N_DENSE.length - 1];
    const px = x => scaleLin(x, xMin, xMax, padL, PLOT_W - padR);
    const py = y => {
        const yv = useLog ? Math.log(Math.max(y, 1e-12)) : y;
        return scaleLin(yv, yMin, yMax, PLOT_H - padB, padT);
    };

    // ---- prior 5-95% outer band ----
    const upperPts95 = N_DENSE.map((n, i) => [px(n), py(band.p95[i])]);
    const lowerPts05 = N_DENSE.map((n, i) => [px(n), py(band.p5[i])]).reverse();
    svg.appendChild(svgEl('path', {
        d: pathFromPoints(upperPts95.concat(lowerPts05)) + ' Z',
        fill: '#d0d6dd', stroke: 'none', opacity: 0.65,
    }));
    // ---- prior 25-75% inner band ----
    const upperPts75 = N_DENSE.map((n, i) => [px(n), py(band.p75[i])]);
    const lowerPts25 = N_DENSE.map((n, i) => [px(n), py(band.p25[i])]).reverse();
    svg.appendChild(svgEl('path', {
        d: pathFromPoints(upperPts75.concat(lowerPts25)) + ' Z',
        fill: '#9aa5b1', stroke: 'none', opacity: 0.55,
    }));
    // ---- prior median line ----
    svg.appendChild(svgEl('path', {
        d: pathFromPoints(N_DENSE.map((n, i) => [px(n), py(band.p50[i])])),
        fill: 'none', stroke: '#555', 'stroke-width': 1.2,
        'stroke-dasharray': '3,2',
    }));

    // ---- axes ----
    svg.appendChild(svgEl('line', { x1: padL, y1: padT, x2: padL,
        y2: PLOT_H - padB, stroke: '#777', 'stroke-width': 0.6 }));
    svg.appendChild(svgEl('line', { x1: padL, y1: PLOT_H - padB,
        x2: PLOT_W - padR, y2: PLOT_H - padB,
        stroke: '#777', 'stroke-width': 0.6 }));
    const yTickRaw = useLog
        ? [Math.exp(yMin + 0.05*(yMax-yMin)),
           Math.exp(0.5*(yMin+yMax)),
           Math.exp(yMax - 0.05*(yMax-yMin))]
        : [yMin + 0.05*(yMax-yMin), 0.5*(yMin+yMax), yMax - 0.05*(yMax-yMin)];
    for (const yt of yTickRaw) {
        const yp = py(yt);
        let label;
        if (!Number.isFinite(yt)) label = '—';
        else if (Math.abs(yt) >= 1e6) label = yt.toExponential(1);
        else if (Math.abs(yt) >= 1000) label = yt.toFixed(0);
        else label = yt.toPrecision(3);
        svg.appendChild(svgEl('text', { x: padL - 4, y: (yp + 3).toFixed(1),
            'font-size': 9, 'text-anchor': 'end', fill: '#666' }, label));
    }
    if (useLog) {
        svg.appendChild(svgEl('text', { x: padL - 4, y: padT + 2,
            'font-size': 8, 'text-anchor': 'end', fill: '#888' }, 'log-y'));
    }
    for (const xt of [0, 50, 150, 500]) {
        if (xt < xMin || xt > xMax) continue;
        const xp = px(xt);
        svg.appendChild(svgEl('text', { x: xp.toFixed(1),
            y: (PLOT_H - padB + 12).toFixed(1),
            'font-size': 9, 'text-anchor': 'middle', fill: '#666' },
            'n=' + xt));
    }

    // ---- scenarios overlaid ----
    for (const [name, ys] of Object.entries(seriesData)) {
        const color = SCENARIOS[name].color;
        const pts = N_DENSE.map((n, i) => [px(n), py(ys[i])]);
        svg.appendChild(svgEl('path', {
            d: pathFromPoints(pts), fill: 'none', stroke: color,
            'stroke-width': SCENARIOS[name].editable ? 2.4 : 1.8 }));
        for (const mn of N_GRID) {
            if (mn < xMin || mn > xMax) continue;
            let yv = 0;
            for (let j = 0; j < N_DENSE.length - 1; j++) {
                if (N_DENSE[j] <= mn && N_DENSE[j+1] >= mn) {
                    const t = (mn - N_DENSE[j]) / (N_DENSE[j+1] - N_DENSE[j]);
                    yv = ys[j] + t * (ys[j+1] - ys[j]);
                    break;
                }
            }
            if (mn === N_DENSE[N_DENSE.length - 1]) yv = ys[ys.length - 1];
            svg.appendChild(svgEl('circle', {
                cx: px(mn).toFixed(1), cy: py(yv).toFixed(1),
                r: 2.2, fill: color }));
        }
    }
}

// ----- render: observation at fixed n with spaghetti -----
function renderObs(plotId, n, kind) {
    const svg = document.getElementById(plotId);
    if (!svg) return;
    clearSvg(svg);
    const padL = 42, padR = 8, padT = 8, padB = 22;

    // Pre-computed prior spaghetti
    const prior = (kind === 'clean_dist') ? PRIOR_SPAG_CLEAN[n] : PRIOR_SPAG_SURV[n];

    // Compute scenario curves
    const sceneCurves = {};
    let allY = [];
    for (const [name, s] of Object.entries(SCENARIOS)) {
        const v = evalAtN(s.theta, n);
        const xs = [], ys = [];
        if (kind === 'clean_dist') {
            for (let i = 0; i <= 100; i++) {
                const logT = OBS_X_LOG_MIN + (OBS_X_LOG_MAX - OBS_X_LOG_MIN) * i / 100;
                const Tms = Math.exp(logT) * 1000;
                let y = 0;
                if (Tms > v.bpt) {
                    const z = (Math.log(Tms - v.bpt) - v.muLog) / v.sigmaLog;
                    y = Math.exp(-0.5 * z * z) /
                        ((Tms - v.bpt) * v.sigmaLog * Math.sqrt(2 * Math.PI));
                }
                xs.push(logT); ys.push(y);
            }
            const ymax = Math.max(...ys);
            if (ymax > 0) for (let i = 0; i < ys.length; i++) ys[i] /= ymax;
        } else {
            for (let i = 0; i <= 100; i++) {
                const sX = i / 100;
                xs.push(sX); ys.push(Math.exp(-v.alphaHaz * sX));
            }
        }
        sceneCurves[name] = { xs, ys };
        allY = allY.concat(ys);
    }

    // x/y ranges
    let xMin, xMax;
    if (kind === 'clean_dist') {
        xMin = OBS_X_LOG_MIN; xMax = OBS_X_LOG_MAX;
    } else {
        xMin = 0; xMax = 1;
    }
    const yMin = 0;
    let yMax = 1.1;
    if (kind !== 'clean_dist') yMax = 1.05;

    const px = x => scaleLin(x, xMin, xMax, padL, PLOT_W - padR);
    const py = y => scaleLin(y, yMin, yMax, PLOT_H - padB, padT);

    // ---- prior spaghetti (thin transparent gray lines) ----
    for (const curve of prior) {
        const pts = curve.xs.map((x, i) => [px(x), py(curve.ys[i])]);
        svg.appendChild(svgEl('path', {
            d: pathFromPoints(pts), fill: 'none',
            stroke: '#7a8794', 'stroke-width': 0.7, opacity: 0.18 }));
    }
    // ---- axes ----
    svg.appendChild(svgEl('line', { x1: padL, y1: padT, x2: padL,
        y2: PLOT_H - padB, stroke: '#777', 'stroke-width': 0.6 }));
    svg.appendChild(svgEl('line', { x1: padL, y1: PLOT_H - padB,
        x2: PLOT_W - padR, y2: PLOT_H - padB,
        stroke: '#777', 'stroke-width': 0.6 }));
    // x-axis ticks
    if (kind === 'clean_dist') {
        const tickSecs = [0.5, 1, 5, 10, 30, 60, 150, 500];
        for (const ts of tickSecs) {
            const lx = Math.log(ts);
            if (lx < xMin - 1e-6 || lx > xMax + 1e-6) continue;
            const xp = px(lx);
            svg.appendChild(svgEl('line', { x1: xp.toFixed(1),
                y1: PLOT_H - padB, x2: xp.toFixed(1),
                y2: PLOT_H - padB + 3, stroke: '#777', 'stroke-width': 0.6 }));
            svg.appendChild(svgEl('text', { x: xp.toFixed(1),
                y: (PLOT_H - padB + 12).toFixed(1),
                'font-size': 9, 'text-anchor': 'middle', fill: '#666' },
                ts < 1 ? ts.toFixed(1) + 's' : ts.toFixed(0) + 's'));
        }
    } else {
        for (const xt of [0, 0.5, 1]) {
            const xp = px(xt);
            svg.appendChild(svgEl('text', { x: xp.toFixed(1),
                y: (PLOT_H - padB + 12).toFixed(1), 'font-size': 9,
                'text-anchor': 'middle', fill: '#666' },
                xt.toFixed(1)));
        }
    }
    // y-axis ticks: 0 and 1 for survival; just 0 / peak for normalized clean_dist
    for (const yt of (kind === 'survival' ? [0, 0.5, 1] : [0, 0.5, 1])) {
        const yp = py(yt);
        svg.appendChild(svgEl('text', { x: padL - 4, y: (yp + 3).toFixed(1),
            'font-size': 9, 'text-anchor': 'end', fill: '#666' },
            yt.toFixed(1)));
    }
    // ---- scenario curves overlaid ----
    for (const [name, c] of Object.entries(sceneCurves)) {
        const color = SCENARIOS[name].color;
        const pts = c.xs.map((x, i) => [px(x), py(c.ys[i])]);
        svg.appendChild(svgEl('path', {
            d: pathFromPoints(pts), fill: 'none', stroke: color,
            'stroke-width': SCENARIOS[name].editable ? 2.4 : 1.8 }));
    }
    svg.appendChild(svgEl('text', { x: PLOT_W / 2, y: PLOT_H - 2,
        'font-size': 10, 'text-anchor': 'middle', fill: '#444' },
        kind === 'clean_dist' ? 'T_clean (s, log scale; each curve normalized)'
                              : 's = τ/T_clean'));
}

// ----- values tables -----
function fmt(v) {
    if (v == null || !Number.isFinite(v)) return '—';
    if (Math.abs(v) >= 1000) return v.toFixed(0);
    return v.toPrecision(4);
}
function updateValueTables() {
    for (let i = 0; i < 10; i++) {
        const tbl = document.getElementById('latent-vals-' + i);
        if (!tbl) continue;
        const kind = LATENT_KIND[i];
        let h = `<tr><th>scenario</th><th>uncon</th><th>${NAT_LABELS[i]}</th></tr>`;
        // prior median row (first)
        const tvP = PRIOR_MU[i];
        const natP = kind === 'log' ? Math.exp(tvP) : tvP;
        h += `<tr><td>prior median</td><td>${tvP.toFixed(3)}</td><td>${fmt(natP)}</td></tr>`;
        // prior 95% CI in natural units
        const lo = PRIOR_MU[i] - 1.96 * PRIOR_SD[i];
        const hi = PRIOR_MU[i] + 1.96 * PRIOR_SD[i];
        const natLo = kind === 'log' ? Math.exp(lo) : lo;
        const natHi = kind === 'log' ? Math.exp(hi) : hi;
        h += `<tr><td>prior 95% CI</td><td>[${lo.toFixed(2)}, ${hi.toFixed(2)}]</td>` +
             `<td>[${fmt(natLo)}, ${fmt(natHi)}]</td></tr>`;
        for (const [name, s] of Object.entries(SCENARIOS)) {
            const tv = s.theta[i];
            const nat = kind === 'log' ? Math.exp(tv) : tv;
            h += `<tr><td>${name}</td><td>${tv.toFixed(3)}</td><td>${fmt(nat)}</td></tr>`;
        }
        tbl.innerHTML = h;
    }
    for (let i = 0; i < PER_N_SPECS.length; i++) {
        const tbl = document.getElementById('curve-vals-' + i);
        if (!tbl) continue;
        const band = PRIOR_BANDS[PER_N_SPECS[i].key];
        let h = '<tr><th>n</th><th>prior 5%</th><th>prior 50%</th><th>prior 95%</th>';
        for (const name of Object.keys(SCENARIOS)) h += `<th>${name}</th>`;
        h += '</tr>';
        for (const nv of N_GRID) {
            let idx = N_DENSE.indexOf(nv);
            if (idx < 0) {
                for (let j = 0; j < N_DENSE.length - 1; j++) {
                    if (N_DENSE[j] <= nv && N_DENSE[j+1] >= nv) { idx = j; break; }
                }
            }
            h += `<tr><td>n=${nv}</td>` +
                 `<td>${fmt(band.p5[idx])}</td>` +
                 `<td>${fmt(band.p50[idx])}</td>` +
                 `<td>${fmt(band.p95[idx])}</td>`;
            for (const [name, s] of Object.entries(SCENARIOS)) {
                const v = evalAtN(s.theta, nv)[PER_N_SPECS[i].key];
                h += `<td>${fmt(v)}</td>`;
            }
            h += '</tr>';
        }
        tbl.innerHTML = h;
    }
}

// ----- slider wiring -----
function updateSliderDisplays() {
    for (let i = 0; i < 10; i++) {
        const tv = SCENARIOS['current'].theta[i];
        const nat = LATENT_KIND[i] === 'log' ? Math.exp(tv) : tv;
        const el = document.getElementById('slider-nat-' + i);
        if (el) el.textContent = `${tv.toFixed(3)} → ${fmt(nat)}`;
    }
}
function renderAll() {
    for (let i = 0; i < 10; i++) renderLatentPrior(i);
    for (let i = 0; i < PER_N_SPECS.length; i++) renderCurveOverN(i);
    for (const n of OBS_NS) {
        renderObs('cd-' + n, n, 'clean_dist');
        renderObs('surv-' + n, n, 'survival');
    }
    updateValueTables();
    updateSliderDisplays();
}
function attachSliders() {
    for (let i = 0; i < 10; i++) {
        const s = document.getElementById('slider-' + i);
        if (!s) continue;
        s.addEventListener('input', (e) => {
            SCENARIOS['current'].theta[i] = parseFloat(e.target.value);
            renderAll();
        });
    }
}
function snap(to) {
    let target;
    if (to === 'prior median') target = PRIOR_MU.slice();
    else if (SCENARIOS[to]) target = SCENARIOS[to].theta.slice();
    else return;
    SCENARIOS['current'].theta = target;
    for (let i = 0; i < 10; i++) {
        const s = document.getElementById('slider-' + i);
        if (s) s.value = SCENARIOS['current'].theta[i];
    }
    renderAll();
}
window.__snap = snap;

// ----- init -----
function init() {
    samplePrior();
    computePriorBands();
    computePriorSpaghetti();
    attachSliders();
    renderAll();
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
"""


def build_html(scenarios_data, fit_info=None):
    """scenarios_data: ordered dict {label: theta_list}. May contain 'truth',
    'MAP'. 'prior mode' is NOT a scenario any more -- prior is the gray band."""
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>Learning-curve PGM inspector</title>")
    parts.append(f"<style>{CSS}</style>")
    parts.append("</head><body>")
    parts.append("<h1>Speedrun Segment Model — PGM Inspector</h1>")

    # ---- Intro ----
    parts.append("<div class='intro'>")
    parts.append("<b>What you're looking at:</b> the model has 10 latent "
                 "parameters per segment. Each parameter has a prior "
                 "distribution. From those latents, we deduce per-attempt "
                 "predictions (slop_frac, slop_spread, hazard rate per n) and "
                 "observable distributions (clean run times, survival).<br/><br/>")
    parts.append("<b>Reading the plots:</b> the <span style='background:#d0d6dd;"
                 "padding:0 3px'>gray bands</span> and "
                 "<span style='color:#7a8794'>thin spaghetti</span> are the "
                 "<b>prior</b> — sampled 500 times from the joint prior. "
                 "The band on per-attempt curves spans the 5th–95th percentile "
                 "of what the prior thinks the curve could look like at each n. "
                 "The spaghetti on observation panels shows 30 individual "
                 "prior-sampled clean_dist or survival curves.<br/><br/>")
    parts.append("Colored lines on top are <b>specific parameter points</b>: "
                 "<span style='color:#e67e22'><b>truth</b></span> (the synth "
                 "generator's values), ")
    if 'MAP' in scenarios_data:
        parts.append("<span style='color:#2980b9'><b>MAP</b></span> (a fitted "
                     "point estimate), ")
    parts.append(f"and <span style='color:{CURRENT_COLOR}'><b>current</b></span> "
                 "(editable via the sliders below).</div>")

    # ---- Legend ----
    parts.append("<p class='legend'>")
    parts.append("<span class='legend-key'><span class='legend-band' "
                 "style='background:#d0d6dd'></span>prior 5-95% / "
                 "<span class='legend-band' "
                 "style='background:#9aa5b1'></span>25-75% / "
                 "<span class='legend-band' "
                 "style='background:#7a8794;height:2px;width:18px'></span>"
                 "spaghetti</span>")
    for lbl in list(scenarios_data.keys()) + ['current']:
        color = CURRENT_COLOR if lbl == 'current' else SCENARIO_COLORS.get(lbl, CURRENT_COLOR)
        parts.append(f"<span class='legend-key'><span class='legend-swatch' "
                     f"style='background:{color}'></span>{html.escape(lbl)}</span>")
    parts.append("</p>")

    if fit_info:
        parts.append(f"<p class='legend'>Fit: N={fit_info['n_attempts']} "
                     f"seed={fit_info['seed']}, {fit_info['nit']} iters, "
                     f"{fit_info['dt']:.1f}s, "
                     f"converged={fit_info['converged']}, "
                     f"{fit_info['n_died']} died.</p>")

    # ---- Controls: sliders + snap buttons ----
    parts.append("<div class='controls'>")
    parts.append("<div class='toolbar'>")
    parts.append("<b style='margin-right:10px'>'current' (purple) — slide to "
                 "explore. Snap to:</b>")
    parts.append("<button onclick='__snap(\"prior median\")'>"
                 "prior median</button>")
    if 'truth' in scenarios_data:
        parts.append("<button onclick='__snap(\"truth\")'>truth</button>")
    if 'MAP' in scenarios_data:
        parts.append("<button onclick='__snap(\"MAP\")'>MAP</button>")
    parts.append("</div>")
    parts.append("<div class='sliders'>")
    for label, idx, mu, sd, kind, nat_label in LATENT_SPECS:
        s_min = mu - 3 * sd
        s_max = mu + 3 * sd
        step = sd / 50.0
        init = scenarios_data.get('truth', [PRIOR_DEFAULT_FALLBACK[idx]
                                             for idx in range(10)])[idx] \
            if 'truth' in scenarios_data else mu
        parts.append(
            f"<div class='slider-row'>"
            f"<label title='{html.escape(label)}'>{html.escape(label)}</label>"
            f"<input type='range' id='slider-{idx}' "
            f"min='{s_min}' max='{s_max}' step='{step}' value='{init}'/>"
            f"<span class='vals' id='slider-nat-{idx}'>–</span>"
            f"</div>"
        )
    parts.append("</div></div>")

    # ---- Latents section ----
    parts.append("<h2>Latent parameters — prior distributions</h2>")
    parts.append("<p class='legend'>Each is Normal in unconstrained "
                 "(log/identity) space. Tick labels are in NATURAL units "
                 "(seconds, fractions, attempts). The colored vertical lines "
                 "are where each scenario sits on this prior.</p>")
    parts.append("<div class='grid'>")
    for label, idx, mu, sd, kind, nat_label in LATENT_SPECS:
        parts.append(
            f"<div class='node'>"
            f"<h3>{html.escape(label)} <span class='nat'>→ "
            f"{html.escape(nat_label)}</span></h3>"
            f"<div><span class='dist'>Normal(μ={mu:.3f}, σ={sd:.2f})</span></div>"
            f"<svg id='prior-svg-{idx}' width='270' height='120' "
            f"viewBox='0 0 270 120'></svg>"
            f"<table class='values' id='latent-vals-{idx}'></table>"
            f"</div>"
        )
    parts.append("</div>")

    # ---- Per-attempt deterministic ----
    parts.append("<h2>Player behavior over attempts (n = 0 → 500)</h2>")
    parts.append("<p class='legend'>Each plot shows one node evolving over "
                 "attempts. <b>Gray band</b> = where the prior thinks the "
                 "curve could be (5-95% and 25-75% percentile bands across "
                 "500 prior samples), <b>dashed line</b> = prior median. "
                 "Colored lines = specific scenarios.</p>")
    parts.append("<div class='grid'>")
    for i, (label, eq, key, parents) in enumerate(PER_N_SPECS):
        parts.append(
            f"<div class='node'>"
            f"<h3>{html.escape(label)}</h3>"
            f"<div><span class='eq'>{html.escape(eq)}</span></div>"
            f"<div class='parents'>parents: {html.escape(parents)}</div>"
            f"<svg id='curve-svg-{i}' width='270' height='120' "
            f"viewBox='0 0 270 120'></svg>"
            f"<table class='values' id='curve-vals-{i}'></table>"
            f"</div>"
        )
    parts.append("</div>")

    # ---- Observation ----
    parts.append("<h2>Predicted observations at each attempt n</h2>")
    parts.append("<p class='legend'>For each n, the model predicts a "
                 "distribution of clean-run times (T_clean) and a survival "
                 "curve. <b>Thin gray spaghetti</b> = 30 individual prior-"
                 "sample curves (each clean_dist normalized to unit peak so "
                 "they're visually comparable on the log-T axis). "
                 "Colored lines = specific scenarios.</p>")
    parts.append("<div class='grid'>")
    for n in OBS_NS:
        parts.append(
            f"<div class='node'>"
            f"<h3>Clean run time T_clean at attempt n={n}</h3>"
            f"<div><span class='eq'>bpt + Lognormal(μ_log(n), σ_log(n))</span></div>"
            f"<div class='parents'>parents: log bpt, sf(n), ssp(n)</div>"
            f"<svg id='cd-{n}' width='270' height='120' "
            f"viewBox='0 0 270 120'></svg>"
            f"</div>"
        )
        parts.append(
            f"<div class='node'>"
            f"<h3>Probability of being alive at fraction s, n={n}</h3>"
            f"<div><span class='eq'>S(s) = exp(-α(n)·s)</span></div>"
            f"<div class='parents'>parents: α(n)</div>"
            f"<svg id='surv-{n}' width='270' height='120' "
            f"viewBox='0 0 270 120'></svg>"
            f"</div>"
        )
    parts.append("</div>")

    # ---- Sensitivity table ----
    parts.append("<h2>Prior 1σ sensitivity on M_clear(1)</h2>")
    parts.append("<p class='legend'>Move each latent ±1 prior SD from the "
                 "prior median (others fixed). Red = M_clear(1) spans more "
                 "than 10× across the ±1σ excursion; yellow = more than 3×; "
                 "green = under 3×. Red rows are essentially uninformative "
                 "on observable behavior.</p>")
    parts.append("<table class='sensitivity'>")
    parts.append("<tr><th>Latent</th><th>Prior SD</th>"
                 "<th>M_clear(1) at −1σ</th><th>at prior median</th>"
                 "<th>at +1σ</th><th>ratio</th></tr>")
    haz = learning_model.haz1()
    theta_prior = learning_model.initial_theta(haz)
    base_mc = learning_model.m_clear_at_n(theta_prior, haz, 1) / 1000.0
    for label, idx, _mu, sd, _kind, _ in LATENT_SPECS:
        tp = theta_prior.copy(); tp[idx] += sd
        tm = theta_prior.copy(); tm[idx] -= sd
        mc_p = learning_model.m_clear_at_n(tp, haz, 1) / 1000.0
        mc_m = learning_model.m_clear_at_n(tm, haz, 1) / 1000.0
        ratio = max(mc_p, mc_m) / max(1e-9, min(mc_p, mc_m))
        cls = 'diff-bad' if ratio > 10 else ('diff-warn' if ratio > 3 else 'diff-ok')
        parts.append(
            f"<tr><td>{html.escape(label)}</td><td>{sd:.2f}</td>"
            f"<td class='{cls}'>{mc_m:.2f}</td>"
            f"<td>{base_mc:.2f}</td>"
            f"<td class='{cls}'>{mc_p:.2f}</td>"
            f"<td>{ratio:.1f}×</td></tr>"
        )
    parts.append("</table>")

    # ---- JS payload ----
    n_dense = sorted(set(
        [0] + list(range(1, 21)) + list(range(22, 100, 3))
        + list(range(100, 501, 10))
    ))
    n_grid = [0, 1, 5, 25, 50, 150, 500]
    js = JS_RENDER
    js = js.replace('__RESPAWN_MS__', str(config.RESPAWN_MS))
    js = js.replace('__PRIOR_MU__', json.dumps([s[2] for s in LATENT_SPECS]))
    js = js.replace('__PRIOR_SD__', json.dumps([s[3] for s in LATENT_SPECS]))
    js = js.replace('__LATENT_KIND__', json.dumps([s[4] for s in LATENT_SPECS]))
    js = js.replace('__LATENT_LABELS__', json.dumps([s[0] for s in LATENT_SPECS]))
    js = js.replace('__NAT_LABELS__', json.dumps([s[5] for s in LATENT_SPECS]))
    js = js.replace('__N_DENSE__', json.dumps(n_dense))
    js = js.replace('__N_GRID__', json.dumps(n_grid))
    js = js.replace('__OBS_NS__', json.dumps(OBS_NS))
    js = js.replace('__PER_N_SPECS_JS__', json.dumps(
        [{'label': l, 'eq': e, 'key': k, 'parents': p}
         for (l, e, k, p) in PER_N_SPECS]))
    scen_js = {}
    for name, theta in scenarios_data.items():
        scen_js[name] = {
            'theta': theta,
            'color': SCENARIO_COLORS.get(name, CURRENT_COLOR),
            'editable': False,
        }
    js = js.replace('__SCENARIOS_JS__', json.dumps(scen_js))
    js = js.replace('__CURRENT_COLOR__', CURRENT_COLOR)
    parts.append(f"<script>{js}</script>")

    parts.append("</body></html>")
    return "\n".join(parts)


PRIOR_DEFAULT_FALLBACK = [s[2] for s in LATENT_SPECS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='pgm_inspect.html')
    ap.add_argument('--fit-n', type=int, default=None)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    scenarios = {'truth': truth_theta()}
    fit_info = None
    if args.fit_n is not None:
        print(f"Running MAP fit at N={args.fit_n} seed={args.seed}...")
        theta_map, fit_info = maybe_fit(args.fit_n, args.seed)
        scenarios['MAP'] = theta_map
        print(f"  {fit_info['nit']} iters, {fit_info['dt']:.1f}s")

    html_doc = build_html(scenarios, fit_info=fit_info)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html_doc)
    print(f"Wrote inspector to {args.out}")


if __name__ == '__main__':
    main()
