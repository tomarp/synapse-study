"""Bayesian models of ANALYSIS_PLAN.md (section 4), posterior summaries and convergence diagnostics.

Models are written directly in PyMC so that every term is explicit. Random effects are non-centred. Posterior draws of
the quantities needed for reporting are cached per model (``<out>/cache``), keyed by the data and the specification.
"""
from __future__ import annotations

import hashlib
import json

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .common import CACHE, SEED, SUBS, TASKS

SAMPLER = dict(draws=1500, tune=2000, chains=4, target_accept=0.95, nuts_sampler="nutpie", random_seed=SEED,
               progressbar=False)
ROPE = {"log_rt": 0.02, "logit_acc": 0.2, "ordinal": 0.18, "hr": 1.0, "ln_rmssd": 0.05, "temp": 0.1}


# ------------------------------------------------------------------------------------------------ utilities
def _key(name: str, df: pd.DataFrame, spec: dict) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    h.update(json.dumps(spec, sort_keys=True, default=str).encode())
    return f"{name}-{h.hexdigest()[:12]}"


def _post(idata, var: str) -> np.ndarray:
    """Posterior draws with chains stacked: shape (n_draws, ...)."""
    a = idata["posterior"][var].values
    return a.reshape(-1, *a.shape[2:])


def _diagnostics(idata, names: list[str]) -> dict:
    s = az.summary(idata, var_names=names)
    div = int(np.asarray(idata["sample_stats"]["diverging"].values).sum())
    return dict(rhat_max=float(pd.to_numeric(s["r_hat"]).max()), ess_bulk_min=float(pd.to_numeric(s["ess_bulk"]).min()),
                divergences=div)


def _ok(diag: dict) -> bool:
    return diag["rhat_max"] <= 1.01 and diag["ess_bulk_min"] >= 400 and diag["divergences"] == 0


def _fit(name: str, df: pd.DataFrame, spec: dict, build, keep: list[str], check: list[str], ppc=None, **sampler) -> dict:
    """Fit with the standard sampler; a model that misses the convergence criteria (ANALYSIS_PLAN.md, section 5) is
    refitted once with twice the draws and tuning steps."""
    out = _fit_once(name, df, spec, build, keep, check, ppc, **sampler)
    if not _ok(out["_diag"]):
        long = {**SAMPLER, **sampler, "draws": 2 * SAMPLER["draws"], "tune": 2 * SAMPLER["tune"]}
        out = _fit_once(name, df, spec, build, keep, check, ppc, **long)
        out["_diag"] = {**out["_diag"], "refit_longer": True}
    return out


def _fit_once(name: str, df: pd.DataFrame, spec: dict, build, keep: list[str], check: list[str], ppc=None, **sampler) -> dict:
    """Sample (or load from the cache) and return the kept draws plus diagnostics. ``ppc(yrep)`` returns summary
    statistics of posterior-predictive replicates (200 thinned draws), stored with the prefix ``ppc_``."""
    CACHE.mkdir(parents=True, exist_ok=True)
    sampler = {**SAMPLER, **sampler}
    f = CACHE / f"{_key(name, df, {**spec, 'sampler': sampler, 'ppc': ppc is not None})}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        out = {k: z[k] for k in z.files if k != "_diag"}
        out["_diag"] = json.loads(str(z["_diag"]))
        return out
    model = build()
    with model:
        idata = pm.sample(**sampler)
        out = {k: _post(idata, k) for k in keep}
        if ppc is not None:
            thin = idata["posterior"].to_dataset().isel(draw=slice(None, None, 30))
            pp = pm.sample_posterior_predictive(thin, var_names=["y"], random_seed=SEED, progressbar=False)
            yrep = pp["posterior_predictive"]["y"].values
            out.update({f"ppc_{k}": v for k, v in ppc(yrep.reshape(-1, yrep.shape[-1])).items()})
    diag = _diagnostics(idata, check)
    np.savez_compressed(f, **out, _diag=json.dumps(diag))
    out["_diag"] = diag
    return out


def summarise(draws: np.ndarray, rope: float) -> dict:
    """Median, 95 % equal-tailed CrI, probability of direction, probability outside / inside the ROPE."""
    d = np.asarray(draws, dtype=float)
    lo, med, hi = np.percentile(d, [2.5, 50, 97.5])
    pdir = max((d > 0).mean(), (d < 0).mean())
    inside = (np.abs(d) < rope).mean()
    if lo > 0 or hi < 0:
        verdict = "credible"
    elif inside >= 0.95:
        verdict = "negligible"
    elif pdir >= 0.90:
        verdict = "probable"
    else:
        verdict = "inconclusive"
    return dict(median=med, lo=lo, hi=hi, pd=pdir, p_outside_rope=1 - inside, verdict=verdict)


def _idx(values) -> tuple[np.ndarray, list]:
    levels = sorted(pd.unique(values))
    return np.array([levels.index(v) for v in values]), levels


# ------------------------------------------------------------------------------------------- performance (A)
def performance(df: pd.DataFrame, outcome: str, prior_scale: float = 1.0, extra: str | None = None, name: str = "A") -> dict:
    """A1 (outcome 'log_rt', Normal) or A2 (outcome 'correct', Bernoulli-logit), pooled over the six task variants.

    log RT / logit P(correct) = cell + participant + participant x task + (bL + gL[task] + uL[p]) L
                                + (bS + gS[task] + uS[p]) S + bO O + bLS L S + bpos[task] log_pos [+ extra]
    """
    df = df.reset_index(drop=True)
    cell, cells = _idx(df.cell)
    task = np.array([TASKS.index(t) for t in df.task])
    pid = np.array([SUBS.index(p) for p in df.participant_id])
    pt = pid * len(TASKS) + task
    slot, slots = _idx(df.slot)
    gauss = outcome == "log_rt"
    y = df[outcome].astype(float).values
    ybar = y.mean() if gauss else 0.0
    k = prior_scale
    sd = dict(beta=0.1 if gauss else 0.5, tau=0.05 if gauss else 0.25, slope=0.1 if gauss else 0.5,
              p=0.3 if gauss else 1.0, pt=0.2 if gauss else 0.5, pos=0.2 if gauss else 0.5)
    spec = dict(outcome=outcome, k=k, extra=extra, cells=cells, sd=sd)

    def build():
        m = pm.Model()
        with m:
            a = pm.Normal("a", 0.0 if gauss else 3.0, 0.5 if gauss else 2.0, shape=len(cells))
            bL = pm.Normal("bL", 0, sd["beta"] * k)
            bS = pm.Normal("bS", 0, sd["beta"] * k)
            bO = pm.Normal("bO", 0, sd["beta"] * k)
            bLS = pm.Normal("bLS", 0, sd["beta"] * k)
            tauL, tauS = pm.HalfNormal("tauL", sd["tau"]), pm.HalfNormal("tauS", sd["tau"])
            gL = pm.Deterministic("gL", tauL * pm.Normal("zgL", 0, 1, shape=len(TASKS)))
            gS = pm.Deterministic("gS", tauS * pm.Normal("zgS", 0, 1, shape=len(TASKS)))
            sL, sS = pm.HalfNormal("sdL", sd["slope"]), pm.HalfNormal("sdS", sd["slope"])
            uL = pm.Deterministic("uL", sL * pm.Normal("zuL", 0, 1, shape=len(SUBS)))
            uS = pm.Deterministic("uS", sS * pm.Normal("zuS", 0, 1, shape=len(SUBS)))
            sp, spt = pm.HalfNormal("sd_p", sd["p"]), pm.HalfNormal("sd_pt", sd["pt"])
            up = sp * pm.Normal("zp", 0, 1, shape=len(SUBS))
            upt = spt * pm.Normal("zpt", 0, 1, shape=len(SUBS) * len(TASKS))
            bpos = pm.Normal("bpos", 0, sd["pos"], shape=len(TASKS))
            L, S = df.L.values, df.S.values
            mu = (a[cell] + up[pid] + upt[pt] + (bL + gL[task] + uL[pid]) * L + (bS + gS[task] + uS[pid]) * S
                  + bO * df.O.values + bLS * L * S + bpos[task] * df.log_pos.values)
            if extra == "lux_z":
                mu = mu + pm.Normal("bX", 0, sd["beta"]) * df.lux_z.values
            elif extra == "slot":
                ss = pm.HalfNormal("sd_slot", sd["pt"])
                mu = mu + ss * pm.Normal("zslot", 0, 1, shape=len(slots))[slot]
            if gauss:
                sig = pm.HalfNormal("sigma", 0.5, shape=len(TASKS))
                pm.Normal("y", mu, sig[task], observed=y - ybar)
            else:
                pm.Bernoulli("y", logit_p=mu, observed=y)
        return m

    def ppc(yrep: np.ndarray) -> dict:
        obs = y - ybar
        out = {}
        for stat, fn in (("mean", np.mean), ("sd", np.std)) if gauss else (("mean", np.mean),):
            out[stat] = np.stack([fn(yrep[:, task == i], axis=1) for i in range(len(TASKS))], axis=1)
            out[f"obs_{stat}"] = np.array([fn(obs[task == i]) for i in range(len(TASKS))])
        return out

    return _fit(name, df[["participant_id", "task", "cell", outcome, "L", "S", "O", "log_pos", "lux_z", "slot"]],
                spec, build, ["bL", "bS", "bO", "bLS", "gL", "gS", "uL", "uS", "sdL", "sdS", "tauL", "tauS"],
                ["bL", "bS", "bO", "bLS", "tauL", "tauS", "sdL", "sdS", "sd_p", "sd_pt"],
                ppc=ppc if (extra is None and prior_scale == 1.0) else None)


# --------------------------------------------------------------------------------------------- ordinal (B, C)
def ordinal(df: pd.DataFrame, K: int, family: str, prior_scale: float = 1.0, extra: str | None = None,
            name: str = "ord") -> dict:
    """Cumulative-logit model of one item; df has 'y' coded 0..K-1.

    B (T1-T3): eta = bL L + bS S + bsnd snd + bR R + bO O + bLS L S + u_p   (S = R = 0 at T1)
    C (T2-T3): eta = bL L + bS S + bR R + bO O + bLS L S + u_p
    """
    df = df.reset_index(drop=True)
    pid = np.array([SUBS.index(p) for p in df.participant_id])
    slot, slots = _idx(df.slot)
    k = prior_scale
    spec = dict(K=K, family=family, k=k, extra=extra)

    def build():
        m = pm.Model()
        with m:
            b = {n: pm.Normal(n, 0, 1.0 * k) for n in ("bL", "bS", "bR", "bO", "bLS")}
            eta = (b["bL"] * df.L.values + b["bS"] * df.S.values + b["bR"] * df.R.values + b["bO"] * df.O.values
                   + b["bLS"] * df.L.values * df.S.values)
            if family == "B":
                eta = eta + pm.Normal("bsnd", 0, 1.5) * df.snd.values
            sp = pm.HalfNormal("sd_p", 1.0)
            eta = eta + (sp * pm.Normal("zp", 0, 1, shape=len(SUBS)))[pid]
            if extra == "lux_z":
                eta = eta + pm.Normal("bX", 0, 1.0) * df.lux_z.values
            elif extra == "slot":
                eta = eta + (pm.HalfNormal("sd_slot", 1.0) * pm.Normal("zslot", 0, 1, shape=len(slots)))[slot]
            cut = pm.Normal("cut", np.linspace(-2, 2, K - 1), 1.5, shape=K - 1,
                            transform=pm.distributions.transforms.ordered)
            pm.OrderedLogistic("y", eta=eta, cutpoints=cut, observed=df.y.values.astype(int))
            pm.Deterministic("u", sp * m["zp"])
        return m

    def ppc(yrep: np.ndarray) -> dict:
        return {"prop": np.stack([(yrep == c).mean(1) for c in range(K)], axis=1),
                "obs_prop": np.array([(df.y.values == c).mean() for c in range(K)])}

    keep = ["bL", "bS", "bR", "bO", "bLS", "cut", "u"] + (["bsnd"] if family == "B" else [])
    return _fit(name, df[["participant_id", "y", "L", "S", "R", "O", "snd", "lux_z", "slot"]], spec, build, keep,
                ["bL", "bS", "bR", "bO", "bLS", "cut", "sd_p"], ppc=ppc if (extra is None and prior_scale == 1.0) else None)


def ordinal_joint(df: pd.DataFrame, items: list[str], K: int, name: str = "B-joint") -> dict:
    """S8: the eight perception items in one model; cross-modal effects partially pooled (light effects on non-light
    items, soundscape effects on non-acoustic items), manipulation-check effects free."""
    df = df.reset_index(drop=True)
    it = np.array([items.index(i) for i in df["item"]])
    pid = np.array([SUBS.index(p) for p in df.participant_id])
    crossL = np.array([not i.startswith("light") for i in items])
    crossS = np.array([not i.startswith("acoustic") for i in items])

    def build():
        m = pm.Model()
        with m:
            muL, muS = pm.Normal("muL", 0, 1), pm.Normal("muS", 0, 1)
            tL, tS = pm.HalfNormal("tauL", 0.5), pm.HalfNormal("tauS", 0.5)
            zL, zS = pm.Normal("zL", 0, 1, shape=len(items)), pm.Normal("zS", 0, 1, shape=len(items))
            bL = pm.Deterministic("bL", pm.math.where(crossL, muL + tL * zL, zL))
            bS = pm.Deterministic("bS", pm.math.where(crossS, muS + tS * zS, zS))
            bsnd = pm.Normal("bsnd", 0, 1.5, shape=len(items))
            bR = pm.Normal("bR", 0, 1, shape=len(items))
            bO = pm.Normal("bO", 0, 1, shape=len(items))
            sp = pm.HalfNormal("sd_p", 1.0, shape=len(items))
            u = sp[None, :] * pm.Normal("zp", 0, 1, shape=(len(SUBS), len(items)))
            eta = (bL[it] * df.L.values + bS[it] * df.S.values + bsnd[it] * df.snd.values + bR[it] * df.R.values
                   + bO[it] * df.O.values + u[pid, it])
            cut = pm.Normal("cut", np.tile(np.linspace(-2, 2, K - 1), (len(items), 1)), 1.5, shape=(len(items), K - 1),
                            transform=pm.distributions.transforms.ordered)
            pm.OrderedLogistic("y", eta=eta, cutpoints=cut[it], observed=df.y.values.astype(int))
        return m

    return _fit(name, df[["participant_id", "item", "y", "L", "S", "R", "O", "snd"]], dict(items=items, K=K), build,
                ["bL", "bS", "muL", "muS", "tauL", "tauS"], ["bL", "bS", "muL", "muS", "tauL", "tauS"])


def expected_difference(fit: dict, df: pd.DataFrame, factor: str, values: np.ndarray) -> np.ndarray:
    """Average marginal effect on the expected rating (rating units): E[Y | factor = +1/2] - E[Y | factor = -1/2] over
    the observed rows (rows with a soundscape only, for S), participants' posterior effects included."""
    rows = df if factor == "L" else df[df.snd == 1]
    pid = np.array([SUBS.index(p) for p in rows.participant_id])
    base = {n: rows[n].values for n in ("L", "S", "R", "O", "snd")}
    cut, u = fit["cut"], fit["u"]

    def expect(level: float) -> np.ndarray:
        v = dict(base)
        v[factor] = np.full(len(rows), level)
        eta = (fit["bL"][:, None] * v["L"] + fit["bS"][:, None] * v["S"] + fit["bR"][:, None] * v["R"]
               + fit["bO"][:, None] * v["O"] + fit["bLS"][:, None] * v["L"] * v["S"] + u[:, pid])
        if "bsnd" in fit:
            eta = eta + fit["bsnd"][:, None] * v["snd"]
        cum = 1 / (1 + np.exp(-(cut[:, None, :] - eta[:, :, None])))         # P(Y <= k)
        cum = np.concatenate([np.zeros_like(cum[..., :1]), cum, np.ones_like(cum[..., :1])], axis=-1)
        p = np.diff(cum, axis=-1)
        return (p * values[None, None, :]).sum(-1).mean(-1)

    return expect(0.5) - expect(-0.5)


# ------------------------------------------------------------------------------------------ physiology (D)
def gaussian(df: pd.DataFrame, outcome: str, design: str, eff_sd: float, extra: str | None = None, name: str = "D") -> dict:
    """D1/D3 ('phase'): y = a + u_p + w_session + bTask task + bL L + bS S + bR R + bO O + bLS L S
    D2 ('baseline'):    y = a + u_p + bL L + bO O
    D4 ('within'):      y = alpha_session + bTask task + bS S + bR R      (relative values; within-session only)"""
    df = df.dropna(subset=[outcome]).reset_index(drop=True)
    y = df[outcome].values.astype(float)
    my, sy = y.mean(), y.std()
    pid = np.array([SUBS.index(p) for p in df.participant_id])
    ses, sess = _idx(df.participant_id + "/" + df.session_id)
    spec = dict(outcome=outcome, design=design, eff_sd=eff_sd, extra=extra)
    names = {"phase": ["bTask", "bL", "bS", "bR", "bO", "bLS"], "baseline": ["bL", "bO"], "within": ["bTask", "bS", "bR"]}[design]

    def build():
        m = pm.Model()
        with m:
            b = {n: pm.Normal(n, 0, sy if n == "bTask" else eff_sd) for n in names}
            if design == "within":
                mu = pm.Normal("alpha", my, 5 * sy, shape=len(sess))[ses]
            else:
                mu = pm.Normal("a", my, 2 * sy) + (pm.HalfNormal("sd_p", sy) * pm.Normal("zp", 0, 1, shape=len(SUBS)))[pid]
                if design == "phase":
                    mu = mu + (pm.HalfNormal("sd_ses", sy / 2) * pm.Normal("zses", 0, 1, shape=len(sess)))[ses]
            terms = {"bTask": df.task.values, "bL": df.L.values, "bS": df.S.values, "bR": df.R.values, "bO": df.O.values,
                     "bLS": df.L.values * df.S.values}
            for n in names:
                mu = mu + b[n] * terms[n]
            if extra == "lux_z":
                mu = mu + pm.Normal("bX", 0, eff_sd) * df.lux_z.values
            pm.Normal("y", mu, pm.HalfNormal("sigma", sy), observed=y)
        return m

    return _fit(name, df[["participant_id", "session_id", outcome, "task", "L", "S", "R", "O", "lux_z"]], spec, build,
                names, names + ["sigma"], target_accept=0.99)
