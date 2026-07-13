import numpy as np

# channel indices — pred: [N, 6, T]
LOAD_Q10, LOAD_Q50, LOAD_Q90 = 0, 1, 2
PV_Q10, PV_Q50, PV_Q90 = 3, 4, 5

# channel indices — true: [N, 4, T]
TRUE_LOAD, TRUE_PV, TRUE_MASK, TRUE_NET = 0, 1, 2, 3

QUANTILES = [0.1, 0.5, 0.9]


class DisaggregationMetrics:
    """
    All metrics for load/PV disaggregation with quantile predictions.

    pred : [N, 6, T]  — load_q10, load_q50, load_q90, pv_q10, pv_q50, pv_q90
    true : [N, 4, T]  — y_load, y_pv, mask, y_net_demand
    """

    # mirror module-level constants onto the class so cls. access works
    LOAD_Q10, LOAD_Q50, LOAD_Q90 = 0, 1, 2
    PV_Q10, PV_Q50, PV_Q90 = 3, 4, 5
    TRUE_LOAD, TRUE_PV, TRUE_MASK, TRUE_NET = 0, 1, 2, 3

    @staticmethod
    def _prep(true, pred, mask):
        t = np.asarray(true).ravel()
        p = np.asarray(pred).ravel()
        m = np.asarray(mask, dtype=bool).ravel()
        return t[m], p[m]

    @classmethod
    def rmse(cls, t, p, mask):
        t, p = cls._prep(t, p, mask)
        return np.sqrt(np.mean((p - t) ** 2))

    @classmethod
    def nrmse(cls, t, p, mask):
        t, p = cls._prep(t, p, mask)
        return np.sqrt(np.mean((p - t) ** 2)) / np.max(t)

    @classmethod
    def mae(cls, t, p, mask):
        t, p = cls._prep(t, p, mask)
        return np.mean(np.abs(p - t))

    @classmethod
    def mape(cls, t, p, mask):
        t, p = cls._prep(t, p, mask)
        valid = np.abs(t) > 1e-10
        return np.mean(np.abs((p[valid] - t[valid]) / t[valid])) * 100

    @classmethod
    def r2(cls, t, p, mask):
        t, p = cls._prep(t, p, mask)
        return np.corrcoef(p, t)[0, 1] ** 2

    @classmethod
    def mbe(cls, t, p, mask):
        t, p = cls._prep(t, p, mask)
        return np.mean(p - t)

    @classmethod
    def efe(cls, t, p, mask):
        t, p = cls._prep(t, p, mask)
        return np.abs(p.sum() - t.sum()) / t.sum() * 100

    @classmethod
    def pinball(cls, t, p_q, mask, q):
        t, p_q = cls._prep(t, p_q, mask)
        err = t - p_q
        return np.mean(np.where(err >= 0, q * err, (q - 1) * err))

    @classmethod
    def coverage(cls, t, p_low, p_high, mask):
        t = np.asarray(t).ravel()
        lo = np.asarray(p_low).ravel()
        hi = np.asarray(p_high).ravel()
        m = np.asarray(mask, dtype=bool).ravel()
        return np.mean((t[m] >= lo[m]) & (t[m] <= hi[m]))

    @classmethod
    def sharpness(cls, p_low, p_high, mask):
        lo = np.asarray(p_low).ravel()
        hi = np.asarray(p_high).ravel()
        m = np.asarray(mask, dtype=bool).ravel()
        return np.mean(hi[m] - lo[m])

    @classmethod
    def net_demand_rmse(cls, true_net, pred_load_q50, pred_pv_q50, mask):
        pred_net = np.asarray(pred_load_q50) - np.asarray(pred_pv_q50)
        return cls.rmse(true_net, pred_net, mask)

    @classmethod
    def export_violation(cls, true_net, pred_pv_q50, mask):
        """
        A prediction is physically impossible when predicted PV production is
        less than the export implied by the true net demand — you cannot
        export more power than you generate. true_net = consumption -
        generation (net_demand_w), so export = max(-true_net, 0).

        Returns the rate/count/severity of these violations over all masked
        (labeled) timesteps, not just the exporting ones — so the rate is
        directly comparable to coverage/mape denominators elsewhere.
        """
        t_net, p_pv = cls._prep(true_net, pred_pv_q50, mask)
        export_true = np.clip(-t_net, 0, None)
        shortfall = np.clip(export_true - p_pv, 0, None)
        violated = shortfall > 1e-6
        return {
            "rate": float(np.mean(violated)),
            "count": int(violated.sum()),
            "mean_magnitude": float(shortfall[violated].mean()) if violated.any() else 0.0,
        }

    @classmethod
    def all(cls, true, pred) -> dict:
        mask = true[:, cls.TRUE_MASK, :]

        targets = {
            "load": (
                true[:, cls.TRUE_LOAD, :],
                pred[:, cls.LOAD_Q10, :],
                pred[:, cls.LOAD_Q50, :],
                pred[:, cls.LOAD_Q90, :],
            ),
            "pv": (
                true[:, cls.TRUE_PV, :],
                pred[:, cls.PV_Q10, :],
                pred[:, cls.PV_Q50, :],
                pred[:, cls.PV_Q90, :],
            ),
        }

        results = {}
        for name, (t, q10, q50, q90) in targets.items():
            results[name] = {
                "rmse": float(cls.rmse(t, q50, mask)),
                "nrmse": float(cls.nrmse(t, q50, mask)),
                "mae": float(cls.mae(t, q50, mask)),
                "mape": float(cls.mape(t, q50, mask)),
                "r2": float(cls.r2(t, q50, mask)),
                "mbe": float(cls.mbe(t, q50, mask)),
                "efe": float(cls.efe(t, q50, mask)),
                "pinball_q10": float(cls.pinball(t, q10, mask, 0.1)),
                "pinball_q50": float(cls.pinball(t, q50, mask, 0.5)),
                "pinball_q90": float(cls.pinball(t, q90, mask, 0.9)),
                "coverage": float(cls.coverage(t, q10, q90, mask)),
                "sharpness": float(cls.sharpness(q10, q90, mask)),
            }

        results["load"]["net_rmse"] = float(
            cls.net_demand_rmse(
                true[:, cls.TRUE_NET, :],
                pred[:, cls.LOAD_Q50, :],
                pred[:, cls.PV_Q50, :],
                mask,
            )
        )

        export_violation = cls.export_violation(
            true[:, cls.TRUE_NET, :], pred[:, cls.PV_Q50, :], mask
        )
        results["pv"]["export_violation_rate"] = export_violation["rate"]
        results["pv"]["export_violation_count"] = export_violation["count"]
        results["pv"]["export_violation_mag"] = export_violation["mean_magnitude"]

        return results
