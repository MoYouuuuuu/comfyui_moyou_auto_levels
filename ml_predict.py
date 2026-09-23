"""ML 波形推断模块：从图像直方图波形预测 PS「增强亮度和对比度」自动色阶参数 (black, gamma, white)。

特征提取与训练端 ml/features.py 严格一致；模型为 GBDT huber + 高 gamma 加权 + 3 种子集成，
在 300 组 PS 面板真值上训练。模型缺失或加载失败时返回 None，由调用方回退规则算法。
"""
import os
import logging

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "auto_levels_model.pkl")


def extract_features(a):
    """a: H,W,3 float32 in [0,255] -> np.array(178, float32)"""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4:
        a = a[0]
    luma = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    h, w = luma.shape
    npx = h * w

    feats = []

    h32, _ = np.histogram(luma, bins=32, range=(0, 255))
    feats.extend(h32 / npx)

    pcts = np.percentile(luma, [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 25,
                                50, 75, 90, 95, 98, 99, 99.5, 99.8, 99.9, 99.95, 99.99])
    feats.extend(pcts / 255.0)

    feats.extend([luma.mean() / 255.0, np.median(luma) / 255.0,
                  luma.std() / 255.0,
                  float(((luma - luma.mean()) ** 3).mean()) / max(luma.std() ** 3, 1e-6),
                  float(((luma - luma.mean()) ** 4).mean()) / max(luma.std() ** 4, 1e-6)])

    for c in range(3):
        ch = a[..., c]
        feats.extend(np.percentile(ch, [0.1, 1, 5, 50, 95, 99, 99.9]) / 255.0)

    feats.extend([np.percentile(luma, 0.1) / 255.0, np.percentile(luma, 99.9) / 255.0,
                  np.percentile(luma, 0.01) / 255.0, np.percentile(luma, 99.99) / 255.0])

    h256, _ = np.histogram(luma, bins=256, range=(0, 255))
    h256 = h256 / npx
    k = np.ones(5, dtype=np.float32) / 5.0
    hs = np.convolve(h256, k, mode='same')
    pk = int(np.argmax(hs))
    feats.extend([pk / 255.0, float(hs[pk])])
    lo = max(0, pk - 8); hi = min(255, pk + 8)
    mask = np.ones(256, dtype=bool); mask[lo:hi + 1] = False
    if mask.any():
        pk2 = int(np.argmax(hs[mask]))
        feats.extend([(pk2 if pk2 < pk else pk2 + (pk2 - lo)) / 255.0, float(hs[mask][pk2])])
    else:
        feats.extend([0.5, 0.0])
    lmax = float(hs[:128].max()); rmax = float(hs[128:].max())
    feats.extend([lmax, rmax, float(hs[:64].max()), float(hs[64:192].max()), float(hs[192:].max())])
    feats.extend([float(hs[:4].sum()), float(hs[-4:].sum())])
    xs = np.arange(256, dtype=np.float32)
    cent = float((xs * hs).sum() / max(hs.sum(), 1e-6))
    feats.extend([cent / 255.0, float((hs ** 2).sum())])

    hlog = np.log1p(h256 * npx)
    hlog = hlog / max(hlog.max(), 1e-6)
    feats.extend(hlog[::4][:64])

    d = np.diff(hs)
    d = d / max(np.abs(d).max(), 1e-6)
    feats.extend([float(d[:64].sum()), float(d[64:128].sum()),
                  float(d[128:192].sum()), float(d[192:].sum())])
    nz = np.where(hs > 1e-4)[0]
    if len(nz):
        feats.extend([nz[0] / 255.0, nz[-1] / 255.0])
    else:
        feats.extend([0.0, 1.0])
    half = hs[pk] / 2.0
    L = pk; R = pk
    while L > 0 and hs[L] > half: L -= 1
    while R < 255 and hs[R] > half: R += 1
    feats.extend([(R - L) / 255.0])

    for i in range(8):
        seg = hs[i * 32:(i + 1) * 32]
        feats.append(float(seg.sum()))
    cum = np.cumsum(hs) / max(hs.sum(), 1e-6)
    for tgt in [0.2, 0.5, 0.8]:
        j = int(np.searchsorted(cum, tgt))
        feats.append(min(j, 255) / 255.0)

    return np.array(feats, dtype=np.float32)


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import joblib
        _MODEL = joblib.load(_MODEL_PATH)
        logger.info("[moyou] ML 模型加载成功: %s", _MODEL_PATH)
    except Exception as e:
        logger.warning("[moyou] ML 模型加载失败(%s)，回退规则算法: %s", type(e).__name__, e)
        _MODEL = False
    return _MODEL


def predict_levels(arr_uint8):
    """arr_uint8: H,W,3 uint8 RGB -> (black, gamma, white) float 或 None"""
    model = _load_model()
    if not model:
        return None
    try:
        x = extract_features(arr_uint8.astype(np.float32))
        out = []
        for name in ("black", "gamma", "white"):
            preds = np.zeros(len(model[name]))
            for k, m in enumerate(model[name]):
                preds[k] = m.predict(x.reshape(1, -1))[0]
            out.append(preds.mean())
        b, g, w = out
        b = float(np.clip(np.round(b), 0, 255))
        g = float(np.clip(np.round(g, 2), 0.1, 10.0))
        w = float(np.clip(np.round(w), 0, 255))
        return (b, g, w)
    except Exception as e:
        logger.warning("[moyou] ML 预测失败(%s)，回退规则算法: %s", type(e).__name__, e)
        return None
