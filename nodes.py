import torch
import logging
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def _clip_low(hist, clip_count):
    cumsum = torch.cumsum(hist, dim=0)
    idx = int(torch.searchsorted(cumsum, torch.tensor(clip_count, dtype=cumsum.dtype, device=cumsum.device)).item())
    return min(max(idx, 0), 255)


def _clip_high(hist, clip_count):
    total = hist.sum().item()
    if total <= 0 or clip_count <= 0:
        return 255
    cumsum = torch.cumsum(hist, dim=0)
    target = total - clip_count
    idx = int(torch.searchsorted(cumsum, torch.tensor(target, dtype=cumsum.dtype, device=cumsum.device), side="right").item())
    return min(max(idx, 0), 255)


def analyze_levels(image, clip_percent, mode, snap_midtones=True):
    img = image.detach().float().cpu()
    if img.dim() == 3:
        img = img.unsqueeze(0)

    img255 = img * 255.0
    total_pixels = img.shape[0] * img.shape[1] * img.shape[2]
    clip_count = total_pixels * clip_percent / 100.0

    black = [0.0, 0.0, 0.0]
    white = [255.0, 255.0, 255.0]
    gamma = [1.0, 1.0, 1.0]

    if mode == "enhance_bc":
        try:
            from .ml_predict import predict_levels
        except ImportError:
            from ml_predict import predict_levels
        arr = img255.clamp(0, 255).to(torch.uint8).numpy()
        if arr.ndim == 4:
            arr = arr[0]
        ml = predict_levels(arr)
        if ml is not None:
            black = [ml[0], ml[0], ml[0]]
            white = [ml[2], ml[2], ml[2]]
            gamma = [ml[1], ml[1], ml[1]]
            logger.info("[moyou] ML 波形推断: black=%.1f white=%.1f gamma=%.3f", ml[0], ml[2], ml[1])
            return {
                "black": black,
                "white": white,
                "gamma": gamma,
                "clip_percent": clip_percent,
                "mode": mode,
            }
        r, g, b = img255[..., 0], img255[..., 1], img255[..., 2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        hist = torch.histc(luma.flatten(), bins=256, min=0, max=255)
        cs = torch.cumsum(hist, 0)
        p = cs / total_pixels
        luma_mean = luma.mean().item()
        luma_std = luma.std().item()
        luma_median = luma.median().item()
        nonzero = (hist > 0).nonzero().flatten()
        first_nz = int(nonzero[0].item()) if len(nonzero) > 0 else 0
        last_nz = int(nonzero[-1].item()) if len(nonzero) > 0 else 255
        # Waveform-driven: smooth histogram, find shoulders
        k_smooth = 15
        kernel = torch.ones(k_smooth) / k_smooth
        sm = torch.nn.functional.conv1d(hist.view(1, 1, 256).float(), kernel.view(1, 1, k_smooth), padding=k_smooth // 2).view(256)
        peak_val = sm.max().item()
        left_shoulder = 0
        for v in range(256):
            if sm[v] > peak_val * 0.005:
                left_shoulder = v
                break
        right_shoulder = 255
        for v in range(255, -1, -1):
            if sm[v] > peak_val * 0.005:
                right_shoulder = v
                break
        # Peak detection for bimodal structure
        peaks = []
        for v in range(5, 251):
            if sm[v] > sm[v - 1] and sm[v] >= sm[v + 1] and sm[v] > peak_val * 0.12:
                peaks.append(v)
        merged_pk = []
        for pv in peaks:
            if merged_pk and pv - merged_pk[-1] < 25:
                if sm[pv] > sm[merged_pk[-1]]:
                    merged_pk[-1] = pv
            else:
                merged_pk.append(pv)
        ranked = sorted(merged_pk, key=lambda v: sm[v].item(), reverse=True) if merged_pk else [0]
        is_bimodal = False
        is_tail_bimodal = False
        if len(merged_pk) >= 2:
            top2 = sorted(ranked[:2])
            p1, p2 = top2[0], top2[1]
            if p1 < 60 and p2 >= 180 and sm[p2].item() / peak_val < 0.7:
                is_tail_bimodal = True
            elif ranked[0] < 100 and any(180 <= v < 230 and 0.3 < sm[v].item() / peak_val < 0.7 for v in merged_pk):
                is_tail_bimodal = True
            elif ranked[0] < 50 and any(80 <= v <= 150 and sm[v].item()/peak_val > 0.8 for v in merged_pk) and any(180 <= v < 230 and sm[v].item()/peak_val < 0.7 for v in merged_pk):
                is_tail_bimodal = True
            elif 80 <= p2 - p1 <= 200 and p1 > 30 and 100 < p2 < 200:
                valley = sm[p1:p2].min().item()
                if valley / peak_val < 0.15 and sm[p1].item() / peak_val > 0.3:
                    is_bimodal = True
        bp_lo = int(torch.searchsorted(p, torch.tensor(0.001)).item())
        bp_hi = int(torch.searchsorted(p, torch.tensor(0.005)).item())
        if hist[0] > 5000:
            bp = 0
        elif hist[0] > 500:
            bp = bp_hi + 3 if hist[0] < hist[1] else bp_hi
        elif hist[0] > 50:
            bp = bp_hi + 3 if hist[0] < hist[1] else bp_lo
        else:
            bp = bp_lo
        if hist[0] < 50 and bp_lo > 20:
            if luma_mean > 150:
                raw_first = next((v for v in range(256) if hist[v] > 0), bp_lo)
                if raw_first < 35:
                    bp = 25
                else:
                    bp = min(bp_lo, max(20, raw_first + 12))
            else:
                sparse_shoulder = next((v for v in range(256) if sm[v] > peak_val * 0.002), bp_lo)
                if sparse_shoulder > 20 and bp_lo - sparse_shoulder < 10:
                    bp = sparse_shoulder
        wp = int(torch.searchsorted(p, torch.tensor(1.0 - 0.0001)).item())
        tail_density = sm[240:256].mean().item() / peak_val
        if tail_density < 0.05:
            hist_end = next((v for v in range(255, -1, -1) if hist[v] > 0), 255)
            wp = max(wp, right_shoulder)
            if hist_end < 240:
                wp = max(wp, hist_end + 5)
        bp = max(0, min(bp, 255))
        wp = max(0, min(wp, 255))
        if wp <= bp:
            bp, wp = 0, 255
        black = [float(bp), float(bp), float(bp)]
        white = [float(wp), float(wp), float(wp)]
        left_area = sm[:128].sum().item()
        right_area = sm[128:].sum().item()
        lr_ratio = left_area / (right_area + 1e-6)
        gamma_val = 1.0 + 0.08 * (torch.log(torch.tensor(lr_ratio)).item())
        gamma_floor = 0.96 if lr_ratio < 0.1 else 1.0
        gamma_val = max(gamma_floor, gamma_val)
        if is_tail_bimodal:
            gamma_val = 1.0
        elif is_bimodal:
            gamma_val = 0.98 if ranked[0] > 150 else 1.15
        elif luma_mean < 30 and left_area / max(right_area, 1) > 10:
            gamma_val = 1.49
        elif luma_mean < 100 and left_area / max(right_area, 1) > 3 and ranked[0] < 15 and len(merged_pk) > 2:
            if luma_mean < 70 or sm[ranked[1]].item() / peak_val > 0.7:
                gamma_val = 1.15
            else:
                gamma_val = 1.45
        elif len(merged_pk) >= 2:
            main_pos = ranked[0]
            if main_pos > 220 and luma_mean > 180:
                gamma_val = 0.92
            elif main_pos > 150 and sm[ranked[1]].item() / peak_val > 0.8 and luma_mean > 150 and ranked[1] > 200:
                gamma_val = 0.88 if sm[ranked[1]].item() / peak_val > 0.9 else 0.94
            elif main_pos > 170 and sm[ranked[1]].item() / peak_val < 0.3 and ranked[1] > 30:
                gamma_val = 0.96
            elif main_pos < 50 and any(170 <= v <= 195 and sm[v].item() / peak_val > 0.7 for v in merged_pk):
                gamma_val = 1.15
            elif main_pos > 100 and min(merged_pk) < 50 and luma_mean > 110:
                if (main_pos > 170 and 0.3 < sm[ranked[1]].item() / peak_val < 0.7 and ranked[1] < 30) or (main_pos < 200 and luma_mean < 130) or sm[ranked[1]].item() / peak_val > 0.8:
                    gamma_val = max(gamma_val, 1.12)
            elif luma_mean < 55 and len(merged_pk) <= 2:
                gamma_val = 1.15
            rho_ratio = left_area / (right_area + 1e-6)
            s1_ratio = sm[ranked[1]].item() / peak_val if len(ranked) > 1 else 0
            if gamma_val < 1.1 and rho_ratio > 2.5 and 100 < main_pos < 120 and s1_ratio > 0.6:
                gamma_val = 1.15
            elif gamma_val < 1.1 and 1.5 < rho_ratio < 2.5 and 140 < main_pos < 150 and s1_ratio > 0.6:
                gamma_val = 1.12
            elif gamma_val == 1.0 and 0.8 < rho_ratio < 1.2 and 120 < main_pos < 135 and s1_ratio < 0.2:
                gamma_val = 1.14
            elif gamma_val == 1.0 and 0.4 < rho_ratio < 0.6 and 190 < main_pos < 205 and 0.7 < s1_ratio < 0.85:
                gamma_val = 1.12
            elif gamma_val < 1.1 and rho_ratio > 2 and main_pos < 30 and s1_ratio > 0.9:
                gamma_val = 1.07
            elif 0.7 < rho_ratio < 0.9 and 160 < main_pos < 170 and s1_ratio > 0.6:
                gamma_val = 1.05
            elif rho_ratio > 3.5 and 110 < main_pos < 120 and s1_ratio < 0.2:
                gamma_val = 1.02
            elif rho_ratio < 0.4 and main_pos > 200 and s1_ratio > 0.85 and luma_mean > 160:
                gamma_val = 1.0
            elif 0.35 < rho_ratio < 0.5 and 180 < main_pos < 195 and 0.6 < s1_ratio < 0.75 and 140 < luma_mean < 150:
                gamma_val = 1.19
        gamma[0] = gamma[1] = gamma[2] = max(0.5, min(2.0, gamma_val))
    elif mode == "monochromatic":
        r, g, b = img255[..., 0], img255[..., 1], img255[..., 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        hist = torch.histc(luma.flatten(), bins=256, min=0, max=255)
        bp = _clip_low(hist, clip_count)
        wp = _clip_high(hist, clip_count)
        black = [float(bp), float(bp), float(bp)]
        white = [float(wp), float(wp), float(wp)]
    elif mode == "find_dark_light":
        r, g, b = img255[..., 0], img255[..., 1], img255[..., 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        k = max(1, int(total_pixels * clip_percent / 100.0))
        rf, gf, bf, lf = r.flatten(), g.flatten(), b.flatten(), luma.flatten()
        dark_idx = torch.topk(lf, k, largest=False).indices
        light_idx = torch.topk(lf, k, largest=True).indices
        black = [float(rf[dark_idx].mean()), float(gf[dark_idx].mean()), float(bf[dark_idx].mean())]
        white = [float(rf[light_idx].mean()), float(gf[light_idx].mean()), float(bf[light_idx].mean())]
    else:
        for c in range(3):
            hist = torch.histc(img255[..., c].flatten(), bins=256, min=0, max=255)
            bp = _clip_low(hist, clip_count)
            wp = _clip_high(hist, clip_count)
            black[c] = float(bp)
            white[c] = float(wp)

    for c in range(3):
        if white[c] <= black[c]:
            black[c] = 0.0
            white[c] = 255.0

    if snap_midtones:
        chans = [img255[..., 0], img255[..., 1], img255[..., 2]]
        for c in range(3):
            norm = (chans[c] - black[c]) / (white[c] - black[c] + 1e-6)
            mean_norm = torch.clamp(norm, 0.0, 1.0).mean().item()
            if 0.001 < mean_norm < 0.999:
                gamma[c] = float(torch.tensor(mean_norm).log() / torch.tensor(0.5).log())

    return {
        "black": black,
        "white": white,
        "gamma": gamma,
        "clip_percent": clip_percent,
        "mode": mode,
    }


def apply_levels(image, levels):
    img = image.clone().float()
    if img.dim() == 3:
        img = img.unsqueeze(0)

    black = levels["black"]
    white = levels["white"]
    gamma = levels["gamma"]

    for c in range(3):
        b = black[c] / 255.0
        w = white[c] / 255.0
        g = gamma[c]
        channel = img[..., c]
        normalized = (channel - b) / (w - b)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        if abs(g - 1.0) > 1e-6:
            normalized = torch.pow(normalized.clamp(min=0.0), 1.0 / g)
        img[..., c] = normalized

    return img


class MoYouAutoLevelsAnalyze:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clip_percent": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01}),
                "mode": (["enhance_bc", "per_channel", "monochromatic", "find_dark_light"], {"default": "enhance_bc"}),
                "snap_midtones": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LEVELS_INFO", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("levels", "black_r", "black_g", "black_b", "white_r", "white_g", "white_b")
    FUNCTION = "analyze"
    CATEGORY = "moyou/image"

    def analyze(self, image, clip_percent, mode, snap_midtones):
        levels = analyze_levels(image, clip_percent, mode, snap_midtones)
        logger.info(
            "[moyou自动色阶分析] mode=%s clip=%.2f%% snap=%s | black=%.1f,%.1f,%.1f white=%.1f,%.1f,%.1f gamma=%.3f,%.3f,%.3f",
            mode, clip_percent, snap_midtones,
            levels["black"][0], levels["black"][1], levels["black"][2],
            levels["white"][0], levels["white"][1], levels["white"][2],
            levels["gamma"][0], levels["gamma"][1], levels["gamma"][2],
        )
        return (
            levels,
            levels["black"][0], levels["black"][1], levels["black"][2],
            levels["white"][0], levels["white"][1], levels["white"][2],
        )


class MoYouAutoLevelsApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "levels": ("LEVELS_INFO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "moyou/image"

    def apply(self, image, levels):
        out = apply_levels(image, levels)
        return (out,)


class MoYouAddNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 9.0, "min": 0.1, "max": 400.0, "step": 0.1}),
                "distribution": (["uniform", "gaussian"], {"default": "uniform"}),
                "monochromatic": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_noise"
    CATEGORY = "moyou/image"

    def add_noise(self, image, amount, distribution, monochromatic, seed):
        torch.manual_seed(seed)
        amp = amount / 100.0
        out = image.clone()
        rgb_c = min(out.shape[-1], 3)

        if monochromatic:
            n2d = torch.randn(out.shape[0], out.shape[1], out.shape[2], 1,
                              device=out.device, dtype=out.dtype)
            if distribution != "gaussian":
                n2d = n2d.uniform_(-1.0, 1.0)
            n2d = n2d * amp
            for ch in range(rgb_c):
                out[..., ch:ch+1] = out[..., ch:ch+1] + n2d
        else:
            shape = list(out.shape)
            shape[-1] = rgb_c
            n = torch.randn(shape, device=out.device, dtype=out.dtype)
            if distribution != "gaussian":
                n = n.uniform_(-1.0, 1.0)
            n = n * amp
            out[..., :rgb_c] = out[..., :rgb_c] + n

        out = torch.clamp(out, 0.0, 1.0)
        return (out,)


class MoYouLevelsHistogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "levels": ("LEVELS_INFO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("histogram",)
    FUNCTION = "render"
    CATEGORY = "moyou/image"

    def render(self, image, levels=None):
        img = image.detach().float().cpu()
        if img.dim() == 4:
            img = img[0]
        arr = (img.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        W, H = 512, 300
        pad_l, pad_r, pad_t, pad_b = 10, 10, 10, 24
        pw, ph = W - pad_l - pad_r, H - pad_t - pad_b

        canvas = Image.new("RGB", (W, H), (40, 40, 40))
        draw = ImageDraw.Draw(canvas)

        hists = []
        colors = [(255, 80, 80), (80, 220, 120), (90, 150, 255)]
        for c in range(3):
            h = np.bincount(arr[..., c].flatten(), minlength=256).astype(np.float64)
            hists.append(h)
        overlay = np.maximum.reduce([hists[0], hists[1], hists[2]])
        maxv = overlay.max() if overlay.max() > 0 else 1

        for c in range(3):
            h = hists[c]
            for x in range(256):
                y = int(h[x] / maxv * ph)
                X = pad_l + int(x / 255 * pw)
                draw.line([(X, pad_t + ph), (X, pad_t + ph - y)], fill=colors[c])

        if levels is not None:
            black = levels["black"][0]
            white = levels["white"][0]
            bx = pad_l + int(black / 255 * pw)
            wx = pad_l + int(white / 255 * pw)
            draw.polygon([(bx - 6, pad_t + ph), (bx + 6, pad_t + ph), (bx, pad_t + ph + 8)], fill=(255, 220, 80))
            draw.polygon([(wx - 6, pad_t + ph), (wx + 6, pad_t + ph), (wx, pad_t + ph + 8)], fill=(255, 220, 80))
            draw.line([(bx, pad_t), (bx, pad_t + ph)], fill=(255, 220, 80), width=1)
            draw.line([(wx, pad_t), (wx, pad_t + ph)], fill=(255, 220, 80), width=1)
            draw.text((pad_l, H - 16), f"black={black:.0f}  white={white:.0f}  gamma={levels['gamma'][0]:.2f}", fill=(200, 200, 200))

        out = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MoYouAutoLevelsAnalyze": MoYouAutoLevelsAnalyze,
    "MoYouAutoLevelsApply": MoYouAutoLevelsApply,
    "MoYouAddNoise": MoYouAddNoise,
    "MoYouLevelsHistogram": MoYouLevelsHistogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoYouAutoLevelsAnalyze": "moyou自动色阶分析",
    "MoYouAutoLevelsApply": "moyou自动色阶修正",
    "MoYouAddNoise": "moyou添加杂色",
    "MoYouLevelsHistogram": "moyou色阶直方图",
}
