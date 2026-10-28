PARAM_VERSION = 1  # 改动参数列表后 +1, 旧参数文件自动失效
"""
main.py —— MaixCAM2 钢球检测 (LED + OpenCV 亮点)

原理: 打开补光 LED → 钢球表面反光形成亮点 → OpenCV 检测亮点 → 计数 + 定位

触摸:
  右上角 LED 按钮 → 开关补光灯
  左半画面        → 切换调参/全屏模式
  右半画面        → 重置计数
  底部参数按钮    → 选中/调节检测参数
"""

from maix import app, camera, display, image, time, touchscreen, uart, pinmap, gpio, err, webrtc
import cv2, numpy as np, struct, os, json, shutil, threading
import time as _time2
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============================================================
# LED 补光灯 (GPIO B25)
# ============================================================

LED_PIN = "B25"
LED_GPIO = "GPIOB25"


class LED:

    def __init__(self):
        self._led = None
        try:
            err.check_raise(pinmap.set_pin_function(LED_PIN, LED_GPIO))
            self._led = gpio.GPIO(LED_GPIO, gpio.Mode.OUT)
            self.off()
        except Exception as e:
            print(f"[led] 初始化失败: {e}")

    def on(self):
        if self._led:
            self._led.value(1)

    def off(self):
        if self._led:
            self._led.value(0)

    def toggle(self):
        if self._led:
            self._led.value(1 - self._led.value())

    def is_on(self):
        if self._led:
            try:
                return self._led.value() == 1
            except Exception:
                return False
        return False


# ============================================================
# 串口上报 (复用 GangQiu 协议)
# ============================================================

FRAME_HEAD = 0xAA
UART_DEV = "/dev/ttyS4"
UART_BAUD = 115200
UART_PINS = {"A21": "UART4_TX", "A22": "UART4_RX"}


class SerialComm:

    _PK = struct.Struct("<hh")   # s16 x2 = 当前x + 目标x
    _SZ = _PK.size               # 4 bytes

    def __init__(self):
        self._ser = None

    def open(self):
        if self._ser:
            self._ser.close()
        try:
            for pin, func in UART_PINS.items():
                err.check_raise(pinmap.set_pin_function(pin, func))
            self._ser = uart.UART(UART_DEV, UART_BAUD)
        except Exception as e:
            print(f"[comm] 初始化失败: {e}")

    def send_x(self, cur_x, target_x):
        """帧格式: AA | cur_x(s16) | target_x(s16)   (5字节, 无帧尾)"""
        if not self._ser:
            return
        buf = bytearray(1 + self._SZ)  # 1头 + 4数据 = 5
        buf[0] = FRAME_HEAD
        self._PK.pack_into(buf, 1, cur_x, target_x)
        try:
            self._ser.write(bytes(buf))
        except Exception:
            pass


# ============================================================
# OpenCV 亮点检测
# ============================================================

class BallDetector:

    def __init__(self):
        self.thresh = 100           # 二值化阈值
        self.adapt_mode = 0
        self.adapt_block = 31
        self.adapt_c = 8
        self.blur_ksize = 1         # 模糊核 (1=几乎不消耗, 提速)

        # 面积
        self.min_area = 5
        self.max_area = 100

        # 圆度
        self.min_circ = 0.20

        # 去重
        self.margin = 1

        # 形态学 kernel (缓存, 避免每帧创建)
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 单球追踪
        self._last_cx = None
        self._last_cy = None
        self._max_dist = 80
        self._max_lost = 10
        self._lost_frames = 0

    def detect_with_binary(self, img_maix, roi_y1=0, roi_y2=480):
        # ---- 半分辨率处理 ----
        small = img_maix.resize(320, 240)
        img_cv = image.image2cv(small, False, False)

        # ---- ROI 裁剪 (numpy切片, 只处理横带, 大幅提速) ----
        roi_y1 = max(0, roi_y1)
        roi_y2 = min(480, roi_y2)
        has_roi = (roi_y1 > 0 or roi_y2 < 480)
        if has_roi:
            crop_y1 = roi_y1 // 2
            crop_y2 = roi_y2 // 2
            img_cv = img_cv[crop_y1:crop_y2, :, :]
            roi_h = roi_y2 - roi_y1
        else:
            roi_h = 480
            roi_y1 = 0
            crop_y1 = 0

        h, w = img_cv.shape[:2]

        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        # ---- 中值模糊 (核>1时才跑, 保留亮斑峰值) ----
        bs = self.blur_ksize
        if bs > 1:
            if bs % 2 == 0:
                bs += 1
            blur = cv2.medianBlur(gray, bs)
        else:
            blur = gray

        # ---- 二值化 ----
        binary = self._make_binary(blur, self.thresh)

        # ---- 轮廓检测 (SIMPLE 提速) ----
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        m = self.margin
        mx, my = w - m, h - m
        Sx = 2.0                         # x 缩放 320→640
        Sy = roi_h / h if h > 0 else 2.0 # y 缩放 (考虑ROI裁剪)
        SA = Sx * Sy                     # 面积缩放

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh == 0 or bw == 0:
                continue
            if bx < m or by < m or bx + bw > mx or by + bh > my:
                continue

            # 圆度
            peri = cv2.arcLength(cnt, True)
            circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0
            if circ < self.min_circ:
                continue

            # 质心用矩 (比 bbox 中点更稳)
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx_small = M['m10'] / M['m00']
                cy_small = M['m01'] / M['m00']
            else:
                cx_small = bx + bw / 2.0
                cy_small = by + bh / 2.0
            cx = int(cx_small * Sx)
            cy = int(cy_small * Sy) + roi_y1
            candidates.append({
                'x':  int(bx * Sx),  'y':  int(by * Sy) + roi_y1,
                'w':  int(bw * Sx),  'h':  int(bh * Sy),
                'cx': cx, 'cy': cy,
                'area': int(area * SA), 'circ': circ,
                'score': (circ ** 2) * np.sqrt(max(area, 1)),
            })

        # NMS 去重 → 只取最高分的一个 (最像光点)
        candidates.sort(key=lambda o: o['score'], reverse=True)
        objs = []
        for c in candidates:
            ok = True
            for a in objs:
                if self._is_overlap(c, a):
                    ok = False
                    break
            if ok:
                objs.append(c)
                break   # 只要第一名

        # ---- 单球位置过滤: 只认与上次位置相近的点 ----
        if objs:
            best = objs[0]
            if self._last_cx is not None:
                dx = best['cx'] - self._last_cx
                dy = best['cy'] - self._last_cy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self._max_dist:
                    self._lost_frames += 1
                    if self._lost_frames < self._max_lost:
                        objs = []   # 太远, 拒绝
                    else:
                        self._lost_frames = 0
                        self._last_cx = best['cx']
                        self._last_cy = best['cy']
                else:
                    self._lost_frames = 0
                    self._last_cx = best['cx']
                    self._last_cy = best['cy']
            else:
                self._lost_frames = 0
                self._last_cx = best['cx']
                self._last_cy = best['cy']
        else:
            self._lost_frames += 1

        # ---- 拼回完整 320×240 binary (ROI外全黑, 预览不变形) ----
        if has_roi:
            full = np.zeros((240, 320), dtype=np.uint8)
            full[crop_y1:crop_y2, :] = binary
            binary = full

        return objs, binary

    def _make_binary(self, blur, th=None):
        """对模糊后的灰度图做二值化, th=None时用self.thresh"""
        t = th if th is not None else self.thresh
        if self.adapt_mode == 0:
            _, binary = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
        elif self.adapt_mode == 1:
            ab = self.adapt_block
            if ab % 2 == 0:
                ab += 1
            binary = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, ab, self.adapt_c)
        else:
            ab = self.adapt_block
            if ab % 2 == 0:
                ab += 1
            binary = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, ab, self.adapt_c)

        # ---- 形态学清理: 开运算去噪 + 闭运算合拢碎片 ----
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  self._morph_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel)

        return binary

    def _is_overlap(self, c, a):
        l = max(c['x'], a['x'])
        t = max(c['y'], a['y'])
        r = min(c['x'] + c['w'], a['x'] + a['w'])
        b = min(c['y'] + c['h'], a['y'] + a['h'])
        inter = max(0, r - l) * max(0, b - t)
        if inter <= 0:
            return False
        union = c['area'] + a['area'] - inter
        return (inter / union) >= 0.35

    @staticmethod
    def stable_count(n, history, hist_len=5):
        history.append(n)
        if len(history) > hist_len:
            history.pop(0)
        s = sorted(history)
        return s[len(s) // 2]

    def get_params(self):
        return [
            (self.thresh,          30,  255,  5,   "THR"),
            (self.adapt_mode,       0,    2,  1,   "ADAPT"),
            (self.blur_ksize,       1,   15,  2,   "BLUR"),
            (self.min_area,         1,  200,  1,   "MIN_A"),
            (self.max_area,       100, 8000,  10,  "MAX_A"),
            (int(self.min_circ   * 100),    0, 100, 5,  "CIRC%"),
            (self._max_dist,        20,  400, 10,  "DIST"),
        ]

    def apply_params(self, params):
        self.thresh     = params[0]
        self.adapt_mode = params[1]
        self.blur_ksize = params[2]
        self.min_area   = params[3]
        self.max_area   = params[4]
        self.min_circ   = params[5] / 100.0
        self._max_dist  = params[6]

    def get_values(self):
        return [p[0] for p in self.get_params()]

    def adjust_param(self, idx, delta):
        pl = self.get_params()
        val, vmin, vmax, step, _ = pl[idx]
        val = max(vmin, min(vmax, val + delta * step))
        nv = self.get_values(); nv[idx] = val
        self.apply_params(nv)


# ============================================================
# 配置
# ============================================================

CAM_W, CAM_H = 640, 480
FPS_WIN = 10
CNT_HIST = 5
DET_SKIP = 2
TOUCH_CD = 250
PARAM_PATH = f"/root/ball_cv_v{PARAM_VERSION}.txt"
ROI_PATH   = f"/root/ball_roi_v{PARAM_VERSION}.txt"

C_RED    = image.COLOR_RED
C_GREEN  = image.COLOR_GREEN
C_YELLOW = image.Color.from_rgb(255, 255, 0)
C_WHITE  = image.COLOR_WHITE
C_GRAY   = image.Color.from_rgb(140, 140, 140)
C_DARK   = image.Color.from_rgb(30, 30, 35)
C_BLUE   = image.Color.from_rgb(60, 130, 255)
C_BLACK  = image.Color.from_rgb(0, 0, 0)
C_CYAN   = image.Color.from_rgb(0, 200, 200)


# ============================================================
# 工具
# ============================================================

def is_in_button(x, y, btn):
    return x > btn[0] and x < btn[0] + btn[2] and y > btn[1] and y < btn[1] + btn[3]


def build_buttons(bar_y, bar_h, w):
    led_btn    = [w - 100, 2,  96, 40]
    anchor_btn = [w - 100, 44, 96, 40]
    roi_btn    = [w - 100, 86, 96, 40]
    # 底部 3 个按钮: [ - ] [ 参数名 ] [ + ]
    bw = w // 3
    minus_btn  = [0,           bar_y, bw, bar_h]
    param_btn  = [bw,          bar_y, bw, bar_h]
    plus_btn   = [bw * 2,      bar_y, bw, bar_h]
    return led_btn, anchor_btn, roi_btn, minus_btn, param_btn, plus_btn


def load_params(detector):
    plist = detector.get_params()
    try:
        with open(PARAM_PATH) as f:
            lines = f.read().strip().split("\n")
        # 第一行是版本标记 "#vN", 第二行是数据
        if len(lines) >= 2 and lines[0].startswith("#v"):
            file_ver = int(lines[0][2:])
            sv = [int(x) for x in lines[1].split(",")]
        else:
            # 旧版文件, 无版本头, 兼容加载
            file_ver = 0
            sv = [int(x) for x in lines[0].split(",")]
        if file_ver != PARAM_VERSION:
            print(f"[main] 参数版本不匹配 (文件v{file_ver} != 当前v{PARAM_VERSION}), 使用默认参数")
            return
        if len(sv) == len(plist):
            detector.apply_params(sv)
            print("[main] 加载已保存参数:", sv)
    except Exception:
        print("[main] 使用默认参数")


def save_params(detector):
    try:
        with open(PARAM_PATH, "w") as f:
            f.write(f"#v{PARAM_VERSION}\n")
            f.write(",".join(str(x) for x in detector.get_values()) + "\n")
    except Exception:
        pass


def load_roi():
    try:
        with open(ROI_PATH) as f:
            lines = f.read().strip().split("\n")
        if len(lines) >= 2 and lines[0].startswith("#v"):
            v = [int(x) for x in lines[1].split(",")]
        else:
            v = [int(x) for x in lines[0].split(",")]
        return v
    except Exception:
        return None


def save_roi(y1, y2):
    try:
        with open(ROI_PATH, "w") as f:
            f.write(f"#v{PARAM_VERSION}\n")
            f.write(f"{y1},{y2}\n")
    except Exception:
        pass


# ============================================================
# 绘制
# ============================================================

def draw_status_bar(img, n_ball, n_raw, fps_h, led, led_btn,
                    anchor_btn, anchor_mode, roi_btn, roi_mode, tracking):
    # FPS 用 _time2.time() 算, 不用 ticks_ms
    now = _time2.time()
    fps_h.append(now)
    if len(fps_h) > FPS_WIN:
        fps_h.pop(0)
    if len(fps_h) >= 2:
        dt = fps_h[-1] - fps_h[0]
        fps = (len(fps_h) - 1) / dt if dt > 0 else 0
    else:
        fps = 0

    s = f"B:{n_ball} F:{fps:.0f}"
    img.draw_string(8, 8, s, color=C_WHITE, scale=1.0, thickness=1)
    if tracking:
        img.draw_string(8, 24, "TRACK", color=C_GREEN, scale=1.0, thickness=2)
    else:
        img.draw_string(8, 24, "LOST", color=C_RED, scale=1.0, thickness=2)

    led_on = led.is_on()
    bx, by, bw, bh = led_btn
    btn_c = C_YELLOW if led_on else C_GRAY
    img.draw_rect(bx, by, bw, bh, color=btn_c, thickness=-1)
    img.draw_rect(bx, by, bw, bh, color=C_WHITE, thickness=1)
    txt = "LED ON" if led_on else "LED OFF"
    img.draw_string(bx + 14, by + 10, txt,
                    color=C_BLACK if led_on else C_WHITE,
                    scale=1.0, thickness=2)

    # 锚点模式按钮
    ax, ay, aw, ah = anchor_btn
    ac = C_BLUE if anchor_mode else C_GRAY
    img.draw_rect(ax, ay, aw, ah, color=ac, thickness=-1)
    img.draw_rect(ax, ay, aw, ah, color=C_WHITE, thickness=1)
    img.draw_string(ax + 8, ay + 10, "ANCHOR",
                    color=image.Color.from_rgb(0, 0, 0) if anchor_mode else C_WHITE,
                    scale=1.0, thickness=2)

    # ROI 选点按钮
    rx, ry, rw, rh = roi_btn
    rc = C_GREEN if roi_mode else C_GRAY
    img.draw_rect(rx, ry, rw, rh, color=rc, thickness=-1)
    img.draw_rect(rx, ry, rw, rh, color=C_WHITE, thickness=1)
    img.draw_string(rx + 22, ry + 10, "ROI",
                    color=image.Color.from_rgb(0, 0, 0) if roi_mode else C_WHITE,
                    scale=1.0, thickness=2)


def draw_panel(img, detector, n_ball, n_raw, pid, fps_h,
               bar_y, bar_h, w, h,
               minus_btn, param_btn, plus_btn,
               bin_disp, led, led_btn, anchor_btn, anchor_mode,
               roi_btn, roi_mode, tracking, flash_btn=-1):
    draw_status_bar(img, n_ball, n_raw, fps_h, led, led_btn,
                    anchor_btn, anchor_mode, roi_btn, roi_mode, tracking)

    mid = w // 2
    if bin_disp is not None:
        try:
            img.draw_image(mid, 0, bin_disp)
        except Exception:
            pass
        img.draw_line(mid - 1, 0, mid - 1, bar_y, color=C_GRAY)

    # ---- 底部 3 个大按钮: [ - ] [ 参数名:值 ] [ + ] ----
    img.draw_rect(0, bar_y, w, bar_h, color=C_DARK, thickness=-1)
    img.draw_rect(0, bar_y, w, 2, color=C_GRAY, thickness=-1)

    plist = detector.get_params()
    val, vmin, vmax, step, name = plist[pid]

    # - 按钮
    mc = image.Color.from_rgb(255, 100, 100) if flash_btn == 0 else image.Color.from_rgb(160, 35, 35)
    img.draw_rect(*minus_btn, color=mc, thickness=-1)
    img.draw_string(minus_btn[0] + minus_btn[2] // 2 - 10,
                    minus_btn[1] + minus_btn[3] // 2 - 12,
                    "-", color=C_WHITE, scale=1.6, thickness=3)

    # 参数名按钮
    pc = image.Color.from_rgb(60, 60, 80) if flash_btn == 1 else C_DARK
    img.draw_rect(*param_btn, color=pc, thickness=-1)
    label = f"{name}:{val}"
    lw = len(label) * 7
    img.draw_string(param_btn[0] + param_btn[2] // 2 - lw // 2,
                    param_btn[1] + param_btn[3] // 2 - 12,
                    label, color=C_WHITE, scale=1.1, thickness=2)

    # + 按钮
    pc2 = image.Color.from_rgb(100, 255, 100) if flash_btn == 2 else image.Color.from_rgb(35, 150, 35)
    img.draw_rect(*plus_btn, color=pc2, thickness=-1)
    img.draw_string(plus_btn[0] + plus_btn[2] // 2 - 10,
                    plus_btn[1] + plus_btn[3] // 2 - 12,
                    "+", color=C_WHITE, scale=1.6, thickness=3)

    # 进度条 (横跨整个底部)
    bar_h2 = 4
    bar_y2 = bar_y + bar_h - bar_h2
    bar_w2 = w
    img.draw_rect(0, bar_y2, bar_w2, bar_h2, color=image.Color.from_rgb(50, 50, 55), thickness=-1)
    if vmax > vmin:
        ratio = (val - vmin) / (vmax - vmin)
        if ratio > 0:
            img.draw_rect(0, bar_y2, int(bar_w2 * ratio), bar_h2,
                          color=C_YELLOW, thickness=-1)


def draw_minimal(img, n_ball, n_raw, fps_h, led, led_btn,
                 anchor_btn, anchor_mode, roi_btn, roi_mode, tracking):
    draw_status_bar(img, n_ball, n_raw, fps_h, led, led_btn,
                    anchor_btn, anchor_mode, roi_btn, roi_mode, tracking)


def draw_detections(img, objs):
    for obj in objs:
        img.draw_cross(obj['cx'], obj['cy'], color=C_GREEN, size=12)
        q = obj['circ']
        qc = C_GREEN if q > 0.70 else (C_YELLOW if q > 0.50 else C_RED)
        tx = obj['cx'] + 10
        ty = max(obj['cy'] - 20, 8)
        if tx > img.width() - 40:
            tx = obj['cx'] - 40
        img.draw_string(tx, ty, f"{q:.2f}", color=qc, scale=0.75)


# ============================================================
# WebRTC 图传 + 浏览器录像
# ============================================================

STREAM_FPS = 60
STREAM_BITRATE = 3_000_000
STREAM_GOP = 30
WEB_HTTP_PORT = 8000
WEB_SIGNALING_PORT = 8001
STUN_SERVER = "stun:stun.miwifi.com:3478"
LOCAL_RECORD_DIR = "/root/maixcam2_recordings"
LOCAL_RECORD_FREE_MARGIN = 100 * 1024 * 1024
RECORDING_WRITE_LOCK = threading.Lock()

WEB_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>创源启明 maixcam2图传</title>
<style>
html,body{margin:0;min-height:100%;background:#3a3a3a;color:#fff;font-family:Arial,"Microsoft YaHei",sans-serif}
body{display:flex;justify-content:center;align-items:center;box-sizing:border-box;padding:24px}
#container{width:100%;max-width:1350px;text-align:center}
#title{margin-bottom:30px;padding:16px 20px;border-radius:16px;background:rgba(255,255,255,0.09);font-size:clamp(24px,3vw,42px);font-weight:bold;box-shadow:0 8px 28px rgba(0,0,0,0.12)}
#video-wrap{position:relative;overflow:hidden;width:100%;border-radius:22px;background:#111;box-shadow:0 18px 45px rgba(0,0,0,0.42)}
#video{display:block;width:100%;max-height:72vh;background:#111}
#resolution-badge,#status{position:absolute;bottom:18px;padding:8px 14px;border-radius:12px;background:rgba(0,0,0,0.62);font-size:16px;backdrop-filter:blur(6px)}
#resolution-badge{left:18px}#status{right:18px}
#record-controls{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:12px;margin:-12px 0 24px}
#record-target,#record-button{height:42px;border:1px solid rgba(255,255,255,0.18);border-radius:10px;color:#fff;font-size:16px}
#record-target{padding:0 12px;background:#505050}
#record-button{min-width:132px;padding:0 18px;background:#c63737;cursor:pointer}
#record-button:hover:not(:disabled){background:#dc4444}
#record-button.recording{background:#8d2020;box-shadow:0 0 0 4px rgba(220,68,68,0.18)}
#record-button:disabled,#record-target:disabled{cursor:not-allowed;opacity:0.55}
#record-timer{min-width:68px;font-family:Consolas,monospace;font-size:18px}
#record-message{flex-basis:100%;min-height:20px;color:#d7d7d7;font-size:14px}
#burst-controls{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:12px;margin:-12px 0 16px}
#burst-folder-btn,#burst-button{height:42px;border:1px solid rgba(255,255,255,0.18);border-radius:10px;color:#fff;font-size:16px;cursor:pointer}
#burst-folder-btn{background:#3a6;padding:0 14px;min-width:140px}
#burst-folder-btn:hover:not(:disabled){background:#4b8}
#burst-button{min-width:140px;padding:0 18px;background:#c60}
#burst-button:hover:not(:disabled){background:#e80}
#burst-button.bursting{background:#080;box-shadow:0 0 0 4px rgba(0,200,0,0.2)}
#burst-button:disabled,#burst-folder-btn:disabled{cursor:not-allowed;opacity:0.55}
#burst-count{min-width:60px;font-family:Consolas,monospace;font-size:18px}
#burst-status{flex-basis:100%;min-height:20px;color:#d7d7d7;font-size:14px}
#fullscreen-btn{position:fixed;right:30px;bottom:30px;display:none;width:56px;height:56px;border:0;border-radius:50%;background:rgba(255,255,255,0.18);color:#fff;font-size:24px;cursor:pointer;backdrop-filter:blur(8px)}
@media(max-width:600px){body{padding:12px}#title{margin-bottom:16px}#record-controls{margin:0 0 16px;gap:8px}#record-target,#record-button{height:38px;font-size:14px}#resolution-badge,#status{bottom:10px;padding:6px 10px;font-size:13px}#resolution-badge{left:10px}#status{right:10px}}
</style>
</head>
<body>
<main id="container">
<div id="title">创源启明 maixcam2图传</div>
<div id="record-controls">
<label for="record-target">保存位置</label>
<select id="record-target"><option value="computer">电脑端</option><option value="maixcam">MaixCam2 本地</option></select>
<button id="record-button" disabled>● 开始录制</button>
<span id="record-timer">00:00</span>
<div id="record-message">连接视频后即可录制</div>
</div>
<div id="burst-controls">
<button id="burst-folder-btn">📁 选择保存文件夹</button>
<button id="burst-button" disabled>⚡ 开始连拍 (10张/秒)</button>
<span id="burst-count">0 张</span>
<div id="burst-status">请先选择保存文件夹</div>
</div>
<div id="video-wrap">
<video id="video" autoplay playsinline muted></video>
<div id="resolution-badge"><span id="resolution">-- × --</span><span style="margin-left:8px;opacity:0.85">H264</span></div>
<div id="status">正在连接</div>
</div>
</main>
<button id="fullscreen-btn" title="全屏">⛶</button>
<script>
const signalingPort=__SIGNALING_PORT__;
const stunServer="__STUN_SERVER__";
const clientId=Math.random().toString(36).slice(2,12);
const socketUrl=`ws://${window.location.hostname}:${signalingPort}/${clientId}`;
const socket=new WebSocket(socketUrl);
const video=document.getElementById("video");
const statusBox=document.getElementById("status");
const resolutionBox=document.getElementById("resolution");
const fullscreenButton=document.getElementById("fullscreen-btn");
const recordTarget=document.getElementById("record-target");
const recordButton=document.getElementById("record-button");
const recordTimer=document.getElementById("record-timer");
const recordMessage=document.getElementById("record-message");
const recordingBitrate=__RECORD_BITRATE__;
let peer=null,mediaRecorder=null,recordedChunks=[],recordingDestination="computer",recordingStartedAt=0,recordingTimerId=null;
function setStatus(t){statusBox.textContent=t}
function updateResolution(){if(video.videoWidth&&video.videoHeight)resolutionBox.textContent=video.videoWidth+" × "+video.videoHeight}
function timestampName(){const n=new Date;const p=v=>String(v).padStart(2,"0");return n.getFullYear()+p(n.getMonth()+1)+p(n.getDate())+"_"+p(n.getHours())+p(n.getMinutes())+p(n.getSeconds())}
function elapsedText(e){const t=Math.floor(e/1000),o=Math.floor(t/3600),n=Math.floor(t%3600/60),r=t%60;const p=v=>String(v).padStart(2,"0");return o>0?p(o)+":"+p(n)+":"+p(r):p(n)+":"+p(r)}
function recorderExtension(e){return e&&e.toLowerCase().includes("mp4")?"mp4":"webm"}
function createRecorder(e){const t=["video/mp4;codecs=avc1.42E01E","video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm"];for(const o of t){if(MediaRecorder.isTypeSupported&&!MediaRecorder.isTypeSupported(o))continue;try{return new MediaRecorder(e,{mimeType:o,videoBitsPerSecond:recordingBitrate})}catch(e){console.warn("Recorder format unavailable: "+o,e)}}return new MediaRecorder(e,{videoBitsPerSecond:recordingBitrate})}
function downloadRecording(e,t){const o=URL.createObjectURL(e),n=document.createElement("a");n.href=o,n.download=t,document.body.appendChild(n),n.click(),n.remove(),setTimeout(()=>URL.revokeObjectURL(o),30000)}
async function uploadRecording(e,t,o){const n=await fetch("/api/recordings",{method:"POST",headers:{"Content-Type":e.type||"video/"+t,"X-Recording-Extension":t,"X-Recording-Timestamp":o},body:e});let r={};try{r=await n.json()}catch(e){r={}}if(!n.ok||!r.ok)throw new Error(r.error||"HTTP "+n.status);return r}
function restoreRecordControls(){const e=video.srcObject&&video.srcObject.getVideoTracks().some(e=>e.readyState==="live");recordButton.disabled=!e||!("MediaRecorder"in window)||mediaRecorder!==null;recordButton.classList.remove("recording");recordButton.textContent="● 开始录制";recordTarget.disabled=false}
async function finishRecording(){clearInterval(recordingTimerId);recordingTimerId=null;recordButton.disabled=true;recordButton.classList.remove("recording");recordButton.textContent="保存中...";const e=mediaRecorder.mimeType||(recordedChunks[0]&&recordedChunks[0].type)||"video/webm",t=recorderExtension(e),o=timestampName(),n=o+"."+t,r=new Blob(recordedChunks,{type:e});recordedChunks=[];if(!r.size){recordMessage.textContent="录制失败：没有产生视频数据";mediaRecorder=null;restoreRecordControls();return}try{if(recordingDestination==="maixcam"){const e=await uploadRecording(r,t,o);recordMessage.textContent="已保存到 MaixCam2："+e.path}else{downloadRecording(r,n);recordMessage.textContent="已保存到电脑："+n}}catch(e){console.error(e);if(recordingDestination==="maixcam"){downloadRecording(r,n);recordMessage.textContent="MaixCam2 保存失败，已转为电脑下载："+e.message}else{recordMessage.textContent="电脑保存失败："+e.message}}finally{mediaRecorder=null;restoreRecordControls()}}
function startRecording(){if(!video.srcObject||!("MediaRecorder"in window)){recordMessage.textContent="当前浏览器不支持录制或视频尚未连接";return}try{mediaRecorder=createRecorder(video.srcObject);recordedChunks=[];recordingDestination=recordTarget.value;mediaRecorder.ondataavailable=e=>{if(e.data&&e.data.size>0)recordedChunks.push(e.data)};mediaRecorder.onerror=e=>{console.error(e.error);recordMessage.textContent="录制错误："+e.error.message};mediaRecorder.onstop=finishRecording;mediaRecorder.start(1000);recordingStartedAt=performance.now();recordTimer.textContent="00:00";recordingTimerId=setInterval(()=>{recordTimer.textContent=elapsedText(performance.now()-recordingStartedAt)},500);recordTarget.disabled=true;recordButton.classList.add("recording");recordButton.textContent="■ 停止录制";recordMessage.textContent=recordingDestination==="maixcam"?"正在录制，停止后保存到 MaixCam2":"正在录制，停止后下载到电脑"}catch(e){console.error(e);mediaRecorder=null;recordMessage.textContent="无法开始录制："+e.message;restoreRecordControls()}}
function stopRecording(){if(mediaRecorder&&mediaRecorder.state==="recording"){recordButton.disabled=true;recordButton.textContent="正在停止...";mediaRecorder.stop()}}
function waitForIceGathering(e){if(e.iceGatheringState==="complete")return Promise.resolve();return new Promise(t=>{const o=()=>{if(e.iceGatheringState==="complete"){e.removeEventListener("icegatheringstatechange",o);t()}};e.addEventListener("icegatheringstatechange",o);setTimeout(()=>{e.removeEventListener("icegatheringstatechange",o);t()},3000)})}
async function handleOffer(e){if(peer){stopRecording();peer.close()}peer=new RTCPeerConnection({bundlePolicy:"max-bundle",iceServers:[{urls:[stunServer]}]});peer.ontrack=e=>{video.srcObject=e.streams[0];video.play();fullscreenButton.style.display="block";restoreRecordControls();if(!mediaRecorder)recordMessage.textContent="可选择保存位置并开始录制";setStatus("已连接")};peer.onconnectionstatechange=()=>{if(peer.connectionState==="failed"||peer.connectionState==="disconnected"||peer.connectionState==="closed"){stopRecording();recordButton.disabled=true;setStatus("连接断开")}};await peer.setRemoteDescription({type:e.type,sdp:e.sdp});await peer.setLocalDescription(await peer.createAnswer());await waitForIceGathering(peer);socket.send(JSON.stringify({id:"server",type:peer.localDescription.type,sdp:peer.localDescription.sdp}))}
socket.onopen=()=>{setStatus("正在获取视频");socket.send(JSON.stringify({id:"server",type:"request"}))};
socket.onmessage=async e=>{if(typeof e.data!=="string")return;const t=JSON.parse(e.data);if(t.type==="offer"){try{await handleOffer(t)}catch(e){console.error(e);setStatus("连接失败")}}};
socket.onerror=()=>setStatus("信令连接失败");
socket.onclose=()=>{if(!peer||peer.connectionState!=="connected")setStatus("信令已断开")};
video.addEventListener("loadedmetadata",updateResolution);
video.addEventListener("resize",updateResolution);
recordButton.onclick=()=>{if(mediaRecorder&&mediaRecorder.state==="recording"){stopRecording()}else if(!mediaRecorder){startRecording()}};
// ====== 连拍训练: 10张/秒 保存到本地文件夹 ======
const burstFolderBtn=document.getElementById("burst-folder-btn");
const burstButton2=document.getElementById("burst-button");
const burstCountEl=document.getElementById("burst-count");
const burstStatus=document.getElementById("burst-status");
let burstHandle=null,burstCount=0,burstDir=null,burstCanvas=null,burstCtx=null;
burstFolderBtn.onclick=async()=>{
try{
burstDir=await window.showDirectoryPicker({mode:"readwrite"});
burstFolderBtn.textContent="📁 "+burstDir.name;
burstStatus.textContent="已选文件夹,可开始连拍";
burstButton2.disabled=false;
burstCanvas=document.createElement("canvas");
burstCtx=burstCanvas.getContext("2d");
}catch(e){burstStatus.textContent="已取消选择"}
};
async function burstOneFrame(){
if(!burstDir||!video.srcObject||!video.videoWidth)return;
try{
burstCanvas.width=video.videoWidth;burstCanvas.height=video.videoHeight;
burstCtx.drawImage(video,0,0);
const blob=await new Promise(r=>burstCanvas.toBlob(r,"image/jpeg",0.92));
const pad=n=>String(n).padStart(6,"0");
const fh=await burstDir.getFileHandle("train_"+pad(burstCount)+".jpg",{create:true});
const w=await fh.createWritable();
await w.write(blob);await w.close();
burstCount++;burstCountEl.textContent=burstCount+" 张";
burstStatus.textContent="连拍中... "+burstCount+" 张";
}catch(e){burstStatus.textContent="写入失败: "+e.message}
}
burstButton2.onclick=()=>{
if(burstHandle){
clearInterval(burstHandle);burstHandle=null;
burstButton2.classList.remove("bursting");
burstButton2.textContent="⚡ 开始连拍 (10张/秒)";
burstFolderBtn.disabled=false;
burstStatus.textContent="已停止,共 "+burstCount+" 张";
}else{
burstCount=0;burstCountEl.textContent="0 张";
burstStatus.textContent="连拍中...";
burstHandle=setInterval(burstOneFrame,100);
burstButton2.classList.add("bursting");
burstButton2.textContent="⏹ 停止连拍";
burstFolderBtn.disabled=true;
}
};
window.addEventListener("beforeunload",e=>{if(mediaRecorder&&mediaRecorder.state==="recording"){e.preventDefault();e.returnValue=""}});
fullscreenButton.onclick=()=>{if(!document.fullscreenElement){video.requestFullscreen?.()}else{document.exitFullscreen?.()}};
</script>
</body>
</html>
"""

WEB_PAGE_BYTES = (
    WEB_PAGE.replace("__SIGNALING_PORT__", str(WEB_SIGNALING_PORT))
    .replace("__STUN_SERVER__", STUN_SERVER)
    .replace("__RECORD_BITRATE__", str(STREAM_BITRATE))
    .encode("utf-8")
)


class WebPageHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(WEB_PAGE_BYTES)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(WEB_PAGE_BYTES)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/recordings":
            self.send_json(404, {"ok": False, "error": "接口不存在"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self.send_json(400, {"ok": False, "error": "录像数据为空"})
            return
        extension = self.headers.get("X-Recording-Extension", "webm").lower()
        if extension not in ("mp4", "webm"):
            self.send_json(400, {"ok": False, "error": "不支持的录像格式"})
            return
        temporary_path = None
        try:
            os.makedirs(LOCAL_RECORD_DIR, exist_ok=True)
            free_bytes = shutil.disk_usage(LOCAL_RECORD_DIR).free
            if content_length + LOCAL_RECORD_FREE_MARGIN > free_bytes:
                self.send_json(507, {"ok": False, "error": "MaixCam2 存储空间不足，请清理录像后重试"})
                return
            with RECORDING_WRITE_LOCK:
                requested_timestamp = self.headers.get("X-Recording-Timestamp", "")
                ts_valid = (
                    len(requested_timestamp) == 15
                    and requested_timestamp[8] == "_"
                    and requested_timestamp[:8].isdigit()
                    and requested_timestamp[9:].isdigit()
                )
                timestamp = requested_timestamp if ts_valid else datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}.{extension}"
                final_path = os.path.join(LOCAL_RECORD_DIR, filename)
                seq = 1
                while os.path.exists(final_path):
                    filename = f"{timestamp}_{seq:02d}.{extension}"
                    final_path = os.path.join(LOCAL_RECORD_DIR, filename)
                    seq += 1
                temporary_path = f"{final_path}.part"
                remaining = content_length
                with open(temporary_path, "wb") as f:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ConnectionError("录像上传中断")
                        f.write(chunk)
                        remaining -= len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary_path, final_path)
                temporary_path = None
            print(f"Recording saved: {final_path} ({content_length} bytes)")
            self.send_json(200, {"ok": True, "filename": filename, "path": final_path, "size": content_length})
        except Exception as exc:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
            print(f"Failed to save recording: {exc}")
            self.send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format, *args):
        return


class WebPageServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_web_page_server():
    server = WebPageServer(("", WEB_HTTP_PORT), WebPageHandler)
    thread = threading.Thread(target=server.serve_forever, name="web-page", daemon=True)
    try:
        thread.start()
    except Exception:
        server.server_close()
        raise
    return server, thread


def stop_web_page_server(server, thread):
    if server is None:
        return
    try:
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=2)
    except Exception as exc:
        print(f"Warning: failed to stop web page server: {exc}")


# ============================================================
# 主程序
# ============================================================

def main():
    det = BallDetector()
    load_params(det)

    # ====== WebRTC 图传: 双通道 (NV21编码 + RGB检测) ======
    dis = display.Display()
    W, H = dis.width(), dis.height()

    stream_cam = camera.Camera(CAM_W, CAM_H, image.Format.FMT_YVU420SP, fps=STREAM_FPS)
    preview_cam = stream_cam.add_channel(W, H, image.Format.FMT_RGB888, fps=STREAM_FPS)

    web_server = webrtc.WebRTC(
        stream_type=webrtc.WebRTCStreamType.WEBRTC_STREAM_H264,
        rc_type=webrtc.WebRTCRCType.WEBRTC_RC_CBR,
        bitrate=STREAM_BITRATE, gop=STREAM_GOP,
        signaling_port=WEB_SIGNALING_PORT, stun_server=STUN_SERVER,
        http_server=False,
    )
    err.check_raise(web_server.bind_camera(stream_cam))
    err.check_raise(web_server.start())

    page_server, page_thread = start_web_page_server()
    urls = web_server.get_urls()
    print(f"[stream] 图传: {urls[0] if urls else f'http://设备IP:{WEB_HTTP_PORT}'}")
    # ==========================================================
    ts = touchscreen.TouchScreen()
    led = LED()
    comm = SerialComm()
    comm.open()

    BAR_H = 80
    BAR_Y = H - BAR_H

    led_btn, anchor_btn, roi_btn, minus_btn, param_btn, plus_btn = build_buttons(BAR_Y, BAR_H, W)

    show_tune = [True]
    pid = [0]
    cnt_h = []
    fps_h = []
    det_skip = 0
    detect_cache = ([], 0, None)
    touch_pos = (0, 0)
    flash_btn = -1
    flash_t = 0
    last_act = 0
    bin_skip = 0             # 二值化预览缓存计数
    bin_disp = None          # 缓存的二值化MaixPy图像
    target_x = W // 2        # 目标锚点 x (触摸设置)
    anchor_mode = False      # 锚点模式开关
    roi_y1 = 0               # ROI 上边界
    roi_y2 = BAR_Y           # ROI 下边界 (相机画面高度)
    roi_mode = False         # ROI 选点模式
    roi_step = 0             # 0=等下y1, 1=等下y2

    # 加载保存的 ROI
    v = load_roi()
    if v and len(v) == 2 and 0 <= v[0] < v[1] <= BAR_Y:
        roi_y1, roi_y2 = v[0], v[1]
        print(f"[main] 加载ROI: y={roi_y1}~{roi_y2}")

    print(f"[main] 钢球检测启动  cam={CAM_W}x{CAM_H}  参数={det.get_values()}")

    while not app.need_exit():

        # ---- 触摸 (JiGuang 同款: _time.time() + pressed + 简单减法) ----
        x, y, p = ts.read()
        touch_pos = (x, y)
        now = _time2.time() * 1000

        if p and (now - last_act) > TOUCH_CD:
            last_act = now

            if y < BAR_Y:
                if is_in_button(x, y, led_btn):
                    led.toggle()
                    flash_btn = -2
                    flash_t = time.ticks_ms()
                elif is_in_button(x, y, anchor_btn):
                    anchor_mode = not anchor_mode
                    roi_mode = False
                    bin_disp = None
                elif is_in_button(x, y, roi_btn):
                    roi_mode = not roi_mode
                    roi_step = 0
                    anchor_mode = False
                    bin_disp = None
                elif roi_mode:
                    # ROI 选点: 第一次点=上y, 第二次点=下y
                    if roi_step == 0:
                        roi_y1 = y
                        roi_step = 1
                    else:
                        roi_y2_click = y
                        if abs(roi_y2_click - roi_y1) < 10:
                            # 太近, 忽略
                            roi_step = 0
                        else:
                            if roi_y2_click > roi_y1:
                                roi_y2 = roi_y2_click
                            else:
                                roi_y2 = roi_y1
                                roi_y1 = roi_y2_click
                            save_roi(roi_y1, roi_y2)
                            roi_step = 0
                            roi_mode = False
                    bin_disp = None
                elif anchor_mode:
                    # 锚点模式: 触摸设置目标 x
                    target_x = x
                    bin_disp = None
                else:
                    # 非锚点模式: 右半屏切换参数, 左半屏切换调参/全屏
                    if x >= W // 2:
                        n_param = len(det.get_params())
                        pid[0] = (pid[0] + 1) % n_param
                    else:
                        show_tune[0] = not show_tune[0]
            elif show_tune[0]:
                n_param = len(det.get_params())
                if is_in_button(x, y, minus_btn):
                    flash_btn, flash_t = 0, time.ticks_ms()
                    det.adjust_param(pid[0], -1); save_params(det)
                    bin_disp = None
                elif is_in_button(x, y, param_btn):
                    flash_btn, flash_t = 1, time.ticks_ms()
                    pid[0] = (pid[0] + 1) % n_param
                elif is_in_button(x, y, plus_btn):
                    flash_btn, flash_t = 2, time.ticks_ms()
                    det.adjust_param(pid[0], +1); save_params(det)
                    bin_disp = None

        # 闪烁超时
        tn = time.ticks_ms()
        if flash_btn >= 0 and time.ticks_diff(tn, flash_t) > 300:
            flash_btn = -1

        # ---- 采图 ----
        img = preview_cam.read()
        if img is None:
            time.sleep_ms(1)
            continue

        if W != CAM_W or H != CAM_H:
            img = img.resize(W, H)

        # ---- 检测 (半分辨率 + 跳帧) ----
        skip = 1 if show_tune[0] else DET_SKIP   # 调参模式每帧检测确保参数反馈及时
        det_skip += 1
        if det_skip >= skip:
            det_skip = 0
            objs, binary = det.detect_with_binary(img, roi_y1, roi_y2)
            n_ball = BallDetector.stable_count(len(objs), cnt_h, CNT_HIST)
            detect_cache = (objs, n_ball, binary)
            comm.send_x(objs[0]['cx'] if objs else -1, target_x)
        else:
            objs, n_ball, binary = detect_cache

        cur_x = objs[0]['cx'] if objs else -1

        # ---- 二值化预览 (OpenCV binary → 拉大填满右面板) ----
        bin_skip += 1
        if binary is not None and (bin_skip >= 3 or bin_disp is None):
            bin_skip = 0
            try:
                b_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                b_rgb = cv2.resize(b_rgb, (W // 2, BAR_Y))
                bin_disp = image.cv2image(b_rgb, False, False)
            except Exception:
                pass

        # ---- 绘制 ----
        has_roi = (roi_y1 > 0 or roi_y2 < BAR_Y)
        if has_roi and not roi_mode:
            # 遮掉ROI外的区域 (全黑), 含按钮栏
            if roi_y1 > 0:
                img.draw_rect(0, 0, W, roi_y1, color=image.COLOR_BLACK, thickness=-1)
            if roi_y2 < H:
                img.draw_rect(0, roi_y2, W, H - roi_y2, color=image.COLOR_BLACK, thickness=-1)

        draw_detections(img, objs)

        # ROI 边界线 (青色横线)
        if has_roi or roi_mode:
            img.draw_line(0, roi_y1, W, roi_y1, color=C_CYAN, thickness=2)
            img.draw_line(0, roi_y2, W, roi_y2, color=C_CYAN, thickness=2)
            if roi_mode and roi_step == 1:
                img.draw_string(W // 2 - 40, 70, "TAP y2", color=C_YELLOW, scale=1.2)
            elif roi_mode:
                img.draw_string(W // 2 - 40, 70, "TAP y1", color=C_YELLOW, scale=1.2)

        # 目标锚点: 竖线 + 十字
        img.draw_line(target_x, 0, target_x, BAR_Y, color=C_BLUE, thickness=2)
        img.draw_cross(target_x, BAR_Y // 2, color=C_YELLOW, size=12)
        img.draw_string(target_x + 6, 30, f"TGT:{target_x}", color=C_BLUE, scale=0.9)
        img.draw_string(target_x + 6, 48, f"CUR:{cur_x}", color=C_GREEN if cur_x >= 0 else C_RED, scale=0.9)

        # 触点十字 (调试)
        img.draw_cross(touch_pos[0], touch_pos[1], color=C_YELLOW, size=12)

        tracking = len(objs) > 0
        if show_tune[0]:
            draw_panel(img, det, n_ball, len(objs), pid[0], fps_h,
                       BAR_Y, BAR_H, W, H,
                       minus_btn, param_btn, plus_btn,
                       bin_disp, led, led_btn, anchor_btn, anchor_mode,
                       roi_btn, roi_mode, tracking, flash_btn)
        else:
            draw_minimal(img, n_ball, len(objs), fps_h, led, led_btn,
                         anchor_btn, anchor_mode, roi_btn, roi_mode, tracking)

        dis.show(img)


if __name__ == "__main__":
    main()
