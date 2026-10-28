"""
main.py —— MaixCAM2 钢球检测 (LED + OpenCV 亮点)

原理: 打开补光 LED → 钢球表面反光形成亮点 → OpenCV 检测亮点 → 计数 + 定位

触摸:
  右上角 LED 按钮 → 开关补光灯
  左半画面        → 切换调参/全屏模式
  右半画面        → 重置计数
  底部参数按钮    → 选中/调节检测参数
"""

from maix import app, camera, display, image, nn, time, touchscreen, uart, pinmap, gpio, err
import cv2, numpy as np, struct, os
import time as _time2

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
        self.max_area = 5000

        # 圆度
        self.min_circ = 0.20

        # 去重
        self.margin = 3

        # 单球追踪
        self._last_cx = None
        self._last_cy = None
        self._max_dist = 80
        self._max_lost = 10
        self._lost_frames = 0

    def detect(self, img_maix):
        objs, _ = self.detect_with_binary(img_maix)
        return objs

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

        # ---- 高斯模糊 (核>1时才跑, 跳过快) ----
        bs = self.blur_ksize
        if bs > 1:
            if bs % 2 == 0:
                bs += 1
            blur = cv2.GaussianBlur(gray, (bs, bs), 0)
        else:
            blur = gray

        # ---- 单阈值二值化 ----
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

            cx = int((bx + bw / 2) * Sx)
            cy = int((by + bh / 2) * Sy) + roi_y1
            candidates.append({
                'x':  int(bx * Sx),  'y':  int(by * Sy) + roi_y1,
                'w':  int(bw * Sx),  'h':  int(bh * Sy),
                'cx': cx, 'cy': cy,
                'area': int(area * SA), 'circ': circ,
                'score': area * circ,
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
            (self.max_area,       100, 8000, 100,  "MAX_A"),
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


# ============================================================
# 配置
# ============================================================

CAM_W, CAM_H = 640, 480
CAM_FPS = 60
FPS_WIN = 10
CNT_HIST = 5
DET_SKIP = 2
TOUCH_CD = 250
PARAM_PATH = "/root/ball_cv.txt"
ROI_PATH   = "/root/ball_roi.txt"

C_RED    = image.COLOR_RED
C_GREEN  = image.COLOR_GREEN
C_YELLOW = image.Color.from_rgb(255, 255, 0)
C_WHITE  = image.COLOR_WHITE
C_GRAY   = image.Color.from_rgb(140, 140, 140)
C_DARK   = image.Color.from_rgb(30, 30, 35)
C_BLUE   = image.Color.from_rgb(60, 130, 255)


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
            sv = [int(x) for x in f.read().strip().split(",")]
            if len(sv) == len(plist):
                detector.apply_params(sv)
                print("[main] 加载已保存参数:", sv)
    except Exception:
        print("[main] 使用默认参数")


def save_params(detector):
    try:
        with open(PARAM_PATH, "w") as f:
            f.write(",".join(str(x) for x in detector.get_values()))
    except Exception:
        pass


def save_roi(y1, y2):
    try:
        with open(ROI_PATH, "w") as f:
            f.write(f"{y1},{y2}")
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
                    color=image.Color.from_rgb(0, 0, 0) if led_on else C_WHITE,
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
        img.draw_rect(obj['x'], obj['y'], obj['w'], obj['h'],
                      color=C_RED, thickness=2)
        img.draw_cross(obj['cx'], obj['cy'], color=C_GREEN, size=5)
        q = obj['circ']
        qc = C_GREEN if q > 0.70 else (C_YELLOW if q > 0.50 else C_RED)
        lx = obj['x'] + obj['w'] + 4
        if lx > img.width() - 30:
            lx = obj['x'] - 30
        img.draw_string(lx, obj['y'], f"{q:.2f}", color=qc, scale=0.75)


# ============================================================
# 主程序
# ============================================================

def main():
    det = BallDetector()
    load_params(det)

    cam = camera.Camera(CAM_W, CAM_H, fps=CAM_FPS)
    dis = display.Display()
    ts = touchscreen.TouchScreen()
    led = LED()
    comm = SerialComm()
    comm.open()

    W, H = dis.width(), dis.height()

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
    try:
        with open(ROI_PATH) as f:
            v = [int(x) for x in f.read().strip().split(",")]
            if len(v) == 2 and 0 <= v[0] < v[1] <= BAR_Y:
                roi_y1, roi_y2 = v[0], v[1]
                print(f"[main] 加载ROI: y={roi_y1}~{roi_y2}")
    except Exception:
        pass

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
                    pl = det.get_params()
                    val, vmin, vmax, step, _ = pl[pid[0]]
                    val = max(vmin, val - step)
                    nv = det.get_values(); nv[pid[0]] = val
                    det.apply_params(nv); save_params(det)
                    bin_disp = None    # 立即刷新预览
                elif is_in_button(x, y, param_btn):
                    flash_btn, flash_t = 1, time.ticks_ms()
                    pid[0] = (pid[0] + 1) % n_param
                elif is_in_button(x, y, plus_btn):
                    flash_btn, flash_t = 2, time.ticks_ms()
                    pl = det.get_params()
                    val, vmin, vmax, step, _ = pl[pid[0]]
                    val = min(vmax, val + step)
                    nv = det.get_values(); nv[pid[0]] = val
                    det.apply_params(nv); save_params(det)
                    bin_disp = None    # 立即刷新预览

        # 闪烁超时
        tn = time.ticks_ms()
        if flash_btn >= 0 and time.ticks_diff(tn, flash_t) > 300:
            flash_btn = -1

        # ---- 采图 ----
        img = cam.read()
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

            # 串口发送: 当前球x + 目标x
            cur_x = objs[0]['cx'] if objs else -1
            comm.send_x(cur_x, target_x)
        else:
            objs, n_ball, binary = detect_cache

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
            img.draw_line(0, roi_y1, W, roi_y1, color=image.Color.from_rgb(0, 200, 200), thickness=2)
            img.draw_line(0, roi_y2, W, roi_y2, color=image.Color.from_rgb(0, 200, 200), thickness=2)
            if roi_mode and roi_step == 1:
                img.draw_string(W // 2 - 40, 70, "TAP y2", color=C_YELLOW, scale=1.2)
            elif roi_mode:
                img.draw_string(W // 2 - 40, 70, "TAP y1", color=C_YELLOW, scale=1.2)

        # 目标锚点: 竖线 + 十字
        img.draw_line(target_x, 0, target_x, BAR_Y, color=C_BLUE, thickness=2)
        img.draw_cross(target_x, BAR_Y // 2, color=C_YELLOW, size=12)
        cur_x = objs[0]['cx'] if objs else -1
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
