#!/usr/bin/python3
# 说 · Shuo — 底部状态胶囊（XWayland 置顶小窗）
#
# 为什么不用 GNOME Shell 扩展 / layer-shell：
#   - GNOME/Mutter 不支持 wlr-layer-shell；
#   - Shell 扩展要用户注销重登录才会被加载（依赖太重）。
# XWayland 窗口由 mutter 按 EWMH 管理：keep_above + accept_focus=false + 输入区域清空
# 就能做到「置顶、不抢焦点、鼠标穿透」，而且系统只需要已有的 GTK3 / PyGObject。
#
# 协议：监听 $XDG_RUNTIME_DIR/shuo-overlay.sock，收到一行 recording / processing / idle。
# 应用退出（连接断开）时自动退出。
#
# SPDX-License-Identifier: GPL-3.0-or-later

import cairo
import json
import math
import os
import random
import socket
import sys
import threading
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

SOCK = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "shuo-overlay.sock")
HEIGHT = 38
PAD_X = 10
GAP = 6
BAR_N = 9
BAR_W = 2.4
BAR_GAP = 3.0
BARS_W = BAR_N * BAR_W + (BAR_N - 1) * BAR_GAP
MARGIN_BOTTOM = 64

MODE = "idle"
PHASE = 0.0
BARS = [random.uniform(0.15, 1.0) for _ in range(9)]
TARGETS = [random.uniform(0.15, 1.0) for _ in range(9)]

ZH = True
try:
    cfg = json.load(open(os.path.expanduser("~/.shuo/config.json"), encoding="utf-8"))
    lang = str(cfg.get("language", "en"))
    ZH = lang.startswith("zh") or lang == "yue"
except Exception:
    pass

LABELS = {
    "recording": "录音中" if ZH else "Recording",
    "processing": "识别中" if ZH else "Working",
}


class Overlay(Gtk.Window):
    def __init__(self):
        super().__init__(type=Gtk.WindowType.TOPLEVEL)
        self.set_decorated(False)
        self.set_app_paintable(True)
        self.set_keep_above(True)
        self.set_accept_focus(False)
        self.set_focus_on_map(False)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_resizable(False)
        self._w = 160
        self._text_w = 50
        self.set_size_request(self._w, HEIGHT)
        self.set_type_hint(Gdk.WindowTypeHint.UTILITY)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual is not None:
            self.set_visual(visual)

        self.area = Gtk.DrawingArea()
        self.area.set_size_request(self._w, HEIGHT)
        self.area.connect("draw", self.on_draw)
        self.add(self.area)

        self.connect("realize", self.on_realize)
        self.move_to_bottom()
        GLib.timeout_add(40, self.tick)
        self.hide()

    def apply_geometry(self, label):
        layout = self.create_pango_layout(label)
        tw, _ = layout.get_pixel_size()
        width = PAD_X * 2 + tw + GAP + int(BARS_W)
        self._w = width
        self._text_w = tw
        self.set_size_request(width, HEIGHT)
        self.resize(width, HEIGHT)
        self.area.set_size_request(width, HEIGHT)
        self.move_to_bottom()

    def move_to_bottom(self):
        screen = self.get_screen()
        if screen is None:
            return
        monitor = screen.get_primary_monitor()
        geo = screen.get_monitor_geometry(monitor)
        x = geo.x + (geo.width - self._w) // 2
        y = geo.y + geo.height - HEIGHT - MARGIN_BOTTOM
        self.move(x, y)

    def on_realize(self, *_):
        self.move_to_bottom()
        win = self.get_window()
        if win is not None:
            try:
                # 输入区域清空：鼠标点击穿透
                win.input_shape_combine_region(cairo.Region())
            except Exception:
                pass

    def on_draw(self, w, cr):
        alloc = w.get_allocation()
        width, height = alloc.width, alloc.height
        cr.set_operator(cairo.OPERATOR_CLEAR)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        cr.new_sub_path()
        r = (height - 2) / 2
        cr.arc(width - 1 - r, 1 + r, r, -math.pi / 2, 0)
        cr.arc(width - 1 - r, height - 1 - r, r, 0, math.pi / 2)
        cr.arc(1 + r, height - 1 - r, r, math.pi / 2, math.pi)
        cr.arc(1 + r, 1 + r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()
        cr.set_source_rgba(0.055, 0.071, 0.094, 0.84)
        cr.fill_preserve()
        cr.set_source_rgba(1, 1, 1, 0.10)
        cr.set_line_width(1)
        cr.stroke()

        label = LABELS.get(MODE, "")
        cr.set_source_rgba(0.93, 0.95, 0.97, 1)
        cr.select_font_face(
            "Noto Sans CJK SC",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL,
        )
        cr.set_font_size(13)
        ext = cr.text_extents(label)
        tx = PAD_X
        ty = height / 2 + ext.height / 2 - 1
        cr.move_to(tx, ty)
        cr.show_text(label)
        cr.new_path()

        ax = tx + self._text_w + GAP
        aw = width - ax - PAD_X
        cy = height / 2
        if MODE == "processing":
            cx = ax + aw / 2
            rr = 8
            cr.set_line_width(2.2)
            cr.new_sub_path()
            cr.set_source_rgba(1, 1, 1, 0.18)
            cr.arc(cx, cy, rr, 0, 2 * math.pi)
            cr.stroke()
            a = (PHASE * 4.2) % (2 * math.pi)
            cr.new_sub_path()
            cr.set_source_rgba(1, 0.42, 0.37, 1)
            cr.arc(cx, cy, rr, a, a + 1.5)
            cr.stroke()
            cr.new_path()
        elif MODE == "recording":
            total = BAR_N * BAR_W + (BAR_N - 1) * BAR_GAP
            x0 = ax + (aw - total) / 2
            for i in range(BAR_N):
                bh = 3.0 + BARS[i] * 10.5
                cr.set_source_rgba(1, 0.54, 0.45, 0.95)
                cr.rectangle(x0 + i * (BAR_W + BAR_GAP), cy - bh / 2, BAR_W, bh)
                cr.fill()

    def tick(self):
        global PHASE
        if MODE != "idle":
            PHASE += 0.04
            if MODE == "recording":
                for i in range(len(BARS)):
                    BARS[i] += (TARGETS[i] - BARS[i]) * 0.28
                    if abs(BARS[i] - TARGETS[i]) < 0.06:
                        TARGETS[i] = random.uniform(0.12, 1.0)
            self.area.queue_draw()
        return True


overlay = None
PRIMARY = None


def apply_state(cmd):
    global MODE
    if cmd not in ("recording", "processing", "idle"):
        return False
    MODE = cmd
    if cmd == "recording":
        for i in range(len(BARS)):
            TARGETS[i] = random.uniform(0.12, 1.0)
    overlay.apply_geometry(LABELS.get(cmd, ""))
    if cmd == "idle":
        overlay.hide()
    else:
        overlay.move_to_bottom()
        if not overlay.get_visible():
            overlay.show_all()
            overlay.move_to_bottom()
        overlay.area.queue_draw()
    return False


def handle(conn):
    global PRIMARY
    if PRIMARY is None:
        PRIMARY = conn
    primary = PRIMARY is conn
    try:
        f = conn.makefile("r", encoding="utf-8")
        for line in f:
            GLib.idle_add(apply_state, line.strip())
    except Exception:
        pass
    finally:
        # 只有应用的主连接断开才退出；send() 的短连接断开不影响
        if primary:
            GLib.idle_add(Gtk.main_quit)


def serve():
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o600)
    srv.listen(4)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def main():
    global overlay
    try:
        overlay = Overlay()
    except Exception as e:
        print(f"shuo-overlay: 初始化失败：{e}", file=sys.stderr)
        return 1
    threading.Thread(target=serve, daemon=True).start()
    Gtk.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
