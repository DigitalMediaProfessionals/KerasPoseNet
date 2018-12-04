#!/usr/bin/python3
from __future__ import division, print_function
"""
    Copyright 2018 Digital Media Professionals Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
"""Posenet inference.

Uses parts of code from:
https://github.com/ildoonet/tf-pose-estimation/
git SHA1 ID: b119759e8a41828c633bd39b5c883bf5a56a214f
Apache 2.0 License
"""
import sys


if __name__ != "__main__":
    sys.stderr.write("%s must be used as main module\n" % __file__)
    sys.exit(1)


import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("INPUT", type=str,
                    help="Input video file or capture device in OpenCV format")
parser.add_argument("-p", "--peaks_threshold", type=float, default=0.2,
                    help="Peaks threshold")
parser.add_argument("-l", "--flog", type=str, default="run.log",
                    help="Additional file to write logs to")
args = parser.parse_args()


import logging
logging.basicConfig(level=logging.DEBUG)

import os

thisdir = os.path.dirname(__file__)
if not len(thisdir):
    thisdir = "."
if thisdir not in sys.path:
    sys.path.append(thisdir)

import time

logging.debug("Importing modules...")
t0 = time.time()
import numpy as np
logging.debug("Imported numpy in %.3f sec", time.time() - t0)
t0 = time.time()
import cv2
logging.debug("Imported cv2 in %.3f sec", time.time() - t0)
t0 = time.time()
import py_keras_pose_net as pose_net
logging.debug("Imported py_keras_pose_net in %.3f sec", time.time() - t0)
t0 = time.time()
from common import estimate_paf, draw_humans
logging.debug("Imported common in %.3f sec", time.time() - t0)

import ctypes
from ctypes import c_size_t, c_int, c_float

import curses

logging.debug("Imported modules")


lib_fb = ctypes.CDLL("%s/fb.so" % thisdir)

init_fb = lib_fb.init_fb
init_fb.restype = c_int

release_fb = lib_fb.release_fb

get_screen_width = lib_fb.get_screen_width
get_screen_width.restype = c_int

get_screen_height = lib_fb.get_screen_height
get_screen_height.restype = c_int

get_console_mode = lib_fb.get_console_mode
get_console_mode.restype = c_int

set_console_mode = lib_fb.set_console_mode
set_console_mode.argtypes = [c_int]
set_console_mode.restype = c_int

swap_buffer = lib_fb.swap_buffer
swap_buffer.restype = c_int

get_frame_ptr = lib_fb.get_frame_ptr
get_frame_ptr.restype = c_size_t

copy_to_frame = lib_fb.copy_to_frame
copy_to_frame.argtypes = [c_size_t, c_int]
copy_to_frame.restype = c_int


lib_mx = ctypes.CDLL("%s/mx.so" % thisdir)

extract_heatmap_peaks = lib_mx.extract_heatmap_peaks
extract_heatmap_peaks.argtypes = [c_size_t, c_size_t, c_float]
extract_heatmap_peaks.restype = c_int


FONT = cv2.FONT_HERSHEY_SIMPLEX


def array_address(a):
    addr = int(a.__array_interface__["data"][0])
    b = np.ascontiguousarray(a)
    assert int(b.__array_interface__["data"][0]) == addr
    return addr


class Logger(object):
    def __init__(self, stdscr, flog):
        self.stdscr = stdscr
        self.flog = flog

    def _log_msg(self, msg):
        self.flog.write("%s\n" % msg)
        self.flog.flush()
        sys.stdout.write("%s\r\n" % msg)
        self.stdscr.refresh()

    def info(self, msg, *args):
        self._log_msg("INFO: %s" % (msg % args))

    def error(self, msg, *args):
        self._log_msg("ERROR: %s" % (msg % args))

    def warning(self, msg, *args):
        self._log_msg("WARNING: %s" % (msg % args))


class FB(Logger):
    def __init__(self, stdscr, flog):
        self.was_graphics = False
        super(FB, self).__init__(stdscr, flog)

        self.info("Initializing linux framebuffer...")
        if init_fb():
            raise RuntimeError("init_fb() failed")
        self.was_graphics = (get_console_mode() == 1)
        if not self.was_graphics:
            self.info("Switching console to graphics mode")
            set_console_mode(1)
        else:
            self.info("Console is already in graphics mode")
        self.info("Initialized linux framebuffer")

    def update(self, frame_data):
        frame_data = np.ascontiguousarray(frame_data)
        if copy_to_frame(
                int(frame_data.__array_interface__["data"][0]),
                frame_data.nbytes):
            raise RuntimeError("Could not copy data to framebuffer")
        if swap_buffer():
            raise RuntimeError("swap_buffer() failed")

    def release(self):
        if not self.was_graphics:
            self.info("Switching console to text mode")
            set_console_mode(0)
            self.was_graphics = (get_console_mode() == 1)
        else:
            self.info("Console was in graphics mode before, "
                      "will not switch it to text mode")
        release_fb()


class Main(Logger):
    def __init__(self, stdscr, flog, capture_source, peaks_threshold):
        super(Main, self).__init__(stdscr, flog)
        self.fb = FB(self.stdscr, flog)
        self.capture_source = capture_source
        self.peaks_threshold = peaks_threshold

    def execute(self):
        self.stdscr.nodelay(True)

        self.info("Initializing PoseNet...")
        self.net = pose_net.create()
        if not pose_net.initialize(self.net):
            raise RuntimeError("Failed to initialize PoseNet")
        if not pose_net.load_weights(self.net, "KerasPoseNet_weights.bin"):
            raise RuntimeError(
                "Failed to load weights from KerasPoseNet_weights.bin")
        if not pose_net.commit(self.net):
            raise RuntimeError("Failed to commit PoseNet")
        self.info("Successfully initialized PoseNet")

        if not self.open_video_capture():
            return

        width = 432
        height = 368
        out_width = width // 8
        out_height = height // 8
        peaks_chw = np.zeros((18, out_height, out_width), dtype=np.float32)
        heatmap_hwc = np.zeros((out_height, out_width, 18), dtype=np.float32)
        paf_hwc = np.zeros((out_height, out_width, 38), dtype=np.float32)
        input_data_hwc = np.zeros((height, width, 3), dtype=np.uint8)

        frame = np.zeros([get_screen_height(), get_screen_width(), 3],
                         dtype=np.uint8)

        image_original = None
        image_resized = None
        image_resized_bgr = None
        while True:
            if image_original is None:
                ret_val, image_original = self.cap.read()
            else:
                ret_val, image_original = self.cap.read(image_original)
            if not ret_val:
                self.info("End of %s reached, will reopen",
                          self.capture_source)
                self.cap.release()
                self.cap = None
                image_original = None
                image_resized = None
                image_resized_bgr = None
                t0 = time.time()
                import gc
                gc.collect()
                self.info("Garbage collection completed in %.3f sec",
                          time.time() - t0)
                if not self.open_video_capture():
                    return
                continue

            if image_resized is None:
                orig_w = image_original.shape[1]
                orig_h = image_original.shape[0]
                dst_w = width
                dst_h = orig_h * dst_w // orig_w
                if dst_h > height:
                    dst_h = height
                    dst_w = orig_w * dst_h // orig_h
                image_resized = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
                image_resized_bgr = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

            t00 = time.time()

            t0pre = t00
            cv2.resize(image_original, (dst_w, dst_h), image_resized_bgr)
            cv2.cvtColor(image_resized_bgr, cv2.COLOR_BGR2RGB,
                         dst=image_resized)

            # import pydevd
            # pydevd.settrace("172.16.40.212")

            if input_data_hwc.shape != image_resized.shape:
                input_data_hwc[:] = 0
                y_offs = (height - image_resized.shape[0]) >> 1
                x_offs = (width - image_resized.shape[1]) >> 1
                input_data_hwc[y_offs:y_offs + image_resized.shape[0],
                               x_offs:x_offs + image_resized.shape[1]] = \
                    image_resized
            else:
                input_data_hwc[:] = image_resized
            input_data_whc16 = (np.moveaxis(
                input_data_hwc, [0, 1, 2], [1, 0, 2]).astype(np.float32) -
                128.0).astype(np.float16)

            pose_net.put_input(
                self.net, np.ascontiguousarray(input_data_whc16))
            dt_pre = time.time() - t0pre
            self.info("Preprocessed input in %.3f sec", dt_pre)

            t0 = time.time()
            pose_net.run_network(self.net)
            dt_inference = time.time() - t0
            self.info("Inference in %.3f sec", dt_inference)
            output_whc8 = pose_net.get_final_output(self.net)

            t0pp = time.time()
            t0 = t0pp
            extract_heatmap_peaks(
                array_address(output_whc8), array_address(peaks_chw),
                self.peaks_threshold)
            self.info("heatmap peaks extracted in %.3f sec", time.time() - t0)

            t0 = time.time()
            peaks_hwc = np.moveaxis(peaks_chw, [0, 1, 2], [2, 0, 1])
            data56 = output_whc8[:out_width * out_height * 56].\
                reshape(56 // 8, out_width, out_height, 8)
            for ch in range(0, 18):
                heatmap_hwc[:, :, ch] = np.moveaxis(
                    data56[ch // 8, :, :, ch % 8], [0, 1], [1, 0])
            for ch in range(19, 56):
                paf_hwc[:, :, ch - 19] = np.moveaxis(
                    data56[ch // 8, :, :, ch % 8], [0, 1], [1, 0])
            self.info("Repacked numpy arrays in %.3f sec", time.time() - t0)

            t0 = time.time()
            humans = estimate_paf(peaks_hwc, heatmap_hwc, paf_hwc)
            self.info("Extracted %d humans in %.3f sec",
                      len(humans), time.time() - t0)
            dt_pp = time.time() - t0pp

            t0 = time.time()
            draw_humans(image_resized_bgr, humans, imgcopy=False)
            self.info("Drawn humans in %.3f sec", time.time() - t0)

            t0 = time.time()
            frame[:] = 0
            x0 = (frame.shape[1] - image_resized_bgr.shape[1]) // 2
            y0 = (frame.shape[0] - image_resized_bgr.shape[0]) // 4
            frame[y0:y0 + image_resized_bgr.shape[0],
                  x0:x0 + image_resized_bgr.shape[1]] = image_resized_bgr

            text_color = (16, 147, 222)  # bgr
            text_scale = 1.0
            text_thickness = 2
            text = "CONV: %.0f msec" % (
                0.001 * pose_net.get_conv_usec(self.net))
            sz, b = cv2.getTextSize(text, FONT, text_scale, text_thickness)
            h = sz[1] + sz[1] // 2
            left = x0
            top = y0 + image_resized_bgr.shape[0] + h
            cv2.putText(frame, text, (left, top), FONT, text_scale,
                        text_color, text_thickness)
            top += h

            text = "CONV+CPU+MemSync: %.0f msec" % (1000.0 * dt_inference)
            sz, b = cv2.getTextSize(text, FONT, text_scale, text_thickness)
            cv2.putText(frame, text, (left, top), FONT, text_scale,
                        text_color, text_thickness)
            top += h

            text = "Post-processing: %.0f msec" % (1000.0 * dt_pp)
            sz, b = cv2.getTextSize(text, FONT, text_scale, text_thickness)
            cv2.putText(frame, text, (left, top), FONT, text_scale,
                        text_color, text_thickness)
            top += h

            text = "Pre-processing: %.0f msec" % (1000.0 * dt_pre)
            sz, b = cv2.getTextSize(text, FONT, text_scale, text_thickness)
            cv2.putText(frame, text, (left, top), FONT, text_scale,
                        text_color, text_thickness)
            top += h

            text = "Total: %.0f msec" % (1000.0 * (time.time() - t00))
            sz, b = cv2.getTextSize(text, FONT, text_scale, text_thickness)
            cv2.putText(frame, text, (left, top), FONT, text_scale,
                        text_color, text_thickness)

            self.info("Filled framebuffer in %.3f sec", time.time() - t0)

            self.info("Total time: %.3f", time.time() - t00)

            self.fb.update(frame)

            paused = False
            while True:
                c = self.stdscr.getch()
                if c == -1:
                    if paused:
                        continue
                    break
                if c == ord(' '):
                    if paused:
                        paused = False
                        self.info("Unpaused")
                        continue
                    paused = True
                    self.info("Paused")
                    continue
                if c == 27:
                    self.cap.release()
                    return

    def open_video_capture(self):
        self.info("Opening %s...", self.capture_source)
        t0 = time.time()
        self.cap = cv2.VideoCapture(self.capture_source)
        if not self.cap.isOpened():
            self.error("Could not open %s", self.capture_source)
            return False
        self.info("Opened %s in %.3f sec",
                  self.capture_source, time.time() - t0)
        return True


def main(stdscr, capture_source, peaks_threshold, log_fnme):
    with open(log_fnme, "w") as flog:
        obj = Main(stdscr, flog, capture_source, peaks_threshold)
        try:
            obj.execute()
        finally:
            obj.fb.release()
        del obj
    import gc
    gc.collect()


if __name__ == "__main__":
    try:
        logging.info("Executing...")
        curses.wrapper(main, args.INPUT, args.peaks_threshold, args.flog)
    except KeyboardInterrupt:
        logging.info("Ctrl+C pressed, exiting...")
    except Exception as e:
        logging.error("Exception occured: %s", e)
    logging.info("Done, see %s for detailed log", args.flog)
