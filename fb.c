/*
 *  Copyright 2018 Digital Media Professionals Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/**
* @file fb.c
* @brief Functions for python for drawing to Linux Framebuffer.
*/

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <linux/videodev2.h>
#include <linux/kd.h>
#include <errno.h>


static int g_fb_file = -1;
static struct fb_fix_screeninfo g_fb_fix_info;
static struct fb_var_screeninfo g_fb_var_info;
static uint32_t g_fb_pixfmt = 0;
static uint8_t *g_fb_mem = NULL;
static uint8_t *g_frame_ptr = NULL;


#define SCREEN_W (g_fb_var_info.xres)
#define SCREEN_H (g_fb_var_info.yres)


#define ERR(...) fprintf(stderr, __VA_ARGS__); fflush(stderr)


static void update_frame_ptr() {
  if (g_fb_var_info.yoffset) {
    g_frame_ptr = g_fb_mem;
  }
  else {
    g_frame_ptr = g_fb_mem + SCREEN_H * SCREEN_W * (g_fb_var_info.bits_per_pixel >> 3);
  }
}


static int set_pan(uint32_t pan_x, uint32_t pan_y) {
  struct fb_var_screeninfo var_info;
  memset(&var_info, 0, sizeof var_info);
  var_info.xoffset = pan_x;
  var_info.yoffset = pan_y;

  int res = ioctl(g_fb_file, FBIOPAN_DISPLAY, &var_info);
  if (res < 0) {
    ERR("ioctl(FBIOPAN_DISPLAY) failed for /dev/fb0\n");
    return -1;
  }

  g_fb_var_info.xoffset = var_info.xoffset;
  g_fb_var_info.yoffset = var_info.yoffset;

  update_frame_ptr();

  return 0;
}


uint8_t* get_frame_ptr() {
  return g_frame_ptr;
}


int get_screen_width() {
  return g_fb_var_info.xres;
}


int get_screen_height() {
  return g_fb_var_info.yres;
}


int copy_to_frame(const void *src, int size) {
  uint8_t *dst = get_frame_ptr();
  if (!dst) {
    ERR("copy_to_frame() failed: destination is NULL (framebuffer not initialized)\n");
    return -1;
  }
  const int n = get_screen_width() * get_screen_height() * 3;
  if (n != size) {
    ERR("copy_to_frame() failed: size=%d while %d is required\n", size, n);
    return -1;
  }
  memcpy(dst, src, size);
  return 0;
}


void release_fb() {
  if (g_fb_mem) {
    munmap(g_fb_mem, g_fb_fix_info.smem_len);
    g_fb_mem = NULL;
  }
  if (g_fb_file != -1) {
    close(g_fb_file);
    g_fb_file = -1;
  }
}


int init_fb() {
  if (g_fb_file != -1) {
    ERR("Framebuffer is already opened\n");
    return -1;
  }
  g_fb_file = open("/dev/fb0", O_RDWR | O_CLOEXEC);
  if (g_fb_file == -1) {
    ERR("open() failed for /dev/fb0\n");
    return -1;
  }

  int res = ioctl(g_fb_file, FBIOGET_FSCREENINFO, &g_fb_fix_info);
  if (res < 0) {
    ERR("ioctl(FBIOGET_FSCREENINFO) failed for /dev/fb0\n");
    release_fb();
    return -1;
  }
  res = ioctl(g_fb_file, FBIOGET_VSCREENINFO, &g_fb_var_info);
  if (res < 0) {
    ERR("ioctl(FBIOGET_FSCREENINFO) failed for /dev/fb0\n");
    release_fb();
    return -1;
  }
  if ((!g_fb_var_info.xres) || (!g_fb_var_info.yres)) {
    ERR("Could not determine framebuffer dimensions: xres=%u yres=%u\n",
        g_fb_var_info.xres, g_fb_var_info.yres);
    release_fb();
    return -1;
  }

  switch (g_fb_var_info.bits_per_pixel) {
    case 16:
      g_fb_pixfmt = V4L2_PIX_FMT_RGB565;
      break;
    case 24:
      if (g_fb_var_info.red.offset == 0) {
        g_fb_pixfmt = V4L2_PIX_FMT_RGB24;
      }
      else {
        g_fb_pixfmt = V4L2_PIX_FMT_BGR24;
      }
      break;
     case 32:
      if (g_fb_var_info.red.offset == 0) {
        g_fb_pixfmt = V4L2_PIX_FMT_RGB32;
      }
      else {
        g_fb_pixfmt = V4L2_PIX_FMT_BGR32;
      }
      break;
      default: {
        g_fb_pixfmt = 0;
      }
  }
  if (g_fb_pixfmt != V4L2_PIX_FMT_BGR24) {
    ERR("Unsupported pixel format: bpp=%d red.offset=%d\n",
        g_fb_var_info.bits_per_pixel, g_fb_var_info.red.offset);
    release_fb();
    return -1;
  }

  if (g_fb_fix_info.smem_len < g_fb_var_info.xres * g_fb_var_info.yres * (g_fb_var_info.bits_per_pixel >> 2)) {
    ERR("Framebuffer doesn't support double buffering\n");
    release_fb();
    return -1;
  }

  if (g_fb_fix_info.line_length != g_fb_var_info.xres * (g_fb_var_info.bits_per_pixel >> 3)) {
    ERR("Support for framebuffer with bigger than visible width %d line length %d is not implemented\n",
        g_fb_var_info.xres * (g_fb_var_info.bits_per_pixel >> 3), g_fb_fix_info.line_length);
    release_fb();
    return -1;
  }

  g_fb_mem = (uint8_t*)mmap(
      NULL, g_fb_fix_info.smem_len, PROT_READ | PROT_WRITE,
      MAP_SHARED, g_fb_file, 0);
  if (!g_fb_mem) {
    ERR("mmap() failed for /dev/fb0\n");
    release_fb();
    return -1;
  }

  // reset screen pan
  set_pan(g_fb_var_info.xoffset, 0);

  return 0;
}


int get_console_mode() {
  long mode = 0;
  int console = open("/dev/tty0", O_RDWR | O_CLOEXEC);
  if (console == -1) {
    ERR("open() failed for /dev/tty0: errno=%d: %s\n", errno, strerror(errno));
    return -1;
  }
  if (ioctl(console, KDGETMODE, &mode) < 0) {
    ERR("Could not determine console mode (text or graphics): errno=%d: %s\n", errno, strerror(errno));
    close(console);
    return -1;
  }
  close(console);
  return mode == KD_GRAPHICS ? 1 : 0;
}


int set_console_mode(int graphics) {
  int console = open("/dev/tty0", O_RDWR | O_CLOEXEC);
  if (console == -1) {
    ERR("open() failed for /dev/tty0: errno=%d: %s\n", errno, strerror(errno));
    return -1;
  }
  if (ioctl(console, KDSETMODE, graphics ? KD_GRAPHICS : KD_TEXT) < 0) {
    ERR("Could not change console to %s mode: errno=%d: %s\n", graphics ? "graphics" : "text", errno, strerror(errno));
    close(console);
    return -1;
  }
  close(console);
  return 0;
}


int swap_buffer() {
  int res = ioctl(g_fb_file, FBIOBLANK, FB_BLANK_UNBLANK);
  if (res < 0) {
    ERR("ioctl(FB_BLANK_UNBLANK) failed for /dev/fb0\n");
    return -1;
  }

  unsigned int screen = 0;
  res = ioctl(g_fb_file, FBIO_WAITFORVSYNC, &screen);
  if (res < 0) {
    ERR("ioctl(FBIO_WAITFORVSYNC) failed for /dev/fb0\n");
    return -1;
  }

  return set_pan(g_fb_var_info.xoffset, g_fb_var_info.yoffset ? 0 : g_fb_var_info.yres);
}
