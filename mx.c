/*
 *  Copyright 2018 Digital Media Professionals Inc.

 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at

 *      http://www.apache.org/licenses/LICENSE-2.0

 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/**
* @file mx.c
* @brief Function for python for finding peaks inside heat map.
*/

#include <stdio.h>


#define OUT_W 54
#define OUT_H 46
#define OUT_C 57


#define N_PARTS 18


static inline int imin(int a, int b) {
  return a < b ? a : b;
}


/// @brief Extract channel plane from packed WHC8 array.
/// @param whc8 Packed WHC8 array.
/// @param c Channel number.
/// @param hw Height-Width packed channel plane.
static int whc8_unpack_channel(const float *whc8, int c, float *hw, float threshold) {
  if ((c < 0) || (c > 56)) {
    fprintf(stderr, "whc8_unpack_channel: Invalid arg: c=%d\n", c);
    return -1;
  }
  const int chan_group = c >> 3;
  const int chan_offs = c & 7;
  const int chan_step = imin(OUT_C - c, 8);
  for (int ix = 0, i_offs = chan_group * (OUT_W * OUT_H * 8) + chan_offs; ix < OUT_W; ++ix) {
    for (int iy = 0, o_offs = ix; iy < OUT_H; ++iy, i_offs += chan_step, o_offs += OUT_W) {
      const float vle = whc8[i_offs];
      hw[o_offs] = vle > threshold ? vle : 0;
    }
  }
  return 0;
}


struct Point {
  short x, y;
};


/// @brief Find peaks inside heat map.
/// @param whc8 w=54 h=46 c=57 in hardware layout.
/// @param peaks_hwc Output heatmap peaks in CHW format, shape=(18, 46, 54).
int extract_heatmap_peaks(const float *whc8, float *peaks_chw, float threshold) {
  static const struct Point shifts[8] = {
    {-1, -1}, {0, -1}, {1, -1},
    {-1, 0}, {1, 0},
    {-1, 1}, {0, 1}, {1, 1}
  };
  const int WS = 512;
  struct Point wavefront[WS];
  int low, high;

  float *hm = peaks_chw;
  for (int i_part = 0; i_part < N_PARTS; ++i_part, hm += OUT_H * OUT_W) {
    if (whc8_unpack_channel(whc8, i_part, hm, threshold)) {
      return -1;
    }

    // Set non-maximums to zero
    for (int iy = 0; iy < OUT_H; ++iy) {
      for (int ix = 0; ix < OUT_W; ++ix) {
        const int start_offs = iy * OUT_W + ix;
        if (!hm[start_offs]) {
          continue;
        }
        low = 0;
        wavefront[low].x = ix;
        wavefront[low].y = iy;
        high = low + 1;
        for (; low != high;) {
          const int cx = wavefront[low].x,
                    cy = wavefront[low].y;
          const int c_offs = cy * OUT_W + cx;
          if (!hm[c_offs]) {
            ++low;
            low &= WS - 1;
            continue;
          }
          const float c_vle = hm[c_offs];
          int k = 0;
          for (int i = 0; i < 8; ++i) {
            const int tx = cx + shifts[i].x,
                      ty = cy + shifts[i].y;
            if ((tx < 0) || (tx >= OUT_W) || (ty < 0) || (ty >= OUT_H)) {
              continue;
            }
            const int t_offs = ty * OUT_W + tx;
            const float t_vle = hm[t_offs];
            if (t_vle < c_vle) {
              hm[t_offs] = 0;
              continue;
            }
            if (!hm[t_offs]) {
              continue;
            }
            wavefront[high].x = tx;
            wavefront[high].y = ty;
            ++high;
            high &= WS - 1;
            if (low == high) {
              fprintf(stderr, "Wavefront array is too small to extract maximums from the heatmap\n");
              return -1;
            }
            ++k;
          }
          if (k) {
            hm[c_offs] = 0;
          }
          ++low;
          low &= WS - 1;
        }
      }
    }
  }
  return 0;
}
