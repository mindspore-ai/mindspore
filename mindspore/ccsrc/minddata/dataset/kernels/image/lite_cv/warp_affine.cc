/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <climits>
#include <cmath>
#include <vector>

#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"

constexpr int kBits = 5;
constexpr int kBits1 = 15;
constexpr int kTabSz = 1 << kBits;
constexpr int kTabSz2 = kTabSz * kTabSz;
constexpr int kRemapScale = 1 << 15;

namespace mindspore {
namespace dataset {
static int16_t BWBlock_i[kTabSz2][2][2];

static double SrcValue(const double *src, const int &y, const int &x) { return (src + y * 3)[x]; }

static double &DstValue(double *dst, const int &y, const int &x) { return (dst + y * 3)[x]; }

static double GetDet3(double *src) {
  double a1 =
    SrcValue(src, 0, 0) * (SrcValue(src, 1, 1) * SrcValue(src, 2, 2) - SrcValue(src, 1, 2) * SrcValue(src, 2, 1));
  double a2 =
    SrcValue(src, 0, 1) * (SrcValue(src, 1, 0) * SrcValue(src, 2, 2) - SrcValue(src, 1, 2) * SrcValue(src, 2, 0));
  double a3 =
    SrcValue(src, 0, 2) * (SrcValue(src, 1, 0) * SrcValue(src, 2, 1) - SrcValue(src, 1, 1) * SrcValue(src, 2, 0));
  return a1 - a2 + a3;
}

static uint8_t UIntToUChar(const uint32_t &v) {
  return static_cast<uint8_t>(v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}

static int16_t IntCastShort(const int &value) {
  return static_cast<int16_t>(static_cast<unsigned>(value - SHRT_MIN) <= static_cast<unsigned>(USHRT_MAX)
                                ? value
                                : value > 0 ? SHRT_MAX : SHRT_MIN);
}

static int16_t FloatToShort(float value) { return IntCastShort(round(value)); }

static void InitWBlockInter(float *wBlock, int wBlockSz) {
  float scale = 1.f / wBlockSz;
  for (int i = 0; i < wBlockSz; i++, wBlock += 2) {
    float value = (i * scale);
    wBlock[0] = 1.f - value;
    wBlock[1] = value;
  }
}

static const void *InitWBlock() {
  static bool initWB = false;
  int16_t *iWBlock = nullptr;
  int ks = 2;

  iWBlock = BWBlock_i[0][0];

  if (!initWB) {
    float *_wblock = new float[8 * kTabSz];
    int i, j, h1, h2;
    InitWBlockInter(_wblock, kTabSz);
    for (i = 0; i < kTabSz; i++) {
      for (j = 0; j < kTabSz; j++, iWBlock += ks * ks) {
        int sum_i = 0;
        for (h1 = 0; h1 < ks; h1++) {
          float vy = _wblock[i * ks + h1];
          for (h2 = 0; h2 < ks; h2++) {
            float v = vy * _wblock[j * ks + h2];
            sum_i += iWBlock[h1 * ks + h2] = FloatToShort(v * kRemapScale);
          }
        }
        if (sum_i != kRemapScale) {
          int df = sum_i - kRemapScale;
          int ks2 = 1;
          int tk1 = ks2;
          int tk2 = ks2;
          int mtk1 = ks2;
          int mtk2 = ks2;
          for (h1 = ks2; h1 < ks2 + 2; h1++) {
            for (h2 = ks2; h2 < ks2 + 2; h2++) {
              if (iWBlock[h1 * ks + h2] < iWBlock[mtk1 * ks + mtk2]) {
                mtk1 = h1, mtk2 = h2;
              } else if (iWBlock[h1 * ks + h2] > iWBlock[tk1 * ks + tk2]) {
                tk1 = h1, tk2 = h2;
              }
            }
          }
          if (df < 0) {
            iWBlock[tk1 * ks + tk2] = (int16_t)(iWBlock[tk1 * ks + tk2] - df);
          } else {
            iWBlock[mtk1 * ks + mtk2] = (int16_t)(iWBlock[mtk1 * ks + mtk2] - df);
          }
        }
      }
    }
    iWBlock -= kTabSz2 * ks * ks;
    delete[] _wblock;
    initWB = true;
  }
  return (const void *)iWBlock;
}

static uint8_t CastToFixed(int v) { return UIntToUChar(((v + (1 << (kBits1 - 1))) >> kBits1)); }

static int BorderPolate(int value, int length, PaddBorderType borderType) {
  if ((unsigned)value < (unsigned)length) {
    return value;
  } else if (borderType == 0) {
    value = -1;
  }
  return value;
}

static void RemapBilinearNotCur1C(int dx, const int16_t *HW, const uint16_t *FHW, const int16_t *wblock,
                                  size_t src_step, const uint8_t *src_ptr, uint8_t *dst_ptr) {
  int shx = HW[dx * 2];
  int shy = HW[dx * 2 + 1];
  const int16_t *w_ptr = wblock + FHW[dx] * 4;
  const uint8_t *t_src_ptr = src_ptr + shy * src_step + shx;
  *dst_ptr = CastToFixed(static_cast<int>(t_src_ptr[0] * w_ptr[0] + t_src_ptr[1] * w_ptr[1] +
                                          t_src_ptr[src_step] * w_ptr[2] + t_src_ptr[src_step + 1] * w_ptr[3]));
}

static void RemapBilinearNotCur2C(int dx, const int16_t *HW, const uint16_t *FHW, const int16_t *wblock,
                                  size_t src_step, const uint8_t *src_ptr, uint8_t *dst_ptr) {
  int shx = HW[dx * 2];
  int shy = HW[dx * 2 + 1];
  const int16_t *w_ptr = wblock + FHW[dx] * 4;
  const uint8_t *t_src_ptr = src_ptr + shy * src_step + shx * 2;
  int v0 = t_src_ptr[0] * w_ptr[0] + t_src_ptr[2] * w_ptr[1] + t_src_ptr[src_step] * w_ptr[2] +
           t_src_ptr[src_step + 2] * w_ptr[3];
  int v1 = t_src_ptr[1] * w_ptr[0] + t_src_ptr[3] * w_ptr[1] + t_src_ptr[src_step + 1] * w_ptr[2] +
           t_src_ptr[src_step + 3] * w_ptr[3];
  dst_ptr[0] = CastToFixed(v0);
  dst_ptr[1] = CastToFixed(v1);
}

static void RemapBilinearNotCur3C(int dx, const int16_t *HW, const uint16_t *FHW, const int16_t *wblock,
                                  size_t src_step, const uint8_t *src_ptr, uint8_t *dst_ptr) {
  int shx = HW[dx * 2];
  int shy = HW[dx * 2 + 1];
  const int16_t *w_ptr = wblock + FHW[dx] * 4;
  const uint8_t *t_src_ptr = src_ptr + shy * src_step + shx * 3;
  int v0 = t_src_ptr[0] * w_ptr[0] + t_src_ptr[3] * w_ptr[1] + t_src_ptr[src_step] * w_ptr[2] +
           t_src_ptr[src_step + 3] * w_ptr[3];
  int v1 = t_src_ptr[1] * w_ptr[0] + t_src_ptr[4] * w_ptr[1] + t_src_ptr[src_step + 1] * w_ptr[2] +
           t_src_ptr[src_step + 4] * w_ptr[3];
  int v2 = t_src_ptr[2] * w_ptr[0] + t_src_ptr[5] * w_ptr[1] + t_src_ptr[src_step + 2] * w_ptr[2] +
           t_src_ptr[src_step + 5] * w_ptr[3];
  dst_ptr[0] = CastToFixed(v0);
  dst_ptr[1] = CastToFixed(v1);
  dst_ptr[2] = CastToFixed(v2);
}

static void RemapBilinearNotCur4C(int dx, const int16_t *HW, const uint16_t *FHW, const int16_t *wblock,
                                  size_t src_step, const uint8_t *src_ptr, uint8_t *dst_ptr) {
  int shx = HW[dx * 2];
  int shy = HW[dx * 2 + 1];
  const int16_t *w_ptr = wblock + FHW[dx] * 4;
  const uint8_t *t_src_ptr = src_ptr + shy * src_step + shx * 4;
  int v0 = t_src_ptr[0] * w_ptr[0] + t_src_ptr[4] * w_ptr[1] + t_src_ptr[src_step] * w_ptr[2] +
           t_src_ptr[src_step + 4] * w_ptr[3];
  int v1 = t_src_ptr[1] * w_ptr[0] + t_src_ptr[5] * w_ptr[1] + t_src_ptr[src_step + 1] * w_ptr[2] +
           t_src_ptr[src_step + 5] * w_ptr[3];
  dst_ptr[0] = CastToFixed(v0);
  dst_ptr[1] = CastToFixed(v1);
  v0 = t_src_ptr[2] * w_ptr[0] + t_src_ptr[6] * w_ptr[1] + t_src_ptr[src_step + 2] * w_ptr[2] +
       t_src_ptr[src_step + 6] * w_ptr[3];
  v1 = t_src_ptr[3] * w_ptr[0] + t_src_ptr[7] * w_ptr[1] + t_src_ptr[src_step + 3] * w_ptr[2] +
       t_src_ptr[src_step + 7] * w_ptr[3];
  dst_ptr[2] = CastToFixed(v0);
  dst_ptr[3] = CastToFixed(v1);
}

static void RemapBilinearNotCurMoreC(int dx, const int16_t *HW, const uint16_t *FHW, const int16_t *wblock,
                                     size_t src_step, int cn, const uint8_t *src_ptr, uint8_t *dst_ptr) {
  int shx = HW[dx * 2];
  int shy = HW[dx * 2 + 1];
  const int16_t *w_ptr = wblock + FHW[dx] * 4;
  const uint8_t *t_src_ptr = src_ptr + shy * src_step + shx * cn;
  for (int k = 0; k < cn; k++) {
    int v0 = t_src_ptr[k] * w_ptr[0] + t_src_ptr[k + cn] * w_ptr[1] + t_src_ptr[src_step + k] * w_ptr[2] +
             t_src_ptr[src_step + k + cn] * w_ptr[3];
    dst_ptr[k] = CastToFixed(v0);
  }
}

static void RemapBilinearCur1C(LiteMat _src, int dx, const int16_t *HW, const uint16_t *FHW, const int16_t *wblock,
                               size_t src_step, const uint8_t *src_ptr, uint8_t *dst_ptr, PaddBorderType borderType,
                               const std::vector<uint8_t> &borderValue) {
  if (borderValue.size() == 0) {
    return;
  }
  int shx = HW[dx * 2];
  int shy = HW[dx * 2 + 1];
  if (borderType == PADD_BORDER_CONSTANT && (shx >= _src.width_ || shx + 1 < 0 || shy >= _src.height_ || shy + 1 < 0)) {
    dst_ptr[0] = borderValue[0];
  } else {
    int sv0;
    int sv1;
    int su0;
    int su1;
    const int16_t *w_ptr = wblock + FHW[dx] * 4;

    sv0 = BorderPolate(shx, _src.width_, borderType);
    sv1 = BorderPolate(shx + 1, _src.width_, borderType);
    su0 = BorderPolate(shy, _src.height_, borderType);
    su1 = BorderPolate(shy + 1, _src.height_, borderType);
    uint8_t v0 = sv0 >= 0 && su0 >= 0 ? src_ptr[su0 * src_step + sv0] : borderValue[0];
    uint8_t v1 = sv1 >= 0 && su0 >= 0 ? src_ptr[su0 * src_step + sv1] : borderValue[0];
    uint8_t v2 = sv0 >= 0 && su1 >= 0 ? src_ptr[su1 * src_step + sv0] : borderValue[0];
    uint8_t v3 = sv1 >= 0 && su1 >= 0 ? src_ptr[su1 * src_step + sv1] : borderValue[0];
    dst_ptr[0] = CastToFixed(static_cast<int>(v0 * w_ptr[0] + v1 * w_ptr[1] + v2 * w_ptr[2] + v3 * w_ptr[3]));
  }
}

static void RemapBilinearCurMoreC(LiteMat _src, int dx, const int16_t *HW, const uint16_t *FHW, const int16_t *wblock,
                                  size_t src_step, int cn, const uint8_t *src_ptr, uint8_t *dst_ptr,
                                  PaddBorderType borderType, const std::vector<uint8_t> &borderValue) {
  int shx = HW[dx * 2];
  int shy = HW[dx * 2 + 1];
  if (borderValue.size() < cn || borderValue.size() == 0) {
    return;
  }
  if (borderType == PADD_BORDER_CONSTANT && (shx >= _src.width_ || shx + 1 < 0 || shy >= _src.height_ || shy + 1 < 0)) {
    for (int k = 0; k < cn; k++) {
      dst_ptr[k] = borderValue[k];
    }
  } else {
    int sv0;
    int sv1;
    int su0;
    int su1;
    const int16_t *w_ptr = wblock + FHW[dx] * 4;
    sv0 = BorderPolate(shx, _src.width_, borderType);
    sv1 = BorderPolate(shx + 1, _src.width_, borderType);
    su0 = BorderPolate(shy, _src.height_, borderType);
    su1 = BorderPolate(shy + 1, _src.height_, borderType);
    const uint8_t *v0 = sv0 >= 0 && su0 >= 0 ? src_ptr + su0 * src_step + sv0 * cn : &borderValue[0];
    const uint8_t *v1 = sv1 >= 0 && su0 >= 0 ? src_ptr + su0 * src_step + sv1 * cn : &borderValue[0];
    const uint8_t *v2 = sv0 >= 0 && su1 >= 0 ? src_ptr + su1 * src_step + sv0 * cn : &borderValue[0];
    const uint8_t *v3 = sv1 >= 0 && su1 >= 0 ? src_ptr + su1 * src_step + sv1 * cn : &borderValue[0];

    for (int k = 0; k < cn; k++) {
      dst_ptr[k] =
        CastToFixed(static_cast<int>(v0[k] * w_ptr[0] + v1[k] * w_ptr[1] + v2[k] * w_ptr[2] + v3[k] * w_ptr[3]));
    }
  }
}

static void RemapBilinear(const LiteMat &_src, LiteMat &_dst, const LiteMat &_hw, const LiteMat &_fhw,  // NOLINT
                          const void *_wblock, const PaddBorderType borderType,
                          const std::vector<uint8_t> &borderValue) {
  const int cn = _src.channel_;
  const int16_t *wblock = (const int16_t *)_wblock;
  const uint8_t *src_ptr = _src.ptr<uint8_t>(0);
  size_t src_step = _src.steps_[0];
  unsigned src_width = std::max(_src.width_ - 1, 0);
  unsigned src_height = std::max(_src.height_ - 1, 0);

  for (int dy = 0; dy < _dst.height_; dy++) {
    uint8_t *dst_ptr = _dst.ptr<uint8_t>(dy);
    const int16_t *HW = _hw.ptr<int16_t>(dy);
    const uint16_t *FHW = _fhw.ptr<uint16_t>(dy);
    int tt = 0;
    bool prevLine = false;

    for (int dx = 0; dx <= _dst.width_; dx++) {
      bool curLine =
        dx < _dst.width_ ? (unsigned)HW[dx * 2] < src_width && (unsigned)HW[dx * 2 + 1] < src_height : !prevLine;
      if (curLine == prevLine) {
        continue;
      }

      int H1 = dx;
      dx = tt;
      tt = H1;
      prevLine = curLine;

      if (!curLine) {
        int length = 0;
        dst_ptr += length * cn;
        dx += length;

        if (cn == 1) {
          for (; dx < H1; dx++, dst_ptr++) {
            RemapBilinearNotCur1C(dx, HW, FHW, wblock, src_step, src_ptr, dst_ptr);
          }
        } else if (cn == 2) {
          for (; dx < H1; dx++, dst_ptr += 2) {
            RemapBilinearNotCur2C(dx, HW, FHW, wblock, src_step, src_ptr, dst_ptr);
          }
        } else if (cn == 3) {
          for (; dx < H1; dx++, dst_ptr += 3) {
            RemapBilinearNotCur3C(dx, HW, FHW, wblock, src_step, src_ptr, dst_ptr);
          }
        } else if (cn == 4) {
          for (; dx < H1; dx++, dst_ptr += 4) {
            RemapBilinearNotCur4C(dx, HW, FHW, wblock, src_step, src_ptr, dst_ptr);
          }
        } else {
          for (; dx < H1; dx++, dst_ptr += cn) {
            RemapBilinearNotCurMoreC(dx, HW, FHW, wblock, src_step, cn, src_ptr, dst_ptr);
          }
        }
      } else {
        if (cn == 1) {
          for (; dx < H1; dx++, dst_ptr++) {
            RemapBilinearCur1C(_src, dx, HW, FHW, wblock, src_step, src_ptr, dst_ptr, borderType, borderValue);
          }
        } else {
          for (; dx < H1; dx++, dst_ptr += cn) {
            RemapBilinearCurMoreC(_src, dx, HW, FHW, wblock, src_step, cn, src_ptr, dst_ptr, borderType, borderValue);
          }
        }
      }
    }
  }
}

static void Remap(const LiteMat &src, LiteMat &dst, LiteMat &map1, const LiteMat &map2,  // NOLINT
                  const PaddBorderType borderType, const std::vector<uint8_t> &borderValue) {
  int x, y, x1, y1;
  const int buf_size = 1 << 14;
  int dst_height = std::min(128, dst.height_);
  int dst_width = std::min(buf_size / dst_height, dst.width_);
  dst_height = std::min(buf_size / dst_width, dst.height_);

  const void *wblock = InitWBlock();

  LiteMat _xyblock(dst_width, dst_height, 2, LDataType::INT16);
  LiteMat _ablock(dst_width, dst_height, 1, LDataType::UINT16);
  for (y = 0; y < dst.height_; y += dst_height) {
    for (x = 0; x < dst.width_; x += dst_width) {
      int bheight = std::min(dst_height, dst.height_ - y);
      int bwidth = std::min(dst_width, dst.width_ - x);
      LiteMat lite_part;
      dst.GetROI(x, y, bwidth, bheight, lite_part);

      LiteMat xy_ptr;
      _xyblock.GetROI(0, 0, bwidth, bheight, xy_ptr);
      LiteMat a_ptr;
      _ablock.GetROI(0, 0, bwidth, bheight, a_ptr);

      for (y1 = 0; y1 < bheight; y1++) {
        uint16_t *t_a_ptr = a_ptr.ptr<uint16_t>(y1);

        map1.GetROI(x, y, bwidth, bheight, xy_ptr);

        const uint16_t *sa_ptr = map2.ptr<uint16_t>(y + y1) + x;
        x1 = 0;
        for (; x1 < bwidth; x1++) {
          t_a_ptr[x1] = (uint16_t)(sa_ptr[x1] & (kTabSz2 - 1));
        }
      }
      RemapBilinear(src, lite_part, xy_ptr, a_ptr, wblock, borderType, borderValue);
    }
  }
}

bool WarpAffineBilinear(const LiteMat &src, LiteMat &dst, const LiteMat &M, int dst_w, int dst_h,  // NOLINT
                        PaddBorderType borderType, std::vector<uint8_t> &borderValue) {            // NOLINT
  if (dst_w <= 0 || dst_h <= 0) {
    return false;
  }

  if (!(M.height_ == 2 && M.width_ == 3)) {
    return false;
  }
  if (borderType != PADD_BORDER_CONSTANT) {
    return false;
  }
  if (borderValue.size() != src.channel_) {
    return false;
  }
  if (dst.IsEmpty()) {
    (void)dst.Init(dst_w, dst_h, src.channel_, LDataType::UINT8);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  } else if (dst.height_ != dst_h || dst.width_ != dst_w || dst.channel_ != src.channel_ ||
             dst.data_type_ != LDataType::UINT8) {
    return false;
  }

  double IM[6];
  const double *M_Ptr = M;
  for (int i = 0; i < 6; i++) {
    IM[i] = M_Ptr[i];
  }

  double D = IM[0] * IM[4] - IM[1] * IM[3];
  D = std::fabs(D) > std::numeric_limits<double>::epsilon() ? 1.0f / D : 0;
  double A11 = IM[4] * D, A22 = IM[0] * D;
  IM[0] = A11;
  IM[1] *= -D;
  IM[3] *= -D;
  IM[4] = A22;
  double b1 = -IM[0] * IM[2] - IM[1] * IM[5];
  double b2 = -IM[3] * IM[2] - IM[4] * IM[5];
  IM[2] = b1;
  IM[5] = b2;

  int *_a = new int[dst.width_ * 2];
  int *a = &_a[0], *b = a + dst.width_;
  const int SCALE = 1 << 10;
  const int B_SIZE = 64;
  int16_t *WH = new int16_t[B_SIZE * B_SIZE * 2];
  int16_t A_Ptr[B_SIZE * B_SIZE];
  int r_delta = SCALE / kTabSz / 2;
  int x, y, x1, y1;
  for (x = 0; x < dst.width_; x++) {
    a[x] = round(IM[0] * x * SCALE);
    b[x] = round(IM[3] * x * SCALE);
  }
  int t_bh0 = std::min(B_SIZE / 2, dst.height_);
  int t_bw0 = std::min(B_SIZE * B_SIZE / t_bh0, dst.width_);
  t_bh0 = std::min(B_SIZE * B_SIZE / t_bw0, dst.height_);

  for (y = 0; y < dst.height_; y += t_bh0) {
    for (x = 0; x < dst.width_; x += t_bw0) {
      int t_bw = std::min(t_bw0, dst.width_ - x);
      int t_bh = std::min(t_bh0, dst.height_ - y);
      LiteMat _HW(t_bw, t_bh, 2, WH, LDataType::INT16);
      LiteMat lite_part;
      dst.GetROI(x, y, t_bw, t_bh, lite_part);

      for (y1 = 0; y1 < t_bh; y1++) {
        int16_t *t_xy = WH + y1 * t_bw * 2;
        int X0 = round((IM[1] * (y + y1) + IM[2]) * SCALE) + r_delta;
        int Y0 = round((IM[4] * (y + y1) + IM[5]) * SCALE) + r_delta;
        int16_t *t_a = A_Ptr + y1 * t_bw;
        x1 = 0;
        for (; x1 < t_bw; x1++) {
          int X = (X0 + a[x + x1]) >> (10 - kBits);
          int Y = (Y0 + b[x + x1]) >> (10 - kBits);
          t_xy[x1 * 2] = IntCastShort(X >> kBits);
          t_xy[x1 * 2 + 1] = IntCastShort(Y >> kBits);
          t_a[x1] = (int16_t)((Y & (kTabSz - 1)) * kTabSz + (X & (kTabSz - 1)));
        }
      }

      LiteMat _matA(t_bw, t_bh, 1, A_Ptr, LDataType::UINT16);
      Remap(src, lite_part, _HW, _matA, borderType, borderValue);
    }
  }
  delete[] WH;
  delete[] _a;
  return true;
}

static void PerspectiveInvert(double *src, double *dst) {
  double value = GetDet3(src);
  if (std::fabs(value) > std::numeric_limits<double>::epsilon()) {
    value = 1. / value;
    double v[9];

    v[0] = (SrcValue(src, 1, 1) * SrcValue(src, 2, 2) - SrcValue(src, 1, 2) * SrcValue(src, 2, 1)) * value;
    v[1] = (SrcValue(src, 0, 2) * SrcValue(src, 2, 1) - SrcValue(src, 0, 1) * SrcValue(src, 2, 2)) * value;
    v[2] = (SrcValue(src, 0, 1) * SrcValue(src, 1, 2) - SrcValue(src, 0, 2) * SrcValue(src, 1, 1)) * value;

    v[3] = (SrcValue(src, 1, 2) * SrcValue(src, 2, 0) - SrcValue(src, 1, 0) * SrcValue(src, 2, 2)) * value;
    v[4] = (SrcValue(src, 0, 0) * SrcValue(src, 2, 2) - SrcValue(src, 0, 2) * SrcValue(src, 2, 0)) * value;
    v[5] = (SrcValue(src, 0, 2) * SrcValue(src, 1, 0) - SrcValue(src, 0, 0) * SrcValue(src, 1, 2)) * value;

    v[6] = (SrcValue(src, 1, 0) * SrcValue(src, 2, 1) - SrcValue(src, 1, 1) * SrcValue(src, 2, 0)) * value;
    v[7] = (SrcValue(src, 0, 1) * SrcValue(src, 2, 0) - SrcValue(src, 0, 0) * SrcValue(src, 2, 1)) * value;
    v[8] = (SrcValue(src, 0, 0) * SrcValue(src, 1, 1) - SrcValue(src, 0, 1) * SrcValue(src, 1, 0)) * value;

    DstValue(dst, 0, 0) = v[0];
    DstValue(dst, 0, 1) = v[1];
    DstValue(dst, 0, 2) = v[2];
    DstValue(dst, 1, 0) = v[3];
    DstValue(dst, 1, 1) = v[4];
    DstValue(dst, 1, 2) = v[5];
    DstValue(dst, 2, 0) = v[6];
    DstValue(dst, 2, 1) = v[7];
    DstValue(dst, 2, 2) = v[8];
  }
}

bool WarpPerspectiveBilinear(const LiteMat &src, LiteMat &dst, const LiteMat &M, int dst_w, int dst_h,  // NOLINT
                             PaddBorderType borderType, std::vector<uint8_t> &borderValue) {            // NOLINT
  if (dst_w <= 0 || dst_h <= 0) {
    return false;
  }
  if (!(M.height_ == 3 && M.width_ == 3)) {
    return false;
  }
  if (borderType != PADD_BORDER_CONSTANT) {
    return false;
  }
  if (borderValue.size() != src.channel_) {
    return false;
  }
  if (dst.IsEmpty()) {
    (void)dst.Init(dst_w, dst_h, src.channel_, LDataType::UINT8);
    RETURN_FALSE_IF_LITEMAT_EMPTY(dst);
  } else if (dst.height_ != dst_h || dst.width_ != dst_w || dst.channel_ != src.channel_) {
    return false;
  } else if (dst.data_type_ != LDataType::UINT8) {
    return false;
  } else {
  }

  double IM[9];
  const double *M_Ptr = M;
  for (int i = 0; i < 9; i++) {
    IM[i] = M_Ptr[i];
  }
  PerspectiveInvert(IM, IM);
  const int B_SZ = 32;
  int16_t HW[B_SZ * B_SZ * 2];
  int16_t TA[B_SZ * B_SZ];
  int x;
  int y;
  int y1;
  int width = dst.width_;
  int height = dst.height_;

  int bheight = std::min(B_SZ / 2, height);
  int bwidth = std::min(B_SZ * B_SZ / bheight, width);
  bheight = std::min(B_SZ * B_SZ / bwidth, height);
  for (y = 0; y < dst.height_; y += bheight) {
    for (x = 0; x < width; x += bwidth) {
      int tw = std::min(bwidth, width - x);
      int th = std::min(bheight, dst.height_ - y);

      LiteMat _HW(tw, th, 2, HW, LDataType::INT16);
      LiteMat lite_part;
      dst.GetROI(x, y, tw, th, lite_part);

      for (y1 = 0; y1 < th; y1++) {
        int16_t *xy = HW + y1 * tw * 2;
        double XV = IM[0] * x + IM[1] * (y + y1) + IM[2];
        double YV = IM[3] * x + IM[4] * (y + y1) + IM[5];
        double WV = IM[6] * x + IM[7] * (y + y1) + IM[8];

        int16_t *t_a = TA + y1 * tw;
        for (int x1 = 0; x1 < tw; x1++) {
          double W = WV + IM[6] * x1;
          W = W ? kTabSz / W : 0;
          double fX = std::max((double)INT_MIN, std::min((double)INT_MAX, (XV + IM[0] * x1) * W));  // NOLINT
          double fY = std::max((double)INT_MIN, std::min((double)INT_MAX, (YV + IM[3] * x1) * W));  // NOLINT
          int X = round(fX);
          int Y = round(fY);

          xy[x1 * 2] = IntCastShort(X >> kBits);
          xy[x1 * 2 + 1] = IntCastShort(Y >> kBits);
          t_a[x1] = (int16_t)((Y & (kTabSz - 1)) * kTabSz + (X & (kTabSz - 1)));
        }
      }
      LiteMat _matA(tw, th, 1, TA, LDataType::UINT16);
      Remap(src, lite_part, _HW, _matA, borderType, borderValue);
    }
  }
  return true;
}

}  // namespace dataset
}  // namespace mindspore
