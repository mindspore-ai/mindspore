/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/image/lite_cv/lite_mat.h"

#include <limits>
#include <algorithm>
#include <cmath>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

namespace mindspore {
namespace dataset {

LiteMat::LiteMat() {
  data_ptr_ = nullptr;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  size_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = nullptr;
  setSteps(0, 0, 0);
}

LiteMat::LiteMat(int width, LDataType data_type) {
  data_ptr_ = nullptr;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = nullptr;
  size_ = 0;
  setSteps(0, 0, 0);
  Init(width, data_type);
}

LiteMat::LiteMat(int width, int height, LDataType data_type) {
  data_ptr_ = nullptr;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = nullptr;
  size_ = 0;
  setSteps(0, 0, 0);
  Init(width, height, data_type);
}

LiteMat::LiteMat(int width, int height, void *p_data, LDataType data_type) {
  data_ptr_ = nullptr;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = nullptr;
  size_ = 0;
  setSteps(0, 0, 0);
  Init(width, height, p_data, data_type);
}

LiteMat::LiteMat(int width, int height, int channel, LDataType data_type) {
  data_ptr_ = nullptr;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = nullptr;
  size_ = 0;
  setSteps(0, 0, 0);
  Init(width, height, channel, data_type);
}

LiteMat::LiteMat(int width, int height, int channel, void *p_data, LDataType data_type) {
  data_ptr_ = nullptr;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  dims_ = 0;
  data_type_ = LDataType::UINT8;
  ref_count_ = nullptr;
  size_ = 0;
  setSteps(0, 0, 0);
  Init(width, height, channel, p_data, data_type);
}

LiteMat::~LiteMat() { Release(); }

int LiteMat::addRef(int *p, int value) {
  int v = *p;
  *p += value;
  return v;
}

LiteMat::LiteMat(const LiteMat &m) {
  data_ptr_ = m.data_ptr_;
  elem_size_ = m.elem_size_;
  width_ = m.width_;
  height_ = m.height_;
  channel_ = m.channel_;
  c_step_ = m.c_step_;
  dims_ = m.dims_;
  data_type_ = m.data_type_;
  ref_count_ = m.ref_count_;
  size_ = 0;
  setSteps(m.steps_[0], m.steps_[1], m.steps_[2]);
  if (ref_count_) {
    addRef(ref_count_, 1);
  }
}

void LiteMat::setSteps(int c0, int c1, int c2) {
  steps_[0] = c0;
  steps_[1] = c1;
  steps_[2] = c2;
}

LiteMat &LiteMat::operator=(const LiteMat &m) {
  if (this == &m) {
    return *this;
  }

  if (m.ref_count_) {
    addRef(m.ref_count_, 1);
  }

  Release();
  data_ptr_ = m.data_ptr_;
  elem_size_ = m.elem_size_;
  width_ = m.width_;
  height_ = m.height_;
  channel_ = m.channel_;
  c_step_ = m.c_step_;
  dims_ = m.dims_;
  data_type_ = m.data_type_;
  ref_count_ = m.ref_count_;
  setSteps(m.steps_[0], m.steps_[1], m.steps_[2]);
  size_ = m.size_;
  return *this;
}

void LiteMat::Init(int width, LDataType data_type) {
  Release();
  data_type_ = data_type;
  InitElemSize(data_type);
  width_ = width;
  dims_ = 1;
  height_ = 1;
  channel_ = 1;
  c_step_ = width;
  size_ = c_step_ * elem_size_;
  data_ptr_ = AlignMalloc(size_);
  ref_count_ = new int[1];
  *ref_count_ = 1;
  steps_[0] = elem_size_;
}

void LiteMat::Init(int width, int height, LDataType data_type) {
  Release();
  data_type_ = data_type;
  InitElemSize(data_type);
  width_ = width;
  height_ = height;
  dims_ = 2;
  channel_ = 1;
  c_step_ = width_ * height_;
  size_ = c_step_ * elem_size_;
  data_ptr_ = AlignMalloc(size_);
  ref_count_ = new int[1];
  *ref_count_ = 1;
  steps_[1] = elem_size_;
  steps_[0] = width_ * steps_[1];
}

void LiteMat::Init(int width, int height, void *p_data, LDataType data_type) {
  data_type_ = data_type;
  InitElemSize(data_type);
  width_ = width;
  height_ = height;
  dims_ = 2;
  channel_ = 1;
  c_step_ = height_ * width_;
  size_ = c_step_ * channel_ * elem_size_;
  data_ptr_ = p_data;
  ref_count_ = nullptr;
  steps_[1] = elem_size_;
  steps_[0] = width_ * steps_[1];
}

void LiteMat::Init(int width, int height, int channel, LDataType data_type) {
  Release();
  data_type_ = data_type;
  InitElemSize(data_type);
  width_ = width;
  height_ = height;
  dims_ = 3;
  channel_ = channel;
  c_step_ = ((height_ * width_ * elem_size_ + ALIGN - 1) & (-ALIGN)) / elem_size_;
  size_ = c_step_ * channel_ * elem_size_;
  data_ptr_ = AlignMalloc(size_);
  ref_count_ = new int[1];
  *ref_count_ = 1;

  steps_[2] = elem_size_;
  steps_[1] = channel * steps_[2];
  steps_[0] = width_ * steps_[1];
}

void LiteMat::Init(int width, int height, int channel, void *p_data, LDataType data_type) {
  data_type_ = data_type;
  InitElemSize(data_type);
  width_ = width;
  height_ = height;
  dims_ = 3;
  channel_ = channel;
  c_step_ = height_ * width_;
  size_ = c_step_ * channel_ * elem_size_;
  data_ptr_ = p_data;
  ref_count_ = nullptr;
  steps_[2] = elem_size_;
  steps_[1] = channel * steps_[2];
  steps_[0] = width_ * steps_[1];
}

bool LiteMat::IsEmpty() const { return data_ptr_ == nullptr || c_step_ * channel_ == 0; }

void LiteMat::Release() {
  if (ref_count_ && (addRef(ref_count_, -1) == 1)) {
    if (data_ptr_) {
      AlignFree(data_ptr_);
    }
    if (ref_count_) {
      delete[] ref_count_;
    }
  }
  data_ptr_ = nullptr;
  elem_size_ = 0;
  width_ = 0;
  height_ = 0;
  channel_ = 0;
  c_step_ = 0;
  ref_count_ = 0;
  size_ = 0;
  setSteps(0, 0, 0);
}

void *LiteMat::AlignMalloc(unsigned int size) {
  unsigned int length = sizeof(void *) + ALIGN - 1;
  if (size > std::numeric_limits<uint32_t>::max() - length) {
    return nullptr;
  }
  void *p_raw = reinterpret_cast<void *>(malloc(size + length));
  if (p_raw) {
    void **p_algin = reinterpret_cast<void **>(((size_t)(p_raw) + length) & ~(ALIGN - 1));
    p_algin[-1] = p_raw;
    return p_algin;
  }
  return nullptr;
}

void LiteMat::AlignFree(void *ptr) {
  (void)free(reinterpret_cast<void **>(ptr)[-1]);
  ptr = nullptr;
}

inline void LiteMat::InitElemSize(LDataType data_type) { elem_size_ = data_type.SizeInBytes(); }

bool LiteMat::GetROI(int x, int y, int w, int h, LiteMat &m) {
  if (x < 0 || y < 0 || x > width_ - w || h > height_ - y || w <= 0 || h <= 0) {
    return false;
  }
  if (!m.IsEmpty()) {
    m.Release();
  }

  if (ref_count_) {
    addRef(ref_count_, 1);
  }

  m.height_ = h;
  m.width_ = w;
  m.dims_ = dims_;
  m.elem_size_ = elem_size_;
  m.data_ptr_ = reinterpret_cast<uint8_t *>(data_ptr_) + y * steps_[0] + x * elem_size_ * channel_;
  m.channel_ = channel_;
  m.c_step_ = c_step_;
  m.data_type_ = data_type_;
  m.ref_count_ = ref_count_;
  m.setSteps(steps_[0], steps_[1], steps_[2]);
  return true;
}

template <typename T>
inline void SubtractImpl(const T *src0, const T *src1, T *dst, int64_t total_size) {
  for (int64_t i = 0; i < total_size; i++) {
    dst[i] = src0[i] - src1[i];
  }
}

template <>
inline void SubtractImpl(const uint8_t *src0, const uint8_t *src1, uint8_t *dst, int64_t total_size) {
  int64_t x = 0;
#ifdef ENABLE_NEON
  const int64_t step = 32;
  for (; x <= total_size - step; x += step) {
    uint8x16_t v_src00 = vld1q_u8(src0 + x);
    uint8x16_t v_src01 = vld1q_u8(src0 + x + 16);
    uint8x16_t v_src10 = vld1q_u8(src1 + x);
    uint8x16_t v_src11 = vld1q_u8(src1 + x + 16);
    uint8x16_t v_dst;

    v_dst = vqsubq_u8(v_src00, v_src10);
    vst1q_u8(dst + x, v_dst);

    v_dst = vqsubq_u8(v_src01, v_src11);
    vst1q_u8(dst + x + 16, v_dst);
  }
#endif
  for (; x < total_size; x++) {
    int32_t val = static_cast<int32_t>(src0[x]) - src1[x];
    dst[x] = std::max<int32_t>(std::numeric_limits<uint8_t>::min(),
                               std::min<int32_t>(std::numeric_limits<uint8_t>::max(), val));
  }
}

template <>
inline void SubtractImpl(const uint16_t *src0, const uint16_t *src1, uint16_t *dst, int64_t total_size) {
  for (int64_t i = 0; i < total_size; i++) {
    int32_t val = static_cast<int32_t>(src0[i]) - src1[i];
    dst[i] = std::max<int32_t>(std::numeric_limits<uint16_t>::min(),
                               std::min<int32_t>(std::numeric_limits<uint16_t>::max(), val));
  }
}

template <>
inline void SubtractImpl(const uint32_t *src0, const uint32_t *src1, uint32_t *dst, int64_t total_size) {
  for (int64_t i = 0; i < total_size; i++) {
    int64_t val = static_cast<int64_t>(src0[i]) - src1[i];
    dst[i] = std::max<int64_t>(std::numeric_limits<uint32_t>::min(),
                               std::min<int64_t>(std::numeric_limits<uint32_t>::max(), val));
  }
}

inline bool CheckSubstract(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst) {
  if (dst == NULL) {
    return false;
  }

  if (src_a.width_ != src_b.width_ || src_a.height_ != src_b.height_ || src_a.channel_ != src_b.channel_) {
    return false;
  }

  if (src_a.data_type_ != src_b.data_type_) {
    return false;
  }
  return true;
}

bool Subtract(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst) {
  if (!CheckSubstract(src_a, src_b, dst)) {
    return false;
  }

  if (dst->IsEmpty()) {
    dst->Init(src_a.width_, src_a.height_, src_a.channel_, src_a.data_type_);
  } else if (src_a.width_ != dst->width_ || src_a.height_ != dst->height_ || src_a.channel_ != dst->channel_) {
    return false;
  } else if (src_a.data_type_ != dst->data_type_) {
    return false;
  }

  int64_t total_size = src_a.height_ * src_a.width_ * src_a.channel_;
  if (src_a.data_type_ == LDataType::BOOL) {
    SubtractImpl<bool>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT8) {
    SubtractImpl<int8_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT8) {
    SubtractImpl<uint8_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT16) {
    SubtractImpl<int16_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT16) {
    SubtractImpl<uint16_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT32) {
    SubtractImpl<int32_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT32) {
    SubtractImpl<uint32_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT64) {
    SubtractImpl<int64_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT64) {
    SubtractImpl<uint64_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::FLOAT32) {
    SubtractImpl<float>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::FLOAT64) {
    SubtractImpl<double>(src_a, src_b, *dst, total_size);
  } else {
    return false;
  }

  return true;
}

#ifdef ENABLE_NEON
inline float32x4_t reciprocal_simd(float32x4_t val) {
  // get an initial estimate of 1/val
  float32x4_t reciprocal = vrecpeq_f32(val);

  // use Newton-Raphson steps to refine the estimate
  reciprocal = vmulq_f32(vrecpsq_f32(val, reciprocal), reciprocal);
  reciprocal = vmulq_f32(vrecpsq_f32(val, reciprocal), reciprocal);
  return reciprocal;
}

inline float32x4_t round_simd(const float32x4_t &v) {
  const int32x4_t signMask = vdupq_n_s32(1U << 31);
  const int32x4_t half = vreinterpretq_s32_f32(vdupq_n_f32(0.5f));
  float32x4_t v_addition = vreinterpretq_f32_s32(vorrq_s32(half, vandq_s32(signMask, vreinterpretq_s32_f32(v))));
  return vaddq_f32(v, v_addition);
}
#endif

template <typename T>
inline void DivideImpl(const T *src0, const T *src1, T *dst, int64_t total_size) {
  for (size_t i = 0; i < total_size; i++) {
    dst[i] = src1[i] ? src0[i] / src1[i] : 0;
  }
}

template <>
inline void DivideImpl(const uint8_t *src0, const uint8_t *src1, uint8_t *dst, int64_t total_size) {
  int64_t x = 0;
#ifdef ENABLE_NEON
  const int64_t step = 16;
  for (; x <= total_size - step; x += step) {
    __builtin_prefetch(reinterpret_cast<const char *>(src0 + x) + 32 * 10);
    __builtin_prefetch(reinterpret_cast<const char *>(src1 + x) + 32 * 10);

    uint8x16_t v_a = vld1q_u8(src0 + x);
    uint8x16_t v_b = vld1q_u8(src1 + x);
    uint8x16_t v_mask = vtstq_u8(v_b, v_b);

    uint16x8_t va_l_16x8 = vmovl_u8(vget_low_u8(v_a));
    uint16x8_t va_h_16x8 = vmovl_u8(vget_high_u8(v_a));
    uint16x8_t vb_l_16x8 = vmovl_u8(vget_low_u8(v_b));
    uint16x8_t vb_h_16x8 = vmovl_u8(vget_high_u8(v_b));

    float32x4_t va_ll_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(va_l_16x8)));
    float32x4_t va_lh_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(va_l_16x8)));
    float32x4_t va_hl_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(va_h_16x8)));
    float32x4_t va_hh_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(va_h_16x8)));
    float32x4_t vb_ll_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vb_l_16x8)));
    float32x4_t vb_lh_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vb_l_16x8)));
    float32x4_t vb_hl_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vb_h_16x8)));
    float32x4_t vb_hh_f32x4 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vb_h_16x8)));

    float32x4_t vb_ll_re_f32x4 = reciprocal_simd(vb_ll_f32x4);
    float32x4_t vb_lh_re_f32x4 = reciprocal_simd(vb_lh_f32x4);
    float32x4_t vb_hl_re_f32x4 = reciprocal_simd(vb_hl_f32x4);
    float32x4_t vb_hh_re_f32x4 = reciprocal_simd(vb_hh_f32x4);

    float32x4_t dst_ll_f32x4 = round_simd(vmulq_f32(va_ll_f32x4, vb_ll_re_f32x4));
    float32x4_t dst_lh_f32x4 = round_simd(vmulq_f32(va_lh_f32x4, vb_lh_re_f32x4));
    float32x4_t dst_hl_f32x4 = round_simd(vmulq_f32(va_hl_f32x4, vb_hl_re_f32x4));
    float32x4_t dst_hh_f32x4 = round_simd(vmulq_f32(va_hh_f32x4, vb_hh_re_f32x4));

    uint32x4_t dst_ll_32x4 = vcvtq_u32_f32(dst_ll_f32x4);
    uint32x4_t dst_lh_32x4 = vcvtq_u32_f32(dst_lh_f32x4);
    uint32x4_t dst_hl_32x4 = vcvtq_u32_f32(dst_hl_f32x4);
    uint32x4_t dst_hh_32x4 = vcvtq_u32_f32(dst_hh_f32x4);

    uint16x4_t dst_ll_16x4 = vqmovn_u32(dst_ll_32x4);
    uint16x4_t dst_lh_16x4 = vqmovn_u32(dst_lh_32x4);
    uint16x4_t dst_hl_16x4 = vqmovn_u32(dst_hl_32x4);
    uint16x4_t dst_hh_16x4 = vqmovn_u32(dst_hh_32x4);

    uint16x8_t dst_l_16x8 = vcombine_u16(dst_ll_16x4, dst_lh_16x4);
    uint16x8_t dst_h_16x8 = vcombine_u16(dst_hl_16x4, dst_hh_16x4);

    int8x8_t dst_l_8x8 = vqmovn_u16(dst_l_16x8);
    int8x8_t dst_h_8x8 = vqmovn_u16(dst_h_16x8);
    int8x16_t dst_8x16 = vcombine_u8(dst_l_8x8, dst_h_8x8);

    dst_8x16 = vandq_u8(dst_8x16, v_mask);
    vst1q_u8(dst + x, dst_8x16);
  }
#endif
  for (; x < total_size; x++) {
    int32_t val = src1[x] ? std::round(src0[x] / src1[x]) : 0;
    dst[x] = std::max<int32_t>(std::numeric_limits<uint8_t>::min(),
                               std::min<int32_t>(std::numeric_limits<uint8_t>::max(), val));
  }
}

template <>
inline void DivideImpl(const uint16_t *src0, const uint16_t *src1, uint16_t *dst, int64_t total_size) {
  for (size_t i = 0; i < total_size; i++) {
    int32_t val = src1[i] ? std::round(src0[i] / src1[i]) : 0;
    dst[i] = std::max<int32_t>(std::numeric_limits<uint16_t>::min(),
                               std::min<int32_t>(std::numeric_limits<uint16_t>::max(), val));
  }
}

template <>
inline void DivideImpl(const uint32_t *src0, const uint32_t *src1, uint32_t *dst, int64_t total_size) {
  for (size_t i = 0; i < total_size; i++) {
    int64_t val = src1[i] ? std::round(src0[i] / src1[i]) : 0;
    dst[i] = std::max<int64_t>(std::numeric_limits<uint32_t>::min(),
                               std::min<int64_t>(std::numeric_limits<uint32_t>::max(), val));
  }
}

inline bool CheckDivide(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst) {
  if (dst == NULL) {
    return false;
  }

  if (src_a.width_ != src_b.width_ || src_a.height_ != src_b.height_ || src_a.channel_ != src_b.channel_) {
    return false;
  }

  if (src_a.data_type_ != src_b.data_type_) {
    return false;
  }
  return true;
}

bool Divide(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst) {
  if (!CheckDivide(src_a, src_b, dst)) {
    return false;
  }

  if (dst->IsEmpty()) {
    dst->Init(src_a.width_, src_a.height_, src_a.channel_, src_a.data_type_);
  } else if (src_a.width_ != dst->width_ || src_a.height_ != dst->height_ || src_a.channel_ != dst->channel_) {
    return false;
  } else if (src_a.data_type_ != dst->data_type_) {
    return false;
  }

  int64_t total_size = src_a.height_ * src_a.width_ * src_a.channel_;
  if (src_a.data_type_ == LDataType::INT8) {
    DivideImpl<int8_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT8) {
    DivideImpl<uint8_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT16) {
    DivideImpl<int16_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT16) {
    DivideImpl<uint16_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT32) {
    DivideImpl<int32_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT32) {
    DivideImpl<uint32_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT64) {
    DivideImpl<int64_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT64) {
    DivideImpl<uint64_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::FLOAT32) {
    DivideImpl<float>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::FLOAT64) {
    DivideImpl<double>(src_a, src_b, *dst, total_size);
  } else {
    return false;
  }
  return true;
}

template <typename T>
inline void MultiplyImpl(const T *src0, const T *src1, T *dst, int64_t total_size) {
  for (size_t i = 0; i < total_size; i++) {
    dst[i] = src0[i] * src1[i];
  }
}

template <>
inline void MultiplyImpl(const uint8_t *src0, const uint8_t *src1, uint8_t *dst, int64_t total_size) {
  int64_t x = 0;
#ifdef ENABLE_NEON
  const int64_t step = 32;
  for (; x <= total_size - step; x += step) {
    uint8x16_t v_src00 = vld1q_u8(src0 + x);
    uint8x16_t v_src01 = vld1q_u8(src0 + x + 16);
    uint8x16_t v_src10 = vld1q_u8(src1 + x);
    uint8x16_t v_src11 = vld1q_u8(src1 + x + 16);
    uint8x16_t v_dst_l, v_dst_h;

    v_dst_l = vmull_u8(vget_low_u8(v_src00), vget_low_u8(v_src10));
    v_dst_h = vmull_u8(vget_high_u8(v_src00), vget_high_u8(v_src10));
    vst1q_u8(dst + x, vcombine_u8(vqmovn_u16(v_dst_l), vqmovn_u16(v_dst_h)));

    v_dst_l = vmull_u8(vget_low_u8(v_src01), vget_low_u8(v_src11));
    v_dst_h = vmull_u8(vget_high_u8(v_src01), vget_high_u8(v_src11));
    vst1q_u8(dst + x + 16, vcombine_u8(vqmovn_u16(v_dst_l), vqmovn_u16(v_dst_h)));
  }
#endif
  for (; x < total_size; x++) {
    int32_t val = src0[x] * src1[x];
    dst[x] = std::max<int32_t>(std::numeric_limits<uint8_t>::min(),
                               std::min<int32_t>(std::numeric_limits<uint8_t>::max(), val));
  }
}

template <>
inline void MultiplyImpl(const uint16_t *src0, const uint16_t *src1, uint16_t *dst, int64_t total_size) {
  for (size_t i = 0; i < total_size; i++) {
    int32_t val = src0[i] * src1[i];
    dst[i] = std::max<int32_t>(std::numeric_limits<uint16_t>::min(),
                               std::min<int32_t>(std::numeric_limits<uint16_t>::max(), val));
  }
}

template <>
inline void MultiplyImpl(const uint32_t *src0, const uint32_t *src1, uint32_t *dst, int64_t total_size) {
  for (size_t i = 0; i < total_size; i++) {
    int64_t val = src0[i] * src1[i];
    dst[i] = std::max<int64_t>(std::numeric_limits<uint32_t>::min(),
                               std::min<int64_t>(std::numeric_limits<uint32_t>::max(), val));
  }
}

inline bool CheckMultiply(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst) {
  if (dst == NULL) {
    return false;
  }

  if (src_a.width_ != src_b.width_ || src_a.height_ != src_b.height_ || src_a.channel_ != src_b.channel_) {
    return false;
  }

  if (src_a.data_type_ != src_b.data_type_) {
    return false;
  }
  return true;
}

bool Multiply(const LiteMat &src_a, const LiteMat &src_b, LiteMat *dst) {
  if (!CheckMultiply(src_a, src_b, dst)) {
    return false;
  }
  if (dst->IsEmpty()) {
    dst->Init(src_a.width_, src_a.height_, src_a.channel_, src_a.data_type_);
  } else if (src_a.width_ != dst->width_ || src_a.height_ != dst->height_ || src_a.channel_ != dst->channel_) {
    return false;
  } else if (src_a.data_type_ != dst->data_type_) {
    return false;
  }

  int64_t total_size = src_a.height_ * src_a.width_ * src_a.channel_;
  if (src_a.data_type_ == LDataType::INT8) {
    MultiplyImpl<int8_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT8) {
    MultiplyImpl<uint8_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT16) {
    MultiplyImpl<int16_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT16) {
    MultiplyImpl<uint16_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT32) {
    MultiplyImpl<int32_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT32) {
    MultiplyImpl<uint32_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::INT64) {
    MultiplyImpl<int64_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::UINT64) {
    MultiplyImpl<uint64_t>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::FLOAT32) {
    MultiplyImpl<float>(src_a, src_b, *dst, total_size);
  } else if (src_a.data_type_ == LDataType::FLOAT64) {
    MultiplyImpl<double>(src_a, src_b, *dst, total_size);
  } else {
    return false;
  }
  return true;
}

}  // namespace dataset
}  // namespace mindspore
