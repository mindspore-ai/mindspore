/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/resize_bicubic_op.cc

Additional modifications made by Huawei Technologies Co., Ltd in 2021.
==============================================================================*/

#include "ms_kernel/resize_bicubic.h"

#include <algorithm>
#include <vector>
#include <limits>

#include "include/securec.h"
#if __has_include("context/common/cpu_kernel_utils.h")
#include "context/common/cpu_kernel_utils.h"
#else
#include "context/inc/cpu_kernel_utils.h"
#endif
#include "cpu_types.h"
#include "inc/kernel_log.h"
#if __has_include("context/utils/status.h")
#include "context/utils/status.h"
#else
#include "context/common/status.h"
#endif
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace aicpu {
namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
const int64_t kTableSize = (1 << 10);
const int kCalNum4 = 4;
const char *kResizeBicubic = "ResizeBicubic";
std::vector<int64_t> out_shape_;
std::vector<int64_t> in_shape_;

constexpr int64_t kCachedValuesHandMax = 4;
constexpr int64_t kCalnum8 = 8;
constexpr int64_t kCalnum5 = 5;
constexpr int64_t kCalnum4 = 4;
constexpr int64_t kCalnum3 = 3;
constexpr int64_t kCalnum2 = 2;
constexpr int64_t kIndex0 = 0;
constexpr int64_t kIndex1 = 1;
constexpr int64_t kIndex2 = 2;
constexpr int64_t kIndex3 = 3;
constexpr size_t kResizeBicubicRank = 4;
}  // namespace

inline float Scaling(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}

void ResizerState::CalculateSize(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape,
                                 bool align_corners_flag) {
  batch_size = x_shape[kIndex0];
  channels = x_shape[kIndex1];
  in_height = x_shape[kIndex2];
  in_width = x_shape[kIndex3];
  out_height = y_shape[kIndex2];
  out_width = y_shape[kIndex3];
  out_hw_size = out_height * out_width;
  in_hw_size = in_height * in_width;
  bchw_size = in_hw_size * channels * batch_size;
  height_scale = Scaling(in_height, out_height, align_corners_flag);
  width_scale = Scaling(in_width, out_width, align_corners_flag);
}

struct HalfPixelScaler {
  HalfPixelScaler() {}
  inline float operator()(const int64_t x, const float scale) const {
    return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }
};
struct LegacyScaler {
  LegacyScaler() {}
  inline float operator()(const int64_t x, const float scale) const { return static_cast<float>(x) * scale; }
};

class ResizeBicubicWeightsInfo {
 public:
  std::array<float, kResizeBicubicRank> weights;
  std::array<int64_t, kResizeBicubicRank> indices;
  size_t advance;

  inline void SetWeightsAndIndices(const int64_t in_loc, const int64_t limit, const int64_t offset,
                                   const std::vector<float> &coeffs_table, const bool use_keys_cubic) {
    auto clamp = [limit](int64_t v) -> int64_t { return std::min(limit - 1, std::max<int64_t>(0, v)); };
    if (use_keys_cubic) {
      indices[kIndex0] = clamp(in_loc - 1);
      indices[kIndex1] = clamp(in_loc);
      indices[kIndex2] = clamp(in_loc + 1);
      indices[kIndex3] = clamp(in_loc + kCalnum2);
      weights[kIndex0] = (indices[kIndex0] == in_loc - 1 ? coeffs_table[offset * kCalnum2 + 1] : 0.0f);
      weights[kIndex1] = (indices[kIndex1] == in_loc ? coeffs_table[offset * kCalnum2] : 0.0f);
      weights[kIndex2] = (indices[kIndex2] == in_loc + 1 ? coeffs_table[(kTableSize - offset) * kCalnum2] : 0.0f);
      weights[kIndex3] =
        (indices[kIndex3] == in_loc + kCalnum2 ? coeffs_table[(kTableSize - offset) * kCalnum2 + 1] : 0.0f);
      NormalizeWeightsIfNeeded();
    } else {
      weights[kIndex0] = coeffs_table[offset * kCalnum2 + 1];
      weights[kIndex1] = coeffs_table[offset * kCalnum2];
      weights[kIndex2] = coeffs_table[(kTableSize - offset) * kCalnum2];
      weights[kIndex3] = coeffs_table[(kTableSize - offset) * kCalnum2 + 1];
      indices[kIndex0] = clamp(in_loc - 1);
      indices[kIndex1] = clamp(in_loc);
      indices[kIndex2] = clamp(in_loc + 1);
      indices[kIndex3] = clamp(in_loc + kCalnum2);
    }
  }

  inline void SetAdvance(const size_t v) { advance = v; }

  inline void NormalizeWeightsIfNeeded() {
    const float weight_sum = weights[kIndex0] + weights[kIndex1] + weights[kIndex2] + weights[kIndex3];
    if (std::abs(weight_sum) >= 1000.0f * std::numeric_limits<float>::min()) {
      const float one_over_weight_sum = 1.0f / weight_sum;
      std::transform(weights.begin(), weights.end(), weights.begin(),
                     [one_over_weight_sum](float w) { return w * one_over_weight_sum; });
    }
  }
};

inline float InterpolateFromArray(const std::array<float, 4> &weights, const float *values) {
  return std::inner_product(weights.begin(), weights.end(), values, 0.0f);
}

template <typename T>
inline float InterpolateYAtX(const std::array<float, 4> &weights, size_t which, const T *y_ptr_0, const T *y_ptr_1,
                             const T *y_ptr_2, const T *y_ptr_3, const std::array<int64_t, 4> &x_indices) {
  const size_t clamped = which <= kIndex3 ? which : kIndex3;
  const int x_index = static_cast<int>(x_indices[clamped]);
  const float vals[4] = {static_cast<float>(y_ptr_0[x_index]), static_cast<float>(y_ptr_1[x_index]),
                         static_cast<float>(y_ptr_2[x_index]), static_cast<float>(y_ptr_3[x_index])};
  return std::inner_product(std::begin(vals), std::end(vals), weights.begin(), 0.0f);
}

class CachedInterpolationCalculator {
 public:
  CachedInterpolationCalculator() : indexes_{-1, -1, -1, -1} {}
  inline size_t Advance(const int64_t x_0, const int64_t x_1, const int64_t x_2, const int64_t x_3) {
    const std::array<int64_t, kResizeBicubicRank> new_x_indices{{x_0, x_1, x_2, x_3}};
    size_t cached_values_hand = 0;
    size_t new_indices_hand = 0;
    while (cached_values_hand < kCachedValuesHandMax) {
      if (indexes_[cached_values_hand] == new_x_indices[new_indices_hand]) {
        if (new_indices_hand < cached_values_hand) {
          indexes_[new_indices_hand] = indexes_[cached_values_hand];
        }
        cached_values_hand++;
        new_indices_hand++;
      } else {
        cached_values_hand++;
      }
    }
    std::copy(new_x_indices.begin() + new_indices_hand, new_x_indices.end(), indexes_.begin() + new_indices_hand);

    return new_indices_hand;
  }

 private:
  std::array<int64_t, kResizeBicubicRank> indexes_;
};

const std::vector<float> &GetCoeffsTable(const bool use_keys_cubic) {
  auto init_coeffs_table = [](const double val) -> std::vector<float> {
    constexpr float kInvTableSize = 1.0f / kTableSize;
    std::vector<float> coeffs_table((kTableSize + 1) * kCalnum2, 0);

    for (int i = 0; i <= kTableSize; ++i) {
      float x = i * kInvTableSize;
      auto base_position = i * kCalnum2;
      coeffs_table[base_position] = ((val + kCalnum2) * x - (val + kCalnum3)) * x * x + 1;
      x += 1.0f;
      coeffs_table[base_position + 1] = ((val * x - kCalnum5 * val) * x + kCalnum8 * val) * x - kCalnum4 * val;
    }
    return coeffs_table;
  };

  static const std::vector<float> keys_coeffs_table = init_coeffs_table(-0.5f);
  static const std::vector<float> mitchell_coeffs_table = init_coeffs_table(-0.75f);
  return use_keys_cubic ? keys_coeffs_table : mitchell_coeffs_table;
}

template <typename Scaler, bool use_keys_cubic>
inline void ComputeInterpolationWeightsForPosition(const float scale, const int64_t out_loc, const int64_t limit,
                                                   ResizeBicubicWeightsInfo *out) {
  const Scaler scaler;
  const float in_loc_f = scaler(out_loc, scale);
  const int64_t in_loc = std::floor(in_loc_f);
  const float delta = in_loc_f - in_loc;
  const int64_t offset = lrintf(delta * kTableSize);
  const auto &coeffs_table = GetCoeffsTable(use_keys_cubic);
  out->SetWeightsAndIndices(in_loc, limit, offset, coeffs_table, use_keys_cubic);
}

static void PrepareHorizontalInterpolationWeights(const ResizerState &resizer_state, const bool half_pixel_centers_,
                                                  std::vector<ResizeBicubicWeightsInfo> *x_wais) {
  CachedInterpolationCalculator calc;
  if (half_pixel_centers_) {
    for (int64_t x = 0; x < resizer_state.out_width; ++x) {
      ComputeInterpolationWeightsForPosition<HalfPixelScaler, true>(
        resizer_state.width_scale, x, resizer_state.in_width, &(*x_wais)[static_cast<size_t>(x)]);
      auto &x_wai = (*x_wais)[static_cast<size_t>(x)];
      x_wai.SetAdvance(
        calc.Advance(x_wai.indices[kIndex0], x_wai.indices[kIndex1], x_wai.indices[kIndex2], x_wai.indices[kIndex3]));
    }
  } else {
    for (int64_t x = 0; x < resizer_state.out_width; ++x) {
      ComputeInterpolationWeightsForPosition<LegacyScaler, false>(resizer_state.width_scale, x, resizer_state.in_width,
                                                                  &(*x_wais)[static_cast<size_t>(x)]);
      auto &x_wai = (*x_wais)[static_cast<size_t>(x)];
      x_wai.SetAdvance(
        calc.Advance(x_wai.indices[kIndex0], x_wai.indices[kIndex1], x_wai.indices[kIndex2], x_wai.indices[kIndex3]));
    }
  }
}

template <typename T1>
void CalSwitch(const ResizeBicubicWeightsInfo &x_wai, float *cached_value, const ResizeBicubicWeightsInfo &y_wai,
               const T1 *y_ptr_0, const T1 *y_ptr_1, const T1 *y_ptr_2, const T1 *y_ptr_3) {
  switch (x_wai.advance) {
    case kIndex3:
      cached_value[kIndex0] = cached_value[kIndex1];
      cached_value[kIndex1] = cached_value[kIndex2];
      cached_value[kIndex2] = cached_value[kIndex3];
      break;
    case kIndex2:
      cached_value[kIndex0] = cached_value[kIndex2];
      cached_value[kIndex1] = cached_value[kIndex3];
      break;
    case kIndex1:
      cached_value[kIndex0] = cached_value[kIndex3];
      break;
  }
  // Set the remaining '4-advance' values by computing.
  for (size_t i = x_wai.advance; i <= kIndex3; i++) {
    cached_value[i] = InterpolateYAtX(y_wai.weights, i, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai.indices);
  }
}

template <typename T1, typename T2>
uint32_t ResizeBicubicCpuKernel::InterpolateWithCache(CpuKernelContext &ctx, const T1 *input_data, T2 *output_data) {
  const ResizerState &RS = state_info_;
  std::vector<ResizeBicubicWeightsInfo> x_wais(RS.out_width);
  PrepareHorizontalInterpolationWeights(RS, half_pixel_centers_, &x_wais);
  const int64_t in_row_width = RS.in_width * RS.in_height;    // hw
  const int64_t in_batch_width = RS.channels * in_row_width;  // chw
  const int64_t out_ch = RS.out_height * RS.channels;
  const int64_t out_chw = out_ch * RS.out_width;
  const int64_t out_hw = RS.out_height * RS.out_width;
  const size_t parallel_num = static_cast<size_t>(out_ch * RS.batch_size);
  auto task = [&](size_t start, size_t end) {
    std::array<float, 4> cached_value{};
    for (size_t i = start; i < end; ++i) {  // nch
      const int64_t b = static_cast<int64_t>(i) / out_ch;
      const int64_t c = static_cast<int64_t>(i) % out_ch / RS.out_height;
      const int64_t y = static_cast<int64_t>(i) % RS.out_height;
      ResizeBicubicWeightsInfo y_wai;
      if (half_pixel_centers_) {
        ComputeInterpolationWeightsForPosition<HalfPixelScaler, true>(RS.height_scale, y, RS.in_height, &y_wai);
      } else {
        ComputeInterpolationWeightsForPosition<LegacyScaler, false>(RS.height_scale, y, RS.in_height, &y_wai);
      }
      const T1 *input_b_ptr = input_data + b * in_batch_width + c * in_row_width;
      T2 *output_y_ptr = output_data + b * out_chw + c * out_hw + y * RS.out_width;
      // Make pointers represent offsets of data in input_b_ptr
      const T1 *y_ptr_0 = input_b_ptr + y_wai.indices[kIndex0] * RS.in_width;
      const T1 *y_ptr_1 = input_b_ptr + y_wai.indices[kIndex1] * RS.in_width;
      const T1 *y_ptr_2 = input_b_ptr + y_wai.indices[kIndex2] * RS.in_width;
      const T1 *y_ptr_3 = input_b_ptr + y_wai.indices[kIndex3] * RS.in_width;
      for (int64_t x = 0; x < RS.out_width; ++x) {
        const ResizeBicubicWeightsInfo &x_wai = x_wais[static_cast<size_t>(x)];
        CalSwitch(x_wai, cached_value.data(), y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3);
        output_y_ptr[x] = static_cast<T2>(InterpolateFromArray(x_wai.weights, cached_value.data()));
      }
    }
  };
  CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, parallel_num, parallel_num, task),
                           "AcosGrad Compute failed.");
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t ResizeBicubicCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input_addr = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto output_addr = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  state_info_.CalculateSize(ctx.Input(0)->GetTensorShape()->GetDimSizes(),
                            ctx.Output(0)->GetTensorShape()->GetDimSizes(), align_corners_);

  if (state_info_.out_height == state_info_.in_height && state_info_.out_width == state_info_.in_width) {
    for (int64_t i = 0; i < state_info_.bchw_size; ++i) {
      output_addr[i] = static_cast<T2>(input_addr[i]);
    }
    return KERNEL_STATUS_OK;
  }

  return InterpolateWithCache(ctx, input_addr, output_addr);
}

uint32_t ResizeBicubicCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  Tensor *output_tensor = ctx.Output(0);
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "ResizeBicubic check params failed.");

  in_shape_ = input_tensor->GetTensorShape()->GetDimSizes();

  out_shape_ = output_tensor->GetTensorShape()->GetDimSizes();
  int64_t out_height = out_shape_[kIndex2];
  int64_t out_width = out_shape_[kIndex3];

  CUST_KERNEL_CHECK_FALSE(ctx, (out_shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                          "Dim of output[0] must be 4, but got[%zu].", out_shape_.size());
  CUST_KERNEL_CHECK_FALSE(ctx, (out_height > 0 && out_width > 0), KERNEL_STATUS_PARAM_INVALID,
                          "output dimensions must be positive.");

  AttrValue *pattr_align_corners = ctx.GetAttr("align_corners");
  if (pattr_align_corners == nullptr) {
    align_corners_ = false;
  } else {
    align_corners_ = pattr_align_corners->GetBool();
  }

  AttrValue *pattr_half_pixel_centers = ctx.GetAttr("half_pixel_centers");
  if (pattr_half_pixel_centers == nullptr) {
    half_pixel_centers_ = false;
  } else {
    half_pixel_centers_ = pattr_half_pixel_centers->GetBool();
  }

  dtype_ = input_tensor->GetDataType();

  return KERNEL_STATUS_OK;
}

uint32_t ResizeBicubicCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  CUST_KERNEL_CHECK_FALSE(ctx, (res == KERNEL_STATUS_OK), res, "GetInputAndCheck failed.");

  if (dtype_ == DT_FLOAT16) {
    res = DoCompute<Eigen::half, Eigen::half>(ctx);
  } else if (dtype_ == DT_FLOAT) {
    res = DoCompute<float, float>(ctx);
  } else if (dtype_ == DT_DOUBLE) {
    res = DoCompute<double, double>(ctx);
  } else {
    CUST_KERNEL_LOG_ERROR(ctx, "ResizeBicubic doesn't support input tensor types: [%s]", DTypeStr(dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  CUST_KERNEL_CHECK_FALSE(ctx, (res == KERNEL_STATUS_OK), res, "ResizeBicubic Compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kResizeBicubic, ResizeBicubicCpuKernel);
}  // namespace aicpu
