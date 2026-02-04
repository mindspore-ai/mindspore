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

  https://github.com/tensorflow/tensorflow/blob/v2.6.2/tensorflow/core/kernels/image/resize_bicubic_op.cc

  Additional modifications made by Huawei Technologies Co., Ltd in 2022-2025.
*/

#include "kernel/cpu/native/resize_bicubic_grad_cpu_kernel.h"
#include <limits>
#include <utility>
#include <array>
#include <memory>
#include <numeric>
#include "include/runtime/hardware_abstract/kernel_base/kernel_utils.h"
#include "mindspore/ops/infer/ops_func_impl/resize_bicubic_grad.h"

namespace mindspore {
namespace kernel {
namespace resize_bicubic_grad_cpu {
namespace {
constexpr size_t kResizeBicubicGradInputsNum = 4;
constexpr size_t kResizeBicubicGradOutputNum = 1;
constexpr int64_t kCachedValuesHandMax = 4;
constexpr int64_t kCalnum8 = 8;
constexpr int64_t kCalnum5 = 5;
constexpr int64_t kCalnum4 = 4;
constexpr int64_t kCalnum3 = 3;
constexpr int64_t kCalnum2 = 2;
constexpr size_t i0 = 0;
constexpr size_t i1 = 1;
constexpr size_t i2 = 2;
constexpr size_t i3 = 3;
constexpr size_t kResizeBicubicGradRank = 4;
static const int64_t kTableSize = (1 << 10);
const int64_t kParallelDataNum = 1024 * 256;
}  // namespace

struct ResizerGradState {
  void CalculateSize(const std::vector<int64_t> &resize_shape, const std::vector<int64_t> &origin_shape,
                     bool align_corners_flag) {
    batch_size = resize_shape[kIndex0];
    channels = resize_shape[kIndex1];
    resized_height = resize_shape[kIndex2];
    resized_width = resize_shape[kIndex3];
    original_height = origin_shape[kIndex2];
    original_width = origin_shape[kIndex3];
    height_scale = Scaling(original_height, resized_height, align_corners_flag);
    width_scale = Scaling(original_width, resized_width, align_corners_flag);
    origin_chw = channels * original_height * original_width;
    origin_hw = original_height * original_width;
    resized_chw = resized_height * resized_width * channels;
    resized_hw = resized_height * resized_width;
  }
  int64_t batch_size;
  int64_t channels;
  int64_t original_height;
  int64_t original_width;
  int64_t resized_height;
  int64_t resized_width;
  float height_scale;
  float width_scale;
  int64_t origin_chw;
  int64_t origin_hw;
  int64_t resized_chw;
  int64_t resized_hw;
};

class ResizeBicubicGradWeightsInfo {
 public:
  ResizeBicubicGradWeightsInfo() : weights{}, indices{}, advance(0) {}

  std::array<float, kResizeBicubicGradRank> weights;
  std::array<int64_t, kResizeBicubicGradRank> indices;
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

struct HalfPixelScalerGrad {
  HalfPixelScalerGrad() {}
  inline float operator()(const int64_t x, const float scale) const {
    return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }
};

struct LegacyScalerGrad {
  LegacyScalerGrad() {}
  inline float operator()(const int64_t x, const float scale) const { return static_cast<float>(x) * scale; }
};

class CachedInterpolationCalculator {
 public:
  CachedInterpolationCalculator() : indexes_{-1, -1, -1, -1} {}
  inline size_t Advance(const int64_t x_0, const int64_t x_1, const int64_t x_2, const int64_t x_3) {
    const std::array<int64_t, kResizeBicubicGradRank> new_x_indices{{x_0, x_1, x_2, x_3}};
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
  std::array<int64_t, kResizeBicubicGradRank> indexes_;
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
inline void GetWeightsAndIndicesGrad(const float scale, const int64_t out_loc, const int64_t limit,
                                     ResizeBicubicGradWeightsInfo *out) {
  const Scaler scaler;
  const float in_loc_f = scaler(out_loc, scale);
  const int64_t in_loc = std::floor(in_loc_f);
  const float delta = in_loc_f - in_loc;
  const int64_t offset = lrintf(delta * kTableSize);
  const auto &coeffs_table = GetCoeffsTable(use_keys_cubic);

  out->SetWeightsAndIndices(in_loc, limit, offset, coeffs_table, use_keys_cubic);
}

static void ComputeGradientXWeightsAndIndices(const ResizerGradState &stat, const bool half_pixel_centers_,
                                              std::vector<ResizeBicubicGradWeightsInfo> *x_wais) {
  CachedInterpolationCalculator calc;
  if (half_pixel_centers_) {
    for (int64_t x = 0; x < stat.resized_width; ++x) {
      GetWeightsAndIndicesGrad<HalfPixelScalerGrad, true>(stat.width_scale, x, stat.original_width,
                                                          &(*x_wais)[static_cast<size_t>(x)]);
      auto &x_wai = (*x_wais)[static_cast<size_t>(x)];
      x_wai.SetAdvance(calc.Advance(x_wai.indices[i0], x_wai.indices[i1], x_wai.indices[i2], x_wai.indices[i3]));
    }
  } else {
    for (int64_t x = 0; x < stat.resized_width; ++x) {
      GetWeightsAndIndicesGrad<LegacyScalerGrad, false>(stat.width_scale, x, stat.original_width,
                                                        &(*x_wais)[static_cast<size_t>(x)]);
      auto &x_wai = (*x_wais)[static_cast<size_t>(x)];
      x_wai.SetAdvance(calc.Advance(x_wai.indices[i0], x_wai.indices[i1], x_wai.indices[i2], x_wai.indices[i3]));
    }
  }
}

template <typename T>
void ResizeCommomCalc(const ResizerGradState &stat, const bool half_pixel_centers,
                      const std::vector<ResizeBicubicGradWeightsInfo> &x_wais, const float *input_grad, T *output_grad,
                      int64_t b, int64_t c, int64_t y) {
#define UNROLL_4X4_LOOP(y_wai, x_wai, curr_input_grad, output_grad_index, b, c, output_grad)                           \
  do {                                                                                                                 \
    const float grad_val = curr_input_grad;                                                                            \
    const auto &y_weights = y_wai.weights;                                                                             \
    const auto &y_indices = y_wai.indices;                                                                             \
    const auto &x_weights = x_wai.weights;                                                                             \
    const auto &x_indices = x_wai.indices;                                                                             \
    output_grad[output_grad_index(b, c, y_indices[i0], x_indices[i0])] += T(grad_val * y_weights[i0] * x_weights[i0]); \
    output_grad[output_grad_index(b, c, y_indices[i0], x_indices[i1])] += T(grad_val * y_weights[i0] * x_weights[i1]); \
    output_grad[output_grad_index(b, c, y_indices[i0], x_indices[i2])] += T(grad_val * y_weights[i0] * x_weights[i2]); \
    output_grad[output_grad_index(b, c, y_indices[i0], x_indices[i3])] += T(grad_val * y_weights[i0] * x_weights[i3]); \
    output_grad[output_grad_index(b, c, y_indices[i1], x_indices[i0])] += T(grad_val * y_weights[i1] * x_weights[i0]); \
    output_grad[output_grad_index(b, c, y_indices[i1], x_indices[i1])] += T(grad_val * y_weights[i1] * x_weights[i1]); \
    output_grad[output_grad_index(b, c, y_indices[i1], x_indices[i2])] += T(grad_val * y_weights[i1] * x_weights[i2]); \
    output_grad[output_grad_index(b, c, y_indices[i1], x_indices[i3])] += T(grad_val * y_weights[i1] * x_weights[i3]); \
    output_grad[output_grad_index(b, c, y_indices[i2], x_indices[i0])] += T(grad_val * y_weights[i2] * x_weights[i0]); \
    output_grad[output_grad_index(b, c, y_indices[i2], x_indices[i1])] += T(grad_val * y_weights[i2] * x_weights[i1]); \
    output_grad[output_grad_index(b, c, y_indices[i2], x_indices[i2])] += T(grad_val * y_weights[i2] * x_weights[i2]); \
    output_grad[output_grad_index(b, c, y_indices[i2], x_indices[i3])] += T(grad_val * y_weights[i2] * x_weights[i3]); \
    output_grad[output_grad_index(b, c, y_indices[i3], x_indices[i0])] += T(grad_val * y_weights[i3] * x_weights[i0]); \
    output_grad[output_grad_index(b, c, y_indices[i3], x_indices[i1])] += T(grad_val * y_weights[i3] * x_weights[i1]); \
    output_grad[output_grad_index(b, c, y_indices[i3], x_indices[i2])] += T(grad_val * y_weights[i3] * x_weights[i2]); \
    output_grad[output_grad_index(b, c, y_indices[i3], x_indices[i3])] += T(grad_val * y_weights[i3] * x_weights[i3]); \
  } while (0)

  auto input_grad_index = [&stat](int64_t x1, int64_t x2, int64_t x3, int64_t x4) -> int64_t {
    return x1 * stat.resized_chw + x2 * stat.resized_hw + x3 * stat.resized_width + x4;
  };

  auto output_grad_index = [&stat](int64_t x1, int64_t x2, int64_t x3, int64_t x4) -> int64_t {
    return x1 * stat.origin_chw + x2 * stat.origin_hw + x3 * stat.original_width + x4;
  };

  ResizeBicubicGradWeightsInfo y_wai;
  if (half_pixel_centers) {
    GetWeightsAndIndicesGrad<HalfPixelScalerGrad, true>(stat.height_scale, y, stat.original_height, &y_wai);
  } else {
    GetWeightsAndIndicesGrad<LegacyScalerGrad, false>(stat.height_scale, y, stat.original_height, &y_wai);
  }

  for (int64_t x = 0; x < stat.resized_width; ++x) {
    const ResizeBicubicGradWeightsInfo &x_wai = x_wais[static_cast<size_t>(x)];
    float curr_input_grad = input_grad[input_grad_index(b, c, y, x)];

    UNROLL_4X4_LOOP(y_wai, x_wai, curr_input_grad, output_grad_index, b, c, output_grad);
  }

#undef UNROLL_4X4_LOOP
}

template <typename T>
void CalNonUtil(const ResizerGradState &stat, const bool half_pixel_centers,
                const std::vector<ResizeBicubicGradWeightsInfo> &x_wais, const float *input_grad, T *output_grad) {
  for (int64_t b = 0; b < stat.batch_size; ++b) {
    for (int64_t c = 0; c < stat.channels; ++c) {
      for (int64_t y = 0; y < stat.resized_height; ++y) {
        ResizeCommomCalc(stat, half_pixel_centers, x_wais, input_grad, output_grad, b, c, y);
      }
    }
  }
}

template <typename T>
inline void ResizeBicubicGrad(const float *input_grad, const ResizerGradState &stat, const bool half_pixel_centers_,
                              T *output_grad) {
  std::vector<ResizeBicubicGradWeightsInfo> x_wais(stat.resized_width);
  ComputeGradientXWeightsAndIndices(stat, half_pixel_centers_, &x_wais);
  bool need_parallel = false;
  if (stat.original_width * stat.original_height * stat.channels * stat.batch_size >= kParallelDataNum) {
    need_parallel = true;
  }
  if (need_parallel) {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        const int64_t b = SizeToLong(i) / (stat.channels * stat.resized_height);
        const int64_t c = SizeToLong(i) / stat.resized_height % stat.channels;
        const int64_t y = SizeToLong(i) % stat.resized_height;
        ResizeCommomCalc(stat, half_pixel_centers_, x_wais, input_grad, output_grad, b, c, y);
      }
    };
    const size_t parallel_num = static_cast<size_t>(stat.batch_size * stat.channels * stat.resized_height);
    CPUKernelUtils::ParallelFor(task, parallel_num);
  } else {
    CalNonUtil(stat, half_pixel_centers_, x_wais, input_grad, output_grad);
  }
}

bool ResizeBicubicGradCPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeBicubicGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeBicubicGradOutputNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ResizeBicubicGradCPUKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  resize_shape_ = inputs.at(kIndex0)->GetDeviceShapeVector();
  origin_shape_ = inputs.at(kIndex1)->GetDeviceShapeVector();
  align_corners_ = inputs.at(kIndex2)->GetValueWithCheck<bool>();
  half_pixel_centers_ = inputs.at(kIndex3)->GetValueWithCheck<bool>();
  return KRET_OK;
}

template <typename T>
bool ResizeBicubicGradCPUKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) const {
  auto dy = GetDeviceAddress<float>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(dy);
  auto dx = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(dx);
  size_t dx_size = outputs[kIndex0]->size();
  if (memset_s(dx, dx_size, 0, dx_size) != EOK) {
    MS_EXCEPTION(ValueError) << "Memset Failed!";
  }
  ResizerGradState stat;
  stat.CalculateSize(resize_shape_, origin_shape_, align_corners_);
  ResizeBicubicGrad(dy, stat, half_pixel_centers_, dx);
  return true;
}

std::vector<std::pair<KernelAttr, ResizeBicubicGradCPUKernelMod::ResizeBicubicGradFunc>>
  ResizeBicubicGradCPUKernelMod::func_list_ = {{KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat32)
                                                  .AddInputAttr(kNumberTypeFloat32)
                                                  .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                  .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                  .AddOutputAttr(kNumberTypeFloat32),
                                                &ResizeBicubicGradCPUKernelMod::LaunchKernel<float>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat32)
                                                  .AddInputAttr(kNumberTypeFloat64)
                                                  .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                  .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                  .AddOutputAttr(kNumberTypeFloat64),
                                                &ResizeBicubicGradCPUKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> ResizeBicubicGradCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeBicubicGradFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeBicubicGrad, ResizeBicubicGradCPUKernelMod);
}  // namespace resize_bicubic_grad_cpu
}  // namespace kernel
}  // namespace mindspore
