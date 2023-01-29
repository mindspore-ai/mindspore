/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/resize_bicubic_grad_cpu_kernel.h"
#include <limits>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/grad/resize_bicubic_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeBicubicGradInputsNum = 2;
constexpr size_t kResizeBicubicGradOutputNum = 1;
constexpr int64_t cached_values_hand_max = 4;
constexpr size_t caseid3 = 3;
constexpr int64_t calnum8 = 8;
constexpr int64_t calnum5 = 5;
constexpr int64_t calnum4 = 4;
constexpr int64_t calnum3 = 3;
constexpr int64_t calnum2 = 2;
static const int64_t kTableSize = (1 << 10);
const int64_t kParallelDataNum = 1024 * 256;
std::vector<int64_t> shape0;
std::vector<int64_t> shape1;
bool align_corners = false;
bool half_pixel_centers = false;
int64_t origin_chw;
int64_t origin_hw;
int64_t resized_chw;
int64_t resized_hw;
}  // namespace

struct ResizerGradState {
  void CalculateSize(const std::vector<int64_t> &shape0, const std::vector<int64_t> &shape1) {
    batch_size = shape0[kIndex0];
    channels = shape0[kIndex1];
    resized_height = shape0[kIndex2];
    resized_width = shape0[kIndex3];
    original_height = shape1[kIndex2];
    original_width = shape1[kIndex3];
    height_scale = Scaling(original_height, resized_height, align_corners);
    width_scale = Scaling(original_width, resized_width, align_corners);
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
};

struct WeightsAndIndices {
  float weight_0;
  float weight_1;
  float weight_2;
  float weight_3;
  int64_t index_0;
  int64_t index_1;
  int64_t index_2;
  int64_t index_3;
  size_t advance;
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
    const std::array<int64_t, 4> new_x_indices{{x_0, x_1, x_2, x_3}};
    size_t cached_values_hand = 0;
    size_t new_indices_hand = 0;
    while (cached_values_hand < cached_values_hand_max) {
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
    std::vector<int64_t> values = {x_0, x_1, x_2, x_3};
    for (size_t i = new_indices_hand; i <= caseid3; ++i) {
      indexes_[i] = values[i];
    }
    return new_indices_hand;
  }

 private:
  int64_t indexes_[4];
};

const float *InitCoeffsTable_(const double a) {
  float *coeffs_table = new float[(kTableSize + 1) * 2];
  for (int64_t i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_table[i * calnum2] = ((a + calnum2) * x - (a + calnum3)) * x * x + 1;
    x += 1.0;
    coeffs_table[i * calnum2 + 1] = ((a * x - calnum5 * a) * x + calnum8 * a) * x - calnum4 * a;
  }

  return coeffs_table;
}

const float *GetCoeffsTable_(const bool use_keys_cubic) {
  if (use_keys_cubic) {
    static const float *coeffs_table = InitCoeffsTable_(-0.5f);
    return coeffs_table;
  }
  static const float *coeffs_table = InitCoeffsTable_(-0.75f);
  return coeffs_table;
}

inline int64_t Bound(int64_t val, int64_t limit) { return std::min(limit - 1, std::max(int64_t{0}, val)); }

template <typename Scaler, bool use_keys_cubic>
inline void GetWeightsAndIndicesGrad(const float scale, const int64_t out_loc, const int64_t limit,
                                     WeightsAndIndices *out) {
  const Scaler scaler;
  const float in_loc_f = scaler(out_loc, scale);
  const int64_t in_loc = std::floor(in_loc_f);
  const float delta = in_loc_f - in_loc;
  const int64_t offset = lrintf(delta * kTableSize);
  const float *coeffs_table = GetCoeffsTable_(use_keys_cubic);
  if (use_keys_cubic) {
    out->index_0 = Bound(in_loc - 1, limit);
    out->weight_0 = (out->index_0 == in_loc - 1 ? coeffs_table[offset * calnum2 + 1] : 0.0f);
    out->index_1 = Bound(in_loc, limit);
    out->weight_1 = (out->index_1 == in_loc ? coeffs_table[offset * calnum2] : 0.0f);
    out->index_2 = Bound(in_loc + 1, limit);
    out->weight_2 = (out->index_2 == in_loc + 1 ? coeffs_table[(kTableSize - offset) * calnum2] : 0.0f);
    out->index_3 = Bound(in_loc + calnum2, limit);
    out->weight_3 = (out->index_3 == in_loc + calnum2 ? coeffs_table[(kTableSize - offset) * calnum2 + 1] : 0.0f);

    const float weight_sum = out->weight_0 + out->weight_1 + out->weight_2 + out->weight_3;
    if (std::abs(weight_sum) >= 1000.0f * std::numeric_limits<float>::min()) {
      const float one_over_weight_sum = 1.0f / weight_sum;
      out->weight_0 *= one_over_weight_sum;
      out->weight_1 *= one_over_weight_sum;
      out->weight_2 *= one_over_weight_sum;
      out->weight_3 *= one_over_weight_sum;
    }
  } else {
    out->weight_0 = coeffs_table[offset * calnum2 + 1];
    out->weight_1 = coeffs_table[offset * calnum2];
    out->weight_2 = coeffs_table[(kTableSize - offset) * calnum2];
    out->weight_3 = coeffs_table[(kTableSize - offset) * calnum2 + 1];
    out->index_0 = Bound(in_loc - 1, limit);
    out->index_1 = Bound(in_loc, limit);
    out->index_2 = Bound(in_loc + 1, limit);
    out->index_3 = Bound(in_loc + calnum2, limit);
  }
}

static void ComputeGradientXWeightsAndIndices(const ResizerGradState &RGS, const bool half_pixel_centers_,
                                              std::vector<WeightsAndIndices> *x_wais) {
  CachedInterpolationCalculator calc;
  if (half_pixel_centers_) {
    for (int64_t x = 0; x < RGS.resized_width; ++x) {
      GetWeightsAndIndicesGrad<HalfPixelScalerGrad, true>(RGS.width_scale, x, RGS.original_width,
                                                          &(*x_wais)[static_cast<size_t>(x)]);
      auto &x_wai = (*x_wais)[static_cast<size_t>(x)];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2, x_wai.index_3);
    }
  } else {
    for (int64_t x = 0; x < RGS.resized_width; ++x) {
      GetWeightsAndIndicesGrad<LegacyScalerGrad, false>(RGS.width_scale, x, RGS.original_width,
                                                        &(*x_wais)[static_cast<size_t>(x)]);
      auto &x_wai = (*x_wais)[static_cast<size_t>(x)];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2, x_wai.index_3);
    }
  }
}

const int64_t Calindex(const ResizerGradState &RGS, const int64_t &x1, const int64_t &x2, const int64_t &x3,
                       const int64_t &x4, bool flag_) {
  if (!flag_) {
    return x1 * origin_chw + x2 * origin_hw + x3 * RGS.original_width + x4;
  } else {
    return x1 * resized_chw + x2 * resized_hw + x3 * RGS.resized_width + x4;
  }
}

template <typename T>
void ResizeCommomCalc(const ResizerGradState &RGS, const bool half_pixel_centers,
                      const std::vector<WeightsAndIndices> &x_wais, const bool flag, const float *input_grad,
                      T *output_grad, int64_t b, int64_t c, int64_t y) {
  WeightsAndIndices y_wai;
  if (half_pixel_centers) {
    GetWeightsAndIndicesGrad<HalfPixelScalerGrad, true>(RGS.height_scale, y, RGS.original_height, &y_wai);
  } else {
    GetWeightsAndIndicesGrad<LegacyScalerGrad, false>(RGS.height_scale, y, RGS.original_height, &y_wai);
  }
  for (int64_t x = 0; x < RGS.resized_width; ++x) {
    const WeightsAndIndices &x_wai = x_wais[static_cast<size_t>(x)];
    float curr_input_grad = input_grad[Calindex(RGS, b, c, y, x, flag)];
    // row 0 of 0, 1, 2, 3
    output_grad[Calindex(RGS, b, c, y_wai.index_0, x_wai.index_0, !flag)] +=
      T(curr_input_grad * y_wai.weight_0 * x_wai.weight_0);
    output_grad[Calindex(RGS, b, c, y_wai.index_0, x_wai.index_1, !flag)] +=
      T(curr_input_grad * y_wai.weight_0 * x_wai.weight_1);
    output_grad[Calindex(RGS, b, c, y_wai.index_0, x_wai.index_2, !flag)] +=
      T(curr_input_grad * y_wai.weight_0 * x_wai.weight_2);
    output_grad[Calindex(RGS, b, c, y_wai.index_0, x_wai.index_3, !flag)] +=
      T(curr_input_grad * y_wai.weight_0 * x_wai.weight_3);

    // row 1 of 0, 1, 2, 3
    output_grad[Calindex(RGS, b, c, y_wai.index_1, x_wai.index_0, !flag)] +=
      T(curr_input_grad * y_wai.weight_1 * x_wai.weight_0);
    output_grad[Calindex(RGS, b, c, y_wai.index_1, x_wai.index_1, !flag)] +=
      T(curr_input_grad * y_wai.weight_1 * x_wai.weight_1);
    output_grad[Calindex(RGS, b, c, y_wai.index_1, x_wai.index_2, !flag)] +=
      T(curr_input_grad * y_wai.weight_1 * x_wai.weight_2);
    output_grad[Calindex(RGS, b, c, y_wai.index_1, x_wai.index_3, !flag)] +=
      T(curr_input_grad * y_wai.weight_1 * x_wai.weight_3);

    // row 2 of 0, 1, 2, 3
    output_grad[Calindex(RGS, b, c, y_wai.index_2, x_wai.index_0, !flag)] +=
      T(curr_input_grad * y_wai.weight_2 * x_wai.weight_0);
    output_grad[Calindex(RGS, b, c, y_wai.index_2, x_wai.index_1, !flag)] +=
      T(curr_input_grad * y_wai.weight_2 * x_wai.weight_1);
    output_grad[Calindex(RGS, b, c, y_wai.index_2, x_wai.index_2, !flag)] +=
      T(curr_input_grad * y_wai.weight_2 * x_wai.weight_2);
    output_grad[Calindex(RGS, b, c, y_wai.index_2, x_wai.index_3, !flag)] +=
      T(curr_input_grad * y_wai.weight_2 * x_wai.weight_3);

    // row 3 of 0, 1, 2, 3
    output_grad[Calindex(RGS, b, c, y_wai.index_3, x_wai.index_0, !flag)] +=
      T(curr_input_grad * y_wai.weight_3 * x_wai.weight_0);
    output_grad[Calindex(RGS, b, c, y_wai.index_3, x_wai.index_1, !flag)] +=
      T(curr_input_grad * y_wai.weight_3 * x_wai.weight_1);
    output_grad[Calindex(RGS, b, c, y_wai.index_3, x_wai.index_2, !flag)] +=
      T(curr_input_grad * y_wai.weight_3 * x_wai.weight_2);
    output_grad[Calindex(RGS, b, c, y_wai.index_3, x_wai.index_3, !flag)] +=
      T(curr_input_grad * y_wai.weight_3 * x_wai.weight_3);
  }
}

template <typename T>
void CalNonUtil(const ResizerGradState &RGS, const bool half_pixel_centers,
                const std::vector<WeightsAndIndices> &x_wais, const bool flag, const float *input_grad,
                T *output_grad) {
  for (int64_t b = 0; b < RGS.batch_size; ++b) {
    for (int64_t c = 0; c < RGS.channels; ++c) {
      for (int64_t y = 0; y < RGS.resized_height; ++y) {
        ResizeCommomCalc(RGS, half_pixel_centers, x_wais, flag, input_grad, output_grad, b, c, y);
      }
    }
  }
}

template <typename T>
inline void ResizeBicubicGrad(const float *input_grad, const ResizerGradState &RGS, const bool half_pixel_centers_,
                              T *output_grad) {
  std::vector<WeightsAndIndices> x_wais(RGS.resized_width);
  ComputeGradientXWeightsAndIndices(RGS, half_pixel_centers_, &x_wais);
  const bool flag = true;
  bool utils_flag = false;
  if (RGS.original_width * RGS.original_height * RGS.channels * RGS.batch_size >= kParallelDataNum) {
    utils_flag = true;
  }
  if (utils_flag) {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        const int64_t b = i / (RGS.channels * RGS.resized_height), c = i / RGS.resized_height % RGS.channels;
        const int64_t y = i % RGS.resized_height;
        ResizeCommomCalc(RGS, half_pixel_centers_, x_wais, flag, input_grad, output_grad, b, c, y);
      }
    };
    const size_t parallel_num = static_cast<size_t>(RGS.batch_size * RGS.channels * RGS.resized_height);
    CPUKernelUtils::ParallelFor(task, parallel_num);
  } else {
    CalNonUtil(RGS, half_pixel_centers_, x_wais, flag, input_grad, output_grad);
  }
}

bool ResizeBicubicGradCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeBicubicGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeBicubicGradOutputNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeBicubicGrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  align_corners = kernel_ptr->get_align_corners();
  half_pixel_centers = kernel_ptr->get_half_pixel_centers();
  return true;
}

int ResizeBicubicGradCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  shape0 = inputs[kIndex0]->GetDeviceShapeAdaptively();
  shape1 = inputs[kIndex1]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename T>
bool ResizeBicubicGradCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &outputs) {
  auto input0_addr = static_cast<float *>(inputs[0]->addr);
  auto output_addr = static_cast<T *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size;
  if (memset_s(output_addr, output_size, 0, output_size) != EOK) {
    MS_EXCEPTION(ValueError) << "Memset Failed!";
  }
  ResizerGradState sta;
  sta.CalculateSize(shape0, shape1);
  ResizeBicubicGrad(input0_addr, sta, half_pixel_centers, output_addr);
  return true;
}

std::vector<std::pair<KernelAttr, ResizeBicubicGradCPUKernelMod::ResizeBicubicGradFunc>>
  ResizeBicubicGradCPUKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ResizeBicubicGradCPUKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ResizeBicubicGradCPUKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> ResizeBicubicGradCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeBicubicGradFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeBicubicGrad, ResizeBicubicGradCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
