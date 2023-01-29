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

#include "plugin/device/cpu/kernel/resize_bicubic_cpu_kernel.h"
#include <limits>
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/resize_bicubic.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kResizeBicubicInputsNum = 2;
constexpr size_t kResizeBicubicOutputsNum = 1;
constexpr int64_t cached_values_hand_max = 4;
constexpr size_t caseid2 = 2;
constexpr size_t caseid3 = 3;
constexpr int64_t calnum8 = 8;
constexpr int64_t calnum5 = 5;
constexpr int64_t calnum4 = 4;
constexpr int64_t calnum3 = 3;
constexpr int64_t calnum2 = 2;
constexpr int64_t kTableSize = (1 << 10);
std::vector<int64_t> shape0;
std::vector<int64_t> shape1;
bool align_corners = false;
bool half_pixel_centers = false;
struct ResizerState {
  void CalculateSize_kernel_node(const std::vector<KernelTensorPtr> &inputs) {
    shape0 = inputs[kIndex0]->GetDeviceShapeAdaptively();
    batch_size = shape0[kIndex0];
    channels = shape0[kIndex1];
    in_height = shape0[kIndex2];
    in_width = shape0[kIndex3];
  }
  void CalculateSize_inputs(const std::vector<kernel::AddressPtr> &inputs) {
    auto *input_addr = static_cast<int32_t *>(inputs[1]->addr);
    out_height = static_cast<int64_t>(input_addr[0]);
    out_width = static_cast<int64_t>(input_addr[1]);

    out_hw_size = out_height * out_width;
    in_hw_size = in_height * in_width;
    bchw_size = in_hw_size * channels * batch_size;
    height_scale = Scaling(in_height, out_height, align_corners);
    width_scale = Scaling(in_width, out_width, align_corners);
  }
  int64_t batch_size;
  int64_t out_height;
  int64_t out_width;
  int64_t in_height;
  int64_t in_width;
  int64_t channels;
  float height_scale;
  float width_scale;
  int64_t out_hw_size;
  int64_t in_hw_size;
  int64_t bchw_size;
};
ResizerState sta;
}  // namespace

struct WeightsAndIndices {
  float weight_0;
  float weight_1;
  float weight_2;
  float weight_3;
  int64_t index_0;
  int64_t index_1;
  int64_t index_2;
  int64_t index_3;
  size_t advance;  // advance value.
};

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
    std::vector<int64_t> x_values = {x_0, x_1, x_2, x_3};
    for (size_t i = new_indices_hand; i < x_values.size(); ++i) {
      indexes_[i] = x_values[i];
    }

    return new_indices_hand;
  }

 private:
  int64_t indexes_[kIndex4];
};

inline int64_t Bound(int64_t val, int64_t limit) { return std::min(limit - 1, std::max(int64_t{0}, val)); }

const float *InitCoeffsTable(const double a) {
  float *coeffs_table = new float[(kTableSize + 1) * 2];
  for (int i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_table[i * calnum2] = ((a + calnum2) * x - (a + calnum3)) * x * x + 1;
    x += 1.0;
    coeffs_table[i * calnum2 + 1] = ((a * x - calnum5 * a) * x + calnum8 * a) * x - calnum4 * a;
  }
  return coeffs_table;
}

const float *GetCoeffsTable(const bool use_keys_cubic) {
  if (use_keys_cubic) {
    static const float *coeffs_table = InitCoeffsTable(-0.5f);
    return coeffs_table;
  }
  static const float *coeffs_table = InitCoeffsTable(-0.75f);
  return coeffs_table;
}

template <typename Scaler, bool use_keys_cubic>
inline void GetWeightsAndIndices(const float scale, const int64_t out_loc, const int64_t limit,
                                 WeightsAndIndices *out) {
  const Scaler scaler;
  const float in_loc_f = scaler(out_loc, scale);
  const int64_t in_loc = std::floor(in_loc_f);
  const float delta = in_loc_f - in_loc;
  const int64_t offset = lrintf(delta * kTableSize);
  const float *coeffs_table = GetCoeffsTable(use_keys_cubic);
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

static void ComputeXWeightsAndIndices(const ResizerState &resizer_state, const bool half_pixel_centers_,
                                      std::vector<WeightsAndIndices> *x_wais) {
  CachedInterpolationCalculator calc;
  if (half_pixel_centers_) {
    for (int64_t x = 0; x < resizer_state.out_width; ++x) {
      GetWeightsAndIndices<HalfPixelScaler, true>(resizer_state.width_scale, x, resizer_state.in_width,
                                                  &(*x_wais)[static_cast<size_t>(x)]);
      auto &x_wai = (*x_wais)[static_cast<size_t>(x)];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2, x_wai.index_3);
    }
  } else {
    for (int64_t x = 0; x < resizer_state.out_width; ++x) {
      GetWeightsAndIndices<LegacyScaler, false>(resizer_state.width_scale, x, resizer_state.in_width,
                                                &(*x_wais)[static_cast<size_t>(x)]);
      auto &x_wai = (*x_wais)[static_cast<size_t>(x)];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2, x_wai.index_3);
    }
  }
}

template <typename T>
inline float Interpolate1D(const float weight_0, const float weight_1, const float weight_2, const float weight_3,
                           const T value_0, const T value_1, const T value_2, const T value_3) {
  return static_cast<float>(value_0) * weight_0 + static_cast<float>(value_1) * weight_1 +
         static_cast<float>(value_2) * weight_2 + static_cast<float>(value_3) * weight_3;
}

template <typename T>
static float ComputeYInterpolation(int which, const WeightsAndIndices &y_wai, const T *y_ptr_0, const T *y_ptr_1,
                                   const T *y_ptr_2, const T *y_ptr_3, const WeightsAndIndices &x_wai) {
  int x_index;  // w
  switch (which) {
    case 0:
      x_index = x_wai.index_0;
      break;
    case 1:
      x_index = x_wai.index_1;
      break;
    case caseid2:
      x_index = x_wai.index_2;
      break;
    default:
      x_index = x_wai.index_3;
      break;
  }
  return Interpolate1D<T>(y_wai.weight_0, y_wai.weight_1, y_wai.weight_2, y_wai.weight_3, y_ptr_0[x_index],
                          y_ptr_1[x_index], y_ptr_2[x_index], y_ptr_3[x_index]);
}

static float Compute_1D(const float values_[4], const float xw_0, const float xw_1, const float xw_2,
                        const float xw_3) {
  return Interpolate1D<float>(xw_0, xw_1, xw_2, xw_3, values_[0], values_[1], values_[2], values_[3]);
}

template <typename T1>
std::vector<float> CalSwitch(const WeightsAndIndices &x_wai, std::vector<float> cached_value, const ResizerState &RS,
                             const WeightsAndIndices &y_wai, const T1 *y_ptr_0, const T1 *y_ptr_1, const T1 *y_ptr_2,
                             const T1 *y_ptr_3) {
  switch (x_wai.advance) {
    case caseid3:
      cached_value[static_cast<size_t>(0)] = cached_value[static_cast<size_t>(1)];
      cached_value[static_cast<size_t>(1)] = cached_value[static_cast<size_t>(calnum2)];
      cached_value[static_cast<size_t>(calnum2)] = cached_value[static_cast<size_t>(calnum3)];
      break;
    case caseid2:
      cached_value[static_cast<size_t>(0)] = cached_value[static_cast<size_t>(calnum2)];
      cached_value[static_cast<size_t>(1)] = cached_value[static_cast<size_t>(calnum3)];
      break;
    case 1:
      cached_value[static_cast<size_t>(0)] = cached_value[static_cast<size_t>(calnum3)];
      break;
  }
  // Set the remaining '4-advance' values by computing.
  for (size_t i = x_wai.advance; i <= caseid3; i++) {
    cached_value[i] = ComputeYInterpolation(i, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
  }

  return cached_value;
}

template <typename T1, typename T2>
void ResizeBicubicCPUKernelMod::interpolate_with_caching(const T1 *input_data, const bool half_pixel_centers_,
                                                         T2 *output_data) {
  const ResizerState &RS = sta;
  std::vector<WeightsAndIndices> x_wais(RS.out_width);
  ComputeXWeightsAndIndices(RS, half_pixel_centers_, &x_wais);
  const int64_t in_row_width = RS.in_width * RS.in_height;    // hw
  const int64_t in_batch_width = RS.channels * in_row_width;  // chw
  const int64_t out_ch = RS.out_height * RS.channels;
  const int64_t out_chw = out_ch * RS.out_width;
  const int64_t out_hw = RS.out_height * RS.out_width;
  const size_t parallel_num = static_cast<size_t>(out_ch * RS.batch_size);
  std::vector<float> cached_value(calnum4, 0);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {  // nch
      const int64_t b = i / out_ch, c = i % out_ch / RS.out_height, y = i % RS.out_height;
      WeightsAndIndices y_wai;
      if (half_pixel_centers_) {
        GetWeightsAndIndices<HalfPixelScaler, true>(RS.height_scale, y, RS.in_height, &y_wai);
      } else {
        GetWeightsAndIndices<LegacyScaler, false>(RS.height_scale, y, RS.in_height, &y_wai);
      }
      const T1 *input_b_ptr = input_data + b * in_batch_width + c * in_row_width;
      T2 *output_y_ptr = output_data + b * out_chw + c * out_hw + y * RS.out_width;
      // Make pointers represent offsets of data in input_b_ptr
      const T1 *y_ptr_0 = input_b_ptr + y_wai.index_0 * RS.in_width;
      const T1 *y_ptr_1 = input_b_ptr + y_wai.index_1 * RS.in_width;
      const T1 *y_ptr_2 = input_b_ptr + y_wai.index_2 * RS.in_width;
      const T1 *y_ptr_3 = input_b_ptr + y_wai.index_3 * RS.in_width;
      for (int64_t x = 0; x < RS.out_width; ++x) {
        const WeightsAndIndices &x_wai = x_wais[static_cast<size_t>(x)];
        cached_value = CalSwitch(x_wai, cached_value, RS, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3);
        output_y_ptr[x] =
          Compute_1D(cached_value.data(), x_wai.weight_0, x_wai.weight_1, x_wai.weight_2, x_wai.weight_3);
      }
    }
  };
  ParallelLaunchAutoSearch(task, parallel_num, this, &parallel_search_info_);
  return;
}

bool ResizeBicubicCPUKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kResizeBicubicInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kResizeBicubicOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeBicubic>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  align_corners = kernel_ptr->get_align_corners();
  half_pixel_centers = kernel_ptr->get_half_pixel_centers();
  return true;
}

int ResizeBicubicCPUKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  shape0 = inputs[kIndex0]->GetDeviceShapeAdaptively();
  shape1 = inputs[kIndex1]->GetDeviceShapeAdaptively();

  sta.CalculateSize_kernel_node(inputs);
  return KRET_OK;
}

template <typename T1, typename T2>
bool ResizeBicubicCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto output_addr = static_cast<T2 *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size;
  if (memset_s(output_addr, output_size, 0, output_size) != EOK) {
    MS_EXCEPTION(ValueError) << "Memset Failed!";
  }
  auto input0_addr = static_cast<T1 *>(inputs[0]->addr);
  sta.CalculateSize_inputs(inputs);
  if (sta.out_height == sta.in_height && sta.out_width == sta.in_width) {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        output_addr[i] = static_cast<T2>(input0_addr[i]);
      }
    };
    ParallelLaunchAutoSearch(task, static_cast<size_t>(sta.bchw_size), this, &parallel_search_info_);
  } else {
    interpolate_with_caching(input0_addr, half_pixel_centers, output_addr);
  }

  return true;
}

std::vector<std::pair<KernelAttr, ResizeBicubicCPUKernelMod::ResizeBicubicFunc>> ResizeBicubicCPUKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<float16, float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<int8_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<uint8_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<int16_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<uint16_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<int32_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<int64_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<float, float>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &ResizeBicubicCPUKernelMod::LaunchKernel<double, float>}};

std::vector<KernelAttr> ResizeBicubicCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ResizeBicubicFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeBicubic, ResizeBicubicCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
