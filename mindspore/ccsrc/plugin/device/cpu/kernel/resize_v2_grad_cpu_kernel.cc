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

#include "plugin/device/cpu/kernel/resize_v2_grad_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include "mindspore/core/ops/grad/resize_v2_grad.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTwice = 2;
constexpr size_t kValueTwo = 2;
constexpr size_t kValueThree = 3;
constexpr size_t kValueFour = 4;
constexpr size_t kValueFive = 5;
constexpr size_t kValueEight = 8;
constexpr int64_t kCoeffSize = 4;
constexpr float kCubicCoeffA = -0.75;
}  // namespace

template <typename T>
static inline T AreaPixelComputeSourceIndex(T scale, size_t dst_index, bool align_corners, bool cubic) {
  if (align_corners) {
    return scale * static_cast<T>(dst_index);
  } else {
    T src_idx = static_cast<T>(static_cast<double>(scale) * (static_cast<double>(dst_index) + 0.5) - 0.5);
    return (!cubic && src_idx < T(0)) ? T(0) : src_idx;
  }
}

template <typename T>
static inline void ComputeSourceIndexAndLambda(size_t in_index[2], T lambda_data[2], T ratio,
                                               std::vector<size_t> io_data, bool align_corners) {
  size_t output_index = io_data[0];
  size_t input_size = io_data[1];
  size_t output_size = io_data[2];
  if (output_size == input_size) {
    in_index[0] = output_index;
    in_index[1] = output_index;
    lambda_data[0] = static_cast<T>(1);
    lambda_data[1] = static_cast<T>(0);
  } else {
    const T real_input_index = AreaPixelComputeSourceIndex<T>(ratio, output_index, align_corners, false);
    in_index[0] = LongToSize(static_cast<int64_t>(real_input_index));
    size_t offset = (in_index[0] < input_size - 1) ? 1 : 0;
    in_index[1] = in_index[0] + offset;
    lambda_data[1] = real_input_index - static_cast<T>(in_index[0]);
    lambda_data[0] = static_cast<T>(1.0) - lambda_data[1];
  }
}

template <typename T>
static inline T AreaPixelComputeScale(size_t input_size, size_t output_size, bool align_corners) {
  if (align_corners) {
    if (output_size > 1) {
      return static_cast<T>(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<T>(0);
    }
  } else {
    // output_size has been verified in Resize
    return (static_cast<T>(input_size) / output_size);
  }
}

static inline size_t ComputeSourceIndexByNearest(const float scale, size_t dst_index, size_t input_size) {
  const size_t src_index = std::min(static_cast<size_t>(floorf(dst_index * scale)), input_size - 1);
  return src_index;
}

static inline size_t NearestIdx(size_t output_index, size_t input_size, size_t output_size, float scale) {
  if (output_size == input_size) {
    return output_index;
  } else if (output_size == kTwice * input_size) {
    return output_index >> 1;
  } else {
    return ComputeSourceIndexByNearest(scale, output_index, input_size);
  }
}

template <typename T>
static inline T CubicConvolutionOne(T x, T A) {
  return ((A + static_cast<T>(kValueTwo)) * x - (A + static_cast<T>(kValueThree))) * x * x + static_cast<T>(1);
}

template <typename T>
static inline T CubicConvolutionTwo(T x, T A) {
  return ((A * x - static_cast<T>(kValueFive) * A) * x + static_cast<T>(kValueEight) * A) * x -
         static_cast<T>(kValueFour) * A;
}

template <typename T>
static inline void GetCubicCoefficients(T coeffs[4], T t, float cubic_coeff_t) {
  T A = static_cast<T>(cubic_coeff_t);

  T x1 = t;
  coeffs[kIndex0] = CubicConvolutionTwo<T>(x1 + static_cast<T>(1.0), A);
  coeffs[kIndex1] = CubicConvolutionOne<T>(x1, A);

  T x2 = static_cast<T>(1.0) - t;
  coeffs[kIndex2] = CubicConvolutionOne<T>(x2, A);
  coeffs[kIndex3] = CubicConvolutionTwo<T>(x2 + static_cast<T>(1.0), A);
}

bool ResizeV2GradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ResizeV2Grad>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();

  std::string coordinate_transformation_mode = kernel_ptr->get_coordinate_transformation_mode();
  if (coordinate_transformation_mode == "align_corners") align_corners_ = true;
  mode_ = kernel_ptr->get_mode();
  if (mode_ != "nearest" && mode_ != "linear" && mode_ != "cubic") {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', mode: " << mode_ << " not support now.";
    return false;
  }

  TypeId input_dtype = inputs[0]->GetDtype();
  if (mode_ != "nearest") {
    if (input_dtype != kNumberTypeFloat16 && input_dtype != kNumberTypeFloat32 && input_dtype != kNumberTypeFloat64) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' linear and cubic mode only support float16, float32, float64.";
      return false;
    }
  }
  sizes_dtype_ = inputs[kIndex3]->GetDtype();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int ResizeV2GradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }

  std::vector<int64_t> grad_shape = inputs[kIndex0]->GetShapeVector();
  batch_ = LongToSize(grad_shape[kIndex0]);
  channel_ = LongToSize(grad_shape[kIndex1]);
  in_height_ = LongToSize(grad_shape[kIndex2]);
  in_width_ = LongToSize(grad_shape[kIndex3]);
  if (in_height_ == 0) {
    MS_EXCEPTION(ValueError) << "For ResizeV2Grad, grads_height got 0, please check it.";
  }
  if (in_width_ == 0) {
    MS_EXCEPTION(ValueError) << "For ResizeV2Grad, grads_width got 0, please check it.";
  }

  return KRET_OK;
}

template <typename T>
bool ResizeV2GradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspaces,
                                            const std::vector<AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input_addr, false);
  MS_ERROR_IF_NULL_W_RET_VAL(output_addr, false);
  auto original_size = inputs[kIndex3];
  if (sizes_dtype_ == kNumberTypeInt64) {
    int64_t *sizes_data = reinterpret_cast<int64_t *>(original_size->addr);
    MS_ERROR_IF_NULL_W_RET_VAL(sizes_data, false);
    out_height_ = LongToSize(sizes_data[kIndex2]);
    out_width_ = LongToSize(sizes_data[kIndex3]);
  } else {
    int32_t *sizes_data = reinterpret_cast<int32_t *>(original_size->addr);
    MS_ERROR_IF_NULL_W_RET_VAL(sizes_data, false);
    std::vector<int64_t> sizes_v;
    sizes_v.push_back(static_cast<int64_t>(sizes_data[kIndex2]));
    sizes_v.push_back(static_cast<int64_t>(sizes_data[kIndex3]));
    out_height_ = LongToSize(sizes_v[kIndex0]);
    out_width_ = LongToSize(sizes_v[kIndex1]);
  }

  channels_ = batch_ * channel_;
  out_hw_size_ = out_height_ * out_width_;
  in_hw_size_ = in_height_ * in_width_;

  if (out_height_ == in_height_ && out_width_ == in_width_) {
    for (size_t i = 0; i < channels_ * in_hw_size_; ++i) {
      output_addr[i] = static_cast<T>(input_addr[i]);
    }
    return true;
  }

  if (mode_ == "nearest") {
    align_corners_ = false;
  }

  width_scale_ = AreaPixelComputeScale<float>(out_width_, in_width_, align_corners_);
  height_scale_ = AreaPixelComputeScale<float>(out_height_, in_height_, align_corners_);

  bool result = false;
  if (mode_ == "nearest") {
    result = LaunchKernelByNearest<T>(inputs, outputs);
  } else if (mode_ == "linear") {
    result = LaunchKernelByLinear<T>(inputs, outputs);
  } else {
    result = LaunchKernelByCubic<T>(inputs, outputs);
  }

  return result;
}

template <typename T>
bool ResizeV2GradCpuKernelMod::LaunchKernelByNearest(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  if (memset_s(output_addr, outputs[kIndex0]->size, 0, outputs[kIndex0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }

  auto task = [input_addr, output_addr, this](size_t start, size_t end) {
    for (size_t c = start; c < end; c++) {
      for (size_t iw = 0; iw < in_width_; iw++) {
        size_t ow = NearestIdx(iw, out_width_, in_width_, width_scale_);
        size_t output_offset = c * out_hw_size_ + ow;
        size_t input_offset = c * in_hw_size_ + iw;
        output_addr[output_offset] += input_addr[input_offset];
      }
    }
  };
  ParallelLaunchAutoSearch(task, channels_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename T>
bool ResizeV2GradCpuKernelMod::LaunchKernelByLinear(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  if (memset_s(output_addr, outputs[kIndex0]->size, 0, outputs[kIndex0]->size) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }

  auto task = [input_addr, output_addr, this](size_t start, size_t end) {
    size_t iw0, iw1;
    float w0lambda, w1lambda;
    size_t in_index[2];
    float lambda_data[2];
    for (size_t c = start; c < end; c++) {
      for (size_t ow = 0; ow < in_width_; ow++) {
        std::vector<size_t> io_data = {ow, out_width_, in_width_};
        ComputeSourceIndexAndLambda<float>(in_index, lambda_data, width_scale_, io_data, align_corners_);
        iw0 = in_index[0];
        iw1 = in_index[1];
        w0lambda = lambda_data[0];
        w1lambda = lambda_data[1];
        float grad_input_value = static_cast<float>(input_addr[c * in_hw_size_ + ow]);
        output_addr[c * out_width_ + iw0] += static_cast<T>(w0lambda * grad_input_value);
        output_addr[c * out_width_ + iw1] += static_cast<T>(w1lambda * grad_input_value);
      }
    }
  };
  ParallelLaunchAutoSearch(task, channels_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename T>
static void BicubicGeneralComputeHelper(T *in, float *out, std::vector<size_t> xyhw_data, float cubic_coeff_a,
                                        std::vector<float> t_xy_data, std::vector<int64_t> out_xy_data) {
  float x_coeffs[kCoeffSize];
  float y_coeffs[kCoeffSize];

  size_t channels = xyhw_data[kIndex0];
  size_t in_width = xyhw_data[kIndex1];
  size_t out_width = xyhw_data[kIndex2];
  size_t in_height = xyhw_data[kIndex3];
  size_t out_height = xyhw_data[kIndex4];
  size_t input_x = xyhw_data[kIndex5];
  size_t input_y = xyhw_data[kIndex6];
  int64_t output_x = out_xy_data[kIndex0];
  int64_t output_y = out_xy_data[kIndex1];

  GetCubicCoefficients<float>(x_coeffs, t_xy_data[0], cubic_coeff_a);
  GetCubicCoefficients<float>(y_coeffs, t_xy_data[1], cubic_coeff_a);

  for (size_t c = 0; c < channels; c++) {
    (void)c;
    T in_value = in[input_y * in_width + input_x];
    for (int64_t i = 0; i < kCoeffSize; i++) {
      for (int64_t j = 0; j < kCoeffSize; j++) {
        int64_t width = static_cast<int64_t>(out_width);
        int64_t height = static_cast<int64_t>(out_height);
        int64_t x = output_x - 1 + i;
        int64_t y = output_y - 1 + j;
        int64_t access_x = std::max(std::min(x, width - 1), static_cast<int64_t>(0));
        int64_t access_y = std::max(std::min(y, height - 1), static_cast<int64_t>(0));
        out[access_y * width + access_x] += static_cast<float>(in_value) * y_coeffs[j] * x_coeffs[i];
      }
    }
    in += in_width * in_height;
    out += out_width * out_height;
  }
}

template <typename T>
bool ResizeV2GradCpuKernelMod::LaunchKernelByCubic(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  size_t n = channels_ * out_hw_size_;
  float *out_temp = new float[n];

  if (memset_s(out_temp, sizeof(float) * n, 0, sizeof(float) * n) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output buffer memset failed.";
  }

  for (size_t input_y = 0; input_y < in_height_; input_y++) {
    for (size_t input_x = 0; input_x < in_width_; input_x++) {
      T *in = input_addr;
      float *out = out_temp;

      const float real_x = AreaPixelComputeSourceIndex<float>(width_scale_, input_x, align_corners_, true);
      int64_t output_x = static_cast<int64_t>(floorf(real_x));
      float t_x = real_x - output_x;

      const float real_y = AreaPixelComputeSourceIndex<float>(height_scale_, input_y, align_corners_, true);
      int64_t output_y = static_cast<int64_t>(floorf(real_y));
      float t_y = real_y - output_y;

      std::vector<size_t> xyhw_data = {channels_, in_width_, out_width_, in_height_, out_height_, input_x, input_y};

      std::vector<int64_t> out_xy_data = {output_x, output_y};

      std::vector<float> t_xy_data = {t_x, t_y};

      BicubicGeneralComputeHelper(in, out, xyhw_data, kCubicCoeffA, t_xy_data, out_xy_data);
    }
  }
  for (size_t i = 0; i < channels_ * out_hw_size_; ++i) {
    output_addr[i] = static_cast<T>(out_temp[i]);
  }
  return true;
}

#define RESIZE_V2_GRAD_CPU_REG(MS_T, T, SIZES_T) \
  KernelAttr()                                   \
    .AddInputAttr(MS_T)                          \
    .AddInputAttr(kNumberTypeFloat32)            \
    .AddInputAttr(kNumberTypeFloat32)            \
    .AddInputAttr(SIZES_T)                       \
    .AddOutputAttr(MS_T),                        \
    &ResizeV2GradCpuKernelMod::LaunchKernel<T>

using ResizeV2GradPair = std::pair<KernelAttr, ResizeV2GradCpuKernelMod::KernelRunFunc>;
const std::vector<ResizeV2GradPair> &ResizeV2GradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ResizeV2GradCpuKernelMod::KernelRunFunc>> func_list = {
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeFloat16, float16, kNumberTypeInt64)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeFloat32, float, kNumberTypeInt64)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeFloat64, double, kNumberTypeInt64)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeInt8, int8_t, kNumberTypeInt64)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeInt16, int16_t, kNumberTypeInt64)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeInt32, int32_t, kNumberTypeInt64)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeInt64, int64_t, kNumberTypeInt64)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeUInt8, uint8_t, kNumberTypeInt64)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeFloat16, float16, kNumberTypeInt32)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeFloat32, float, kNumberTypeInt32)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeFloat64, double, kNumberTypeInt32)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeInt8, int8_t, kNumberTypeInt32)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeInt16, int16_t, kNumberTypeInt32)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeInt32, int32_t, kNumberTypeInt32)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeInt64, int64_t, kNumberTypeInt32)},
    {RESIZE_V2_GRAD_CPU_REG(kNumberTypeUInt8, uint8_t, kNumberTypeInt32)},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ResizeV2Grad, ResizeV2GradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
