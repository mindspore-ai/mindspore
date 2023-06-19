/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/max_pool_grad_with_argmax_v2_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/max_pool_grad_with_argmax_v2.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolGradWithArgmaxV2InputsNum = 3;
constexpr size_t kMaxPoolGradWithArgmaxV2OutputsNum = 1;
constexpr int64_t kIndexChannel = 1;
constexpr int64_t kIndexHeight = 2;
constexpr int64_t kIndexWidth = 3;
}  // namespace

bool MaxPoolGradWithArgmaxV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();

  x_dtype_ = inputs[kIndex0]->GetDtype();
  argmax_dtype_ = inputs[kIndex2]->GetDtype();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolGradWithArgmaxV2>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  ksize_list_ = kernel_ptr->get_kernel_size();
  strides_list_ = kernel_ptr->get_strides();
  pads_list_ = kernel_ptr->get_pads();
  dilation_list_ = kernel_ptr->get_dilation();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaxPoolGradWithArgmaxV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  x_shape_ = inputs[kIndex0]->GetShapeVector();
  grads_shape_ = inputs[kIndex1]->GetShapeVector();
  y_shape_ = outputs[kIndex0]->GetShapeVector();
  if (std::any_of(y_shape_.begin(), y_shape_.end(), [](int64_t dim) { return dim == 0; })) {
    MS_LOG(ERROR) << "The shape of output_y is invalid.";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

std::vector<int64_t> MaxPoolGradWithArgmaxV2CpuKernelMod::GetValidAttr(const std::vector<int64_t> &src_attr) const {
  if (src_attr.size() == kShape1dDims) {
    return {src_attr[kDim0], src_attr[kDim0]};
  } else if (src_attr.size() == kShape4dDims) {
    return {src_attr[kDim2], src_attr[kDim3]};
  } else {
    return {src_attr[kDim0], src_attr[kDim1]};
  }
}

template <typename DATA_T, typename INDICES_T>
bool MaxPoolGradWithArgmaxV2CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &,
                                                       const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolGradWithArgmaxV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolGradWithArgmaxV2OutputsNum, kernel_name_);
  auto input_grads = static_cast<DATA_T *>(inputs[kIndex1]->addr);
  auto input_argmax = static_cast<INDICES_T *>(inputs[kIndex2]->addr);
  auto output_y = static_cast<DATA_T *>(outputs[kIndex0]->addr);

  const int64_t grads_channel = grads_shape_.at(kIndexChannel);
  const int64_t grads_height = grads_shape_.at(kIndexHeight);
  const int64_t grads_width = grads_shape_.at(kIndexWidth);
  const int64_t y_height = y_shape_.at(kIndexHeight);
  const int64_t y_width = y_shape_.at(kIndexWidth);
  const int64_t y_nchw = std::accumulate(y_shape_.begin(), y_shape_.end(), 1, std::multiplies<int64_t>());

  auto valid_ksize_list = GetValidAttr(ksize_list_);
  auto valid_strides_list = GetValidAttr(strides_list_);
  auto valid_pads_list = GetValidAttr(pads_list_);
  auto valid_dilation_list = GetValidAttr(dilation_list_);
  const int64_t k_width = valid_ksize_list[kDim1];
  const int64_t k_height = valid_ksize_list[kDim0];
  const int64_t s_width = valid_strides_list[kDim1];
  const int64_t s_height = valid_strides_list[kDim0];
  const int64_t p_width = valid_pads_list[kDim1];
  const int64_t p_height = valid_pads_list[kDim0];
  const int64_t d_width = valid_dilation_list[kDim1];
  const int64_t d_height = valid_dilation_list[kDim0];

  // Init output_y
  auto init = [output_y](size_t start, size_t end) {
    const DATA_T zero = static_cast<DATA_T>(0);
    for (size_t i = start; i < end; ++i) {
      output_y[i] = zero;
    }
  };
  ParallelLaunchAutoSearch(init, y_nchw, this, &parallel_search_info_);

  // Run Parallel task
  auto task = [input_grads, input_argmax, output_y, &grads_channel, &grads_height, &grads_width, &y_height, &y_width,
               &k_height, &k_width, &s_height, &s_width, &p_height, &p_width, &d_height,
               &d_width](size_t start, size_t end) {
    for (int i = SizeToInt(start); i < SizeToInt(end); ++i) {
      const int pos_n = i / (grads_channel * y_height * y_width);
      const int pos_c = i / (y_height * y_width) % grads_channel;
      const int pos_h = i / y_width % y_height;
      const int pos_w = i % y_width;
      const int grads_start = pos_n * grads_channel * grads_height * grads_width;
      const int grads_stride = pos_c * grads_height * grads_width;
      for (int cur_grads_h = 0; cur_grads_h < grads_height; cur_grads_h++) {
        for (int cur_grads_w = 0; cur_grads_w < grads_width; cur_grads_w++) {
          int start_h = cur_grads_h * s_height - p_height;
          int start_w = cur_grads_w * s_width - p_width;
          int end_h = std::min<int>(start_h + (k_height - 1) * d_height + 1, y_height);
          int end_w = std::min<int>(start_w + (k_width - 1) * d_width + 1, y_width);
          if (start_h < 0) {
            start_h += ceil(-start_h / static_cast<double>(d_height)) * d_height;
          }
          if (start_w < 0) {
            start_w += ceil(-start_w / static_cast<double>(d_width)) * d_width;
          }
          if ((pos_h >= start_h && pos_h < end_h) && (pos_w >= start_w && pos_w < end_w)) {
            if (input_argmax[grads_start + grads_stride + cur_grads_h * grads_width + cur_grads_w] ==
                pos_h * y_width + pos_w) {
              output_y[i] += input_grads[grads_start + grads_stride + cur_grads_h * grads_width + cur_grads_w];
            }
          }
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, y_nchw, this, &parallel_search_info_);
  return true;
}

#define ADD_KERNEL(x_dtype, shape_dtype, x_type, shape_type)                 \
  {                                                                          \
    KernelAttr()                                                             \
      .AddInputAttr(kNumberType##x_dtype)                                    \
      .AddInputAttr(kNumberType##x_dtype)                                    \
      .AddInputAttr(kNumberType##shape_dtype)                                \
      .AddOutputAttr(kNumberType##x_dtype),                                  \
      &MaxPoolGradWithArgmaxV2CpuKernelMod::LaunchKernel<x_type, shape_type> \
  }

std::vector<std::pair<KernelAttr, MaxPoolGradWithArgmaxV2CpuKernelMod::MaxPoolGradWithArgmaxV2Func>>
  MaxPoolGradWithArgmaxV2CpuKernelMod::func_list_ = {
    ADD_KERNEL(Float16, Int32, float16, int32_t), ADD_KERNEL(Float32, Int32, float, int32_t),
    ADD_KERNEL(Float64, Int32, double, int32_t),  ADD_KERNEL(Int8, Int32, int8_t, int32_t),
    ADD_KERNEL(Int16, Int32, int16_t, int32_t),   ADD_KERNEL(Int32, Int32, int32_t, int32_t),
    ADD_KERNEL(Int64, Int32, int64_t, int32_t),   ADD_KERNEL(UInt8, Int32, uint8_t, int32_t),
    ADD_KERNEL(UInt16, Int32, uint16_t, int32_t), ADD_KERNEL(UInt32, Int32, uint32_t, int32_t),
    ADD_KERNEL(UInt64, Int32, uint64_t, int32_t), ADD_KERNEL(Float16, Int64, float16, int64_t),
    ADD_KERNEL(Float32, Int64, float, int64_t),   ADD_KERNEL(Float64, Int64, double, int64_t),
    ADD_KERNEL(Int8, Int64, int8_t, int64_t),     ADD_KERNEL(Int16, Int64, int16_t, int64_t),
    ADD_KERNEL(Int32, Int64, int32_t, int64_t),   ADD_KERNEL(Int64, Int64, int64_t, int64_t),
    ADD_KERNEL(UInt8, Int64, uint8_t, int64_t),   ADD_KERNEL(UInt16, Int64, uint16_t, int64_t),
    ADD_KERNEL(UInt32, Int64, uint32_t, int64_t), ADD_KERNEL(UInt64, Int64, uint64_t, int64_t)};

std::vector<KernelAttr> MaxPoolGradWithArgmaxV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolGradWithArgmaxV2Func> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPoolGradWithArgmaxV2, MaxPoolGradWithArgmaxV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
