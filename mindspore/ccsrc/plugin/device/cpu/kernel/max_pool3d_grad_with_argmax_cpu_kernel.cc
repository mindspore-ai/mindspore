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
 * once_compute_thread_sizetributed under the License is
 * once_compute_thread_sizetributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

#include "plugin/device/cpu/kernel/max_pool3d_grad_with_argmax_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/max_pool3d_grad_with_argmax.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPool3DGradWithArgmaxInputNum = 3;
constexpr size_t kMaxPool3DGradWithArgmaxOutputsNum = 1;
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
const size_t kThree = 3;
const size_t kFour = 4;
const size_t DIM_SIZE_1 = 1;
const size_t DIM_SIZE_5 = 5;
}  // namespace

bool MaxPool3DGradWithArgmaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();

  x_dtype_ = inputs[kZero]->GetDtype();
  argmax_dtype_ = inputs[kTwo]->GetDtype();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPool3DGradWithArgmax>(base_operator);
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

int MaxPool3DGradWithArgmaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  x_shape_ = inputs[kZero]->GetDeviceShapeAdaptively();
  grads_shape_ = inputs[kOne]->GetDeviceShapeAdaptively();
  y_shape_ = outputs[kZero]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename DATA_T>
void MaxPool3DGradWithArgmaxCpuKernelMod::OutPutInitKernel(DATA_T *output, size_t length) {
  for (size_t s = 0; s < length; s++) {
    output[s] = (DATA_T)0;
  }
}

template <typename DATA_T, typename INDICES_T>
void MaxPool3DGradWithArgmaxCpuKernelMod::MaxPool3DGradWithArgmaxSingleCompute(
  DATA_T *input_grad, INDICES_T *input_argmax, DATA_T *output_y, size_t iD, size_t iH, size_t iW, size_t oD, size_t oH,
  size_t oW, size_t kD, size_t kH, size_t kW, size_t sD, size_t sH, size_t sW, size_t pD, size_t pH, size_t pW,
  size_t dD, size_t dH, size_t dW) {
  DATA_T *in_grad = input_grad;
  DATA_T *out_y = output_y;
  INDICES_T *argmax = input_argmax;

  /* calculate max points */
  size_t ti, i, j;
  for (ti = 0; ti < oD; ti++) {
    for (i = 0; i < oH; i++) {
      for (j = 0; j < oW; j++) {
        /* retrieve position of max */
        size_t index = ti * oH * oW + i * oW + j;
        size_t maxp = argmax[index];

        if (maxp != -kOne) {
          /* update gradient */
          out_y[maxp] += in_grad[index];
        }
      }
    }
  }
}

template <typename DATA_T>
bool MaxPool3DGradWithArgmaxCpuKernelMod::CheckIfLessOne(const std::vector<DATA_T> &inputs) {
  const size_t ksize = static_cast<size_t>(inputs[kZero]);
  const size_t strides = static_cast<size_t>(inputs[kOne]);
  const size_t dilation = static_cast<size_t>(inputs[kTwo]);
  if (ksize < kOne || strides < kOne || dilation < kOne) {
    MS_EXCEPTION(ValueError)
      << "for MaxPool3DGradWithArgmax, ksize, strides or dilation should be no less than one, but get ksize " << ksize
      << " , strides " << strides << ", dilation " << dilation << ".";
  } else {
    return true;
  }
}

template <typename DATA_T>
bool MaxPool3DGradWithArgmaxCpuKernelMod::CheckIfLessZero(const std::vector<DATA_T> &inputs) {
  const size_t width = static_cast<size_t>(inputs[kZero]);
  const size_t height = static_cast<size_t>(inputs[kOne]);
  const size_t depth = static_cast<size_t>(inputs[kTwo]);
  if (width < kZero || height < kZero || depth < kZero) {
    MS_EXCEPTION(ValueError) << "for MaxPool3DGradWithArgmax, pads should be no less than zero, but get pads [" << width
                             << ", " << height << ", " << depth << "].";
  } else {
    return true;
  }
}

void MaxPool3DGradWithArgmaxCpuKernelMod::CheckPadsValue(size_t k_width, size_t p_width, size_t k_height,
                                                         size_t p_height, size_t k_depth, size_t p_depth) {
  if (k_width / kTwo < p_width && k_height / kTwo < p_height && k_depth / kTwo < p_depth) {
    MS_EXCEPTION(ValueError)
      << "for " << kernel_name_
      << ", pads should be smaller than or equal to half of kernel size, but the depth, height, width of pads is ["
      << p_depth << ", " << p_height << ", " << p_width << "], the depth, height, width of kernel is [" << k_depth
      << ", " << k_height << ", " << k_width << "].";
  }
}

void MaxPool3DGradWithArgmaxCpuKernelMod::CheckDilationValue(size_t d_width, size_t in_width, size_t d_height,
                                                             size_t in_height, size_t d_depth, size_t in_depth) {
  if (d_width >= in_width && d_height >= in_height && d_depth >= in_depth) {
    MS_EXCEPTION(ValueError)
      << "for " << kernel_name_
      << ", dilation should be smaller than or equal to input, but the depth, height, width of dilation is [" << d_depth
      << ", " << d_height << ", " << d_width << "], while the depth,height,width of input is [" << in_depth << ", "
      << in_height << ", " << in_width << "].";
  }
}

template <typename DATA_T, typename INDICES_T>
bool MaxPool3DGradWithArgmaxCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &,
                                                       const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPool3DGradWithArgmaxInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPool3DGradWithArgmaxOutputsNum, kernel_name_);
  auto input_grads = reinterpret_cast<DATA_T *>(inputs[kOne]->addr);
  auto input_argmax = reinterpret_cast<INDICES_T *>(inputs[kTwo]->addr);
  auto output_y = reinterpret_cast<DATA_T *>(outputs[kZero]->addr);
  auto input_shape_vec = x_shape_;
  auto output_shape_vec = grads_shape_;
  const size_t in_width = input_shape_vec[kFour];
  const size_t in_height = input_shape_vec[kThree];
  const size_t in_depth = input_shape_vec[kTwo];
  const size_t in_channel = input_shape_vec[kOne];
  const size_t in_batch = input_shape_vec[kZero];
  const size_t out_width = output_shape_vec[kFour];
  const size_t out_height = output_shape_vec[kThree];
  const size_t out_depth = output_shape_vec[kTwo];
  const size_t in_stride = in_width * in_height * in_depth;
  const size_t out_stride = out_width * out_height * out_depth;
  const size_t batch = in_batch * in_channel;
  std::vector<int64_t> ksize_temp_list;
  if (ksize_list_.size() == DIM_SIZE_1) {
    ksize_temp_list.push_back(ksize_list_[kZero]);
    ksize_temp_list.push_back(ksize_list_[kZero]);
    ksize_temp_list.push_back(ksize_list_[kZero]);
  } else {
    ksize_temp_list.push_back(ksize_list_[kZero]);
    ksize_temp_list.push_back(ksize_list_[kOne]);
    ksize_temp_list.push_back(ksize_list_[kTwo]);
  }
  std::vector<int64_t> strides_temp_list;
  if (strides_list_.size() == DIM_SIZE_1) {
    strides_temp_list.push_back(strides_list_[kZero]);
    strides_temp_list.push_back(strides_list_[kZero]);
    strides_temp_list.push_back(strides_list_[kZero]);
  } else {
    strides_temp_list.push_back(strides_list_[kZero]);
    strides_temp_list.push_back(strides_list_[kOne]);
    strides_temp_list.push_back(strides_list_[kTwo]);
  }
  std::vector<int64_t> pads_temp_list;
  if (pads_list_.size() == DIM_SIZE_1) {
    pads_temp_list.push_back(pads_list_[kZero]);
    pads_temp_list.push_back(pads_list_[kZero]);
    pads_temp_list.push_back(pads_list_[kZero]);
  } else {
    pads_temp_list.push_back(pads_list_[kZero]);
    pads_temp_list.push_back(pads_list_[kOne]);
    pads_temp_list.push_back(pads_list_[kTwo]);
  }
  std::vector<int64_t> dilation_temp_list;
  if (dilation_list_.size() == DIM_SIZE_1) {
    dilation_temp_list.push_back(dilation_list_[kZero]);
    dilation_temp_list.push_back(dilation_list_[kZero]);
    dilation_temp_list.push_back(dilation_list_[kZero]);
  } else if (dilation_list_.size() == DIM_SIZE_5) {
    dilation_temp_list.push_back(dilation_list_[kTwo]);
    dilation_temp_list.push_back(dilation_list_[kThree]);
    dilation_temp_list.push_back(dilation_list_[kFour]);
  } else {
    dilation_temp_list.push_back(dilation_list_[kZero]);
    dilation_temp_list.push_back(dilation_list_[kOne]);
    dilation_temp_list.push_back(dilation_list_[kTwo]);
  }
  const size_t k_width = ksize_temp_list[kTwo];
  const size_t k_height = ksize_temp_list[kOne];
  const size_t k_depth = ksize_temp_list[kZero];
  const size_t s_width = strides_temp_list[kTwo];
  const size_t s_height = strides_temp_list[kOne];
  const size_t s_depth = strides_temp_list[kZero];
  const size_t p_width = pads_temp_list[kTwo];
  const size_t p_height = pads_temp_list[kOne];
  const size_t p_depth = pads_temp_list[kZero];
  const size_t d_width = dilation_temp_list[kTwo];
  const size_t d_height = dilation_temp_list[kOne];
  const size_t d_depth = dilation_temp_list[kZero];
  const size_t length = batch * out_stride;
  CheckPadsValue(k_width, p_width, k_height, p_height, k_depth, p_depth);
  CheckDilationValue(d_width, in_width, d_height, in_height, d_depth, in_depth);
  CheckIfLessOne(strides_temp_list);
  CheckIfLessOne(dilation_temp_list);
  CheckIfLessOne(ksize_temp_list);
  CheckIfLessZero(pads_temp_list);
  if (p_width * p_height * p_depth < kZero) {
    MS_EXCEPTION(ValueError) << "for " << kernel_name_
                             << ", pads should be no less than zero, but get p_width * p_height * p_depth = "
                             << p_width * p_height * p_depth;
  }  // attributes limitations
  OutPutInitKernel(output_y, length);
  for (size_t i = 0; i < batch; i++) {
    MaxPool3DGradWithArgmaxSingleCompute(input_grads + i * out_stride, input_argmax + i * out_stride,
                                         output_y + i * in_stride, in_depth, in_height, in_width, out_depth, out_height,
                                         out_width, k_depth, k_height, k_width, s_depth, s_height, s_width, p_depth,
                                         p_height, p_width, d_depth, d_height, d_width);
  }
  return true;
}

#define ADD_KERNEL(x_dtype, shape_dtype, x_type, shape_type)                 \
  {                                                                          \
    KernelAttr()                                                             \
      .AddInputAttr(kNumberType##x_dtype)                                    \
      .AddInputAttr(kNumberType##x_dtype)                                    \
      .AddInputAttr(kNumberType##shape_dtype)                                \
      .AddOutputAttr(kNumberType##x_dtype),                                  \
      &MaxPool3DGradWithArgmaxCpuKernelMod::LaunchKernel<x_type, shape_type> \
  }

std::vector<std::pair<KernelAttr, MaxPool3DGradWithArgmaxCpuKernelMod::MaxPool3DGradWithArgmaxFunc>>
  MaxPool3DGradWithArgmaxCpuKernelMod::func_list_ = {
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

std::vector<KernelAttr> MaxPool3DGradWithArgmaxCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPool3DGradWithArgmaxFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
