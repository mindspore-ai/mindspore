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
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
const size_t kThree = 3;
const size_t DIM_SIZE_1 = 1;
const size_t DIM_SIZE_4 = 4;
}  // namespace

bool MaxPoolGradWithArgmaxV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();

  x_dtype_ = inputs[kZero]->GetDtype();
  argmax_dtype_ = inputs[kTwo]->GetDtype();

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

  x_shape_ = inputs[kZero]->GetDeviceShapeAdaptively();
  grads_shape_ = inputs[kOne]->GetDeviceShapeAdaptively();
  y_shape_ = outputs[kZero]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

std::vector<int64_t> MaxPoolGradWithArgmaxV2CpuKernelMod::GetValidAttr(const std::vector<int64_t> &src_attr) const {
  if (src_attr.size() == DIM_SIZE_1) {
    return {src_attr[kZero], src_attr[kZero]};
  } else if (src_attr.size() == DIM_SIZE_4) {
    return {src_attr[kTwo], src_attr[kThree]};
  } else {
    return {src_attr[kZero], src_attr[kOne]};
  }
}

template <typename DATA_T>
void MaxPoolGradWithArgmaxV2CpuKernelMod::OutPutInitKernel(DATA_T *output, size_t length) const {
  for (size_t s = 0; s < length; s++) {
    output[s] = (DATA_T)0;
  }
}

void MaxPoolGradWithArgmaxV2CpuKernelMod::CheckPadsValue(size_t k_width, size_t p_width, size_t k_height,
                                                         size_t p_height) const {
  if (k_width / kTwo < p_width && k_height / kTwo < p_height) {
    MS_EXCEPTION(ValueError)
      << "for " << kernel_name_
      << ", pads should be smaller than or equal to half of kernel size, but the height, width of pads is [" << p_height
      << ", " << p_width << "], the height, width of kernel size is [" << k_height << ", " << k_width << "].";
  }
}

void MaxPoolGradWithArgmaxV2CpuKernelMod::CheckDilationValue(size_t d_width, size_t in_width, size_t d_height,
                                                             size_t in_height) const {
  if (d_width >= in_width && d_height >= in_height) {
    MS_EXCEPTION(ValueError)
      << "for " << kernel_name_
      << ", dilation should be smaller than or equal to input, but the height, width of dilation is [" << d_height
      << ", " << d_width << "], while the height,width of input is [" << in_height << ", " << in_width << "].";
  }
}

template <typename DATA_T, typename INDICES_T>
void MaxPoolGradWithArgmaxV2CpuKernelMod::MaxPoolGradWithArgmaxV2SingleCompute(
  DATA_T *input_grad, INDICES_T *input_argmax, DATA_T *output_y, size_t iH, size_t iW, size_t oH, size_t oW, size_t kH,
  size_t kW, size_t sH, size_t sW, size_t pH, size_t pW, size_t dH, size_t dW) {
  DATA_T *in_grad = input_grad;
  INDICES_T *argmax = input_argmax;
  DATA_T *out_y = output_y;

  /* calculate max points */
  for (size_t i = 0; i < oH; i++) {
    for (size_t j = 0; j < oW; j++) {
      /* retrieve position of max */
      size_t index = i * oW + j;
      size_t maxp = static_cast<size_t>(argmax[index]);

      if (maxp != -kOne) {
        /* update gradient */
        out_y[maxp] += in_grad[index];
      }
    }
  }
}

template <typename DATA_T, typename INDICES_T>
bool MaxPoolGradWithArgmaxV2CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &,
                                                       const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolGradWithArgmaxV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolGradWithArgmaxV2OutputsNum, kernel_name_);
  auto input_grads = static_cast<DATA_T *>(inputs[kOne]->addr);
  auto input_argmax = static_cast<INDICES_T *>(inputs[kTwo]->addr);
  auto output_y = static_cast<DATA_T *>(outputs[kZero]->addr);
  auto input_shape_vec = x_shape_;
  auto output_shape_vec = grads_shape_;
  const size_t in_width = static_cast<size_t>(input_shape_vec[kThree]);
  const size_t in_height = static_cast<size_t>(input_shape_vec[kTwo]);
  const size_t in_channel = static_cast<size_t>(input_shape_vec[kOne]);
  const size_t in_batch = static_cast<size_t>(input_shape_vec[kZero]);
  const size_t out_width = static_cast<size_t>(output_shape_vec[kThree]);
  const size_t out_height = static_cast<size_t>(output_shape_vec[kTwo]);
  const size_t in_stride = in_width * in_height;
  const size_t out_stride = out_width * out_height;
  const size_t batch = in_batch * in_channel;

  auto valid_ksize_list = GetValidAttr(ksize_list_);
  auto valid_strides_list = GetValidAttr(strides_list_);
  auto valid_pads_list = GetValidAttr(pads_list_);
  auto valid_dilation_list = GetValidAttr(dilation_list_);
  const size_t k_width = static_cast<size_t>(valid_ksize_list[kOne]);
  const size_t k_height = static_cast<size_t>(valid_ksize_list[kZero]);
  const size_t s_width = static_cast<size_t>(valid_strides_list[kOne]);
  const size_t s_height = static_cast<size_t>(valid_strides_list[kZero]);
  const size_t p_width = static_cast<size_t>(valid_pads_list[kOne]);
  const size_t p_height = static_cast<size_t>(valid_pads_list[kZero]);
  const size_t d_width = static_cast<size_t>(valid_dilation_list[kOne]);
  const size_t d_height = static_cast<size_t>(valid_dilation_list[kZero]);
  const size_t length = batch * in_width * in_height;

  (void)CheckPadsValue(k_width, p_width, k_height, p_height);
  (void)CheckDilationValue(d_width, in_width, d_height, in_height);
  if (p_width * p_height < kZero) {
    MS_EXCEPTION(ValueError) << "for " << kernel_name_
                             << ", pads should be no less than zero, but get p_width * p_height * p_depth = "
                             << p_width * p_height;
  }  // attributes limitations
  (void)OutPutInitKernel(output_y, length);
  for (size_t i = 0; i < batch; i++) {
    MaxPoolGradWithArgmaxV2SingleCompute(input_grads + i * out_stride, input_argmax + i * out_stride,
                                         output_y + i * in_stride, in_height, in_width, out_height, out_width, k_height,
                                         k_width, s_height, s_width, p_height, p_width, d_height, d_width);
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
