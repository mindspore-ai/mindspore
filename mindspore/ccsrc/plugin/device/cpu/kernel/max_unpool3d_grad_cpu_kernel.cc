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
#include <cmath>
#include <map>
#include <utility>
#include <algorithm>
#include "plugin/device/cpu/kernel/max_unpool3d_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/grad/max_unpool3d_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxUnpool3DGradInputsNum = 3;
constexpr size_t kMaxUnpool3DGradOutputsNum = 1;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kInputIndex4 = 4;
}  // namespace

bool MaxUnpool3DGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxUnpool3DGrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  data_format_ = kernel_ptr->get_format();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxUnpool3DGradFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(ERROR) << "MaxUnpool3DGrad does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaxUnpool3DGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  grads_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  indices_shape_ = inputs[kIndex2]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

template <typename DATA_T>
void MaxUnpool3DGradCpuKernelMod::OutPutInitKernel(DATA_T *raw_output, size_t length) {
  for (size_t s = 0; s < length; s++) {
    raw_output[s] = (DATA_T)0;
  }
}

template <typename DATA_T, typename INDICES_T>
bool MaxUnpool3DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxUnpool3DGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxUnpool3DGradOutputsNum, kernel_name_);
  if (outputs[kInputIndex0]->size == 0) {
    MS_LOG(WARNING) << "MaxUnpool3DGrad output memory size should be greater than 0, but got 0.";
    return false;
  }
  auto *raw_grads = reinterpret_cast<DATA_T *>(inputs[kInputIndex1]->addr);
  auto *raw_indices = reinterpret_cast<INDICES_T *>(inputs[kInputIndex2]->addr);
  auto *raw_output = reinterpret_cast<DATA_T *>(outputs[kInputIndex0]->addr);
  auto num_batch = LongToSize(grads_shape_[kInputIndex0]);
  if (data_format_ == "NDHWC") {
    size_t odepth = LongToSize(grads_shape_[kInputIndex1]);
    size_t oheight = LongToSize(grads_shape_[kInputIndex2]);
    size_t owidth = LongToSize(grads_shape_[kInputIndex3]);
    size_t num_channels = LongToSize(grads_shape_[kInputIndex4]);
    size_t idepth = LongToSize(output_shape_[kInputIndex1]);
    size_t iheight = LongToSize(output_shape_[kInputIndex2]);
    size_t iwidth = LongToSize(output_shape_[kInputIndex3]);
    size_t length = num_batch * iheight * iwidth * idepth * num_channels;
    OutPutInitKernel<DATA_T>(raw_output, length);
    for (size_t n = 0; n < num_batch; n++) {
      size_t noutput_offset = n * num_channels * iwidth * iheight * idepth;
      size_t n_grads_offset = n * num_channels * owidth * oheight * odepth;
      DATA_T *output_p_k = raw_output + noutput_offset;
      DATA_T *grads_p_k = raw_grads + n_grads_offset;
      INDICES_T *ind_p_k = raw_indices + noutput_offset;
      size_t maxp;
      size_t ind_p_k_id;
      for (size_t k = 0; k < num_channels; k++) {
        for (size_t t = 0; t < idepth; t++) {
          for (size_t i = 0; i < iheight; i++) {
            for (size_t j = 0; j < iwidth; j++) {
              ind_p_k_id = t * iwidth * iheight * num_channels + i * iwidth * num_channels + j * num_channels + k;
              maxp = static_cast<size_t>(ind_p_k[ind_p_k_id]);
              if (ind_p_k[ind_p_k_id] < 0 || maxp >= owidth * oheight * odepth) {
                MS_LOG(EXCEPTION) << "MaxUnpool3DGrad: internal error, output_size D * H * W should "
                                     "be bigger than some indicis, now D * H * W is "
                                  << odepth * owidth * oheight << " and value of argmax is " << maxp << "."
                                  << std::endl;
              } else {
                output_p_k[ind_p_k_id] = grads_p_k[maxp * num_channels + k];
              }
            }
          }
        }
      }
    }
  } else {
    size_t odepth = LongToSize(grads_shape_[kInputIndex2]);
    size_t oheight = LongToSize(grads_shape_[kInputIndex3]);
    size_t owidth = LongToSize(grads_shape_[kInputIndex4]);
    size_t num_channels = LongToSize(grads_shape_[kInputIndex1]);
    size_t idepth = LongToSize(output_shape_[kInputIndex2]);
    size_t iheight = LongToSize(output_shape_[kInputIndex3]);
    size_t iwidth = LongToSize(output_shape_[kInputIndex4]);
    size_t length = num_batch * iheight * iwidth * idepth * num_channels;
    OutPutInitKernel<DATA_T>(raw_output, length);
    for (size_t n = 0; n < num_batch; n++) {
      size_t noutput_offset = n * num_channels * iwidth * iheight * idepth;
      size_t n_grads_offset = n * num_channels * owidth * oheight * odepth;
      size_t k = 0;
      for (k = 0; k < num_channels; k++) {
        size_t final_output_offset = noutput_offset + k * iwidth * iheight * idepth;
        size_t final_grads_offset = n_grads_offset + k * owidth * oheight * odepth;
        DATA_T *output_p_k = raw_output + final_output_offset;
        DATA_T *grads_p_k = raw_grads + final_grads_offset;
        INDICES_T *ind_p_k = raw_indices + final_output_offset;
        size_t maxp;
        size_t ind_p_k_id;
        for (size_t t = 0; t < idepth; t++) {
          for (size_t i = 0; i < iheight; i++) {
            for (size_t j = 0; j < iwidth; j++) {
              ind_p_k_id = t * iheight * iwidth + i * iwidth + j;
              maxp = static_cast<size_t>(ind_p_k[ind_p_k_id]);
              if (ind_p_k[ind_p_k_id] < 0 || maxp >= owidth * oheight * odepth) {
                MS_LOG(EXCEPTION) << "MaxUnpool3DGrad: internal error, output_size D * H * W should "
                                     "be bigger than some indicis, now D * H * W is "
                                  << odepth * owidth * oheight << " and value of argmax is " << maxp << "."
                                  << std::endl;
              } else {
                output_p_k[ind_p_k_id] = grads_p_k[maxp];
              }
            }
          }
        }
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, MaxUnpool3DGradCpuKernelMod::MaxUnpool3DGradFunc>>
  MaxUnpool3DGradCpuKernelMod::func_list_ = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt8),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<uint8_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeUInt8)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeUInt8),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<uint8_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt16),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<uint16_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeUInt16)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeUInt16),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<uint16_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt32),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<uint32_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeUInt32)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeUInt32),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<uint32_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeUInt64),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<uint64_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeUInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeUInt64),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<uint64_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt8),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<int8_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt8)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt8),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<int8_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt16),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<int16_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt16)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt16),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<int16_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<int32_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt32),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<int32_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeInt64),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<int64_t, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeInt64),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<int64_t, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<float16, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeFloat16)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeFloat16),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<float16, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<float, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeFloat32),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<float, int64_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeInt32)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<double, int32_t>},
                                             {KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeFloat64)
                                                .AddInputAttr(kNumberTypeInt64)
                                                .AddOutputAttr(kNumberTypeFloat64),
                                              &MaxUnpool3DGradCpuKernelMod::LaunchKernel<double, int64_t>}};

std::vector<KernelAttr> MaxUnpool3DGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxUnpool3DGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxUnpool3DGrad, MaxUnpool3DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
