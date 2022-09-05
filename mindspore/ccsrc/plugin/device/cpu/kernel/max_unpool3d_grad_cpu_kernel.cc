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

void MaxUnpool3DGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kInputIndex0);
  grads_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kInputIndex1);
  indices_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kInputIndex2);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, kInputIndex0);
  data_format_ = common::AnfAlgo::GetNodeAttr<string>(kernel_node, FORMAT);

  if (AnfAlgo::IsShapesDynamic({input_shape_, grads_shape_, indices_shape_, output_shape_})) {
    return;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxUnpool3DGradFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MaxUnpool3DGrad does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
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
  auto node = node_wpt_.lock();
  if (!node) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
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
