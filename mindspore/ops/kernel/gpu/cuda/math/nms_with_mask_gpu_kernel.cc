/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/math/nms_with_mask_gpu_kernel.h"
#include <map>
#include <utility>
#include <vector>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kOutputNum = 3;
}  // namespace

using KernelRunFunc = NMSWithMaskFwdGpuKernelMod::KernelRunFunc;
bool NMSWithMaskFwdGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << inputs.size()
                  << "input(s).";
  }
  if (outputs.size() != kOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs must be 3, but got " << outputs.size()
                  << "output(s).";
  }
  if (primitive_->HasAttr(kAttrIouThreshold)) {
    iou_value_ = GetValue<float>(primitive_->GetAttr(kAttrIouThreshold));
  }
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int NMSWithMaskFwdGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto in_shape = inputs[kIndex0]->GetShapeVector();
  if (!IsValidShape(in_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }

  num_input_ = LongToSizeClipNeg(in_shape[0]);  // Get N value in [N,5] data
  auto ceil_power_2 = NmsRoundUpPower2(num_input_);
  auto dtype_byte = abstract::TypeIdSize(inputs[kIndex0]->dtype_id());

  output_size_list_.clear();
  output_size_list_.push_back(num_input_ * dtype_byte * box_size_);
  output_size_list_.push_back(num_input_ * sizeof(int));
  output_size_list_.push_back(num_input_ * sizeof(bool));

  // N sized workspace arrs
  workspace_size_list_.clear();
  workspace_size_list_.push_back(ceil_power_2 * dtype_byte);               // data buff
  workspace_size_list_.push_back(ceil_power_2 * sizeof(int));              // index buff
  workspace_size_list_.push_back(num_input_ * num_input_ * sizeof(bool));  // mask list
  return KRET_OK;
}

template <typename T>
bool NMSWithMaskFwdGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  T *data_buff = GetDeviceAddress<T>(workspace, kIndex0);
  int *index_buff = GetDeviceAddress<int>(workspace, kIndex1);
  bool *row_mask = GetDeviceAddress<bool>(workspace, kIndex2);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  int *sel_idx = GetDeviceAddress<int>(outputs, kIndex1);
  bool *sel_boxes = GetDeviceAddress<bool>(outputs, kIndex2);
  cudaError_t status = cudaErrorNotReady;
  status = CalSort(num_input_, input, output, index_buff, data_buff, box_size_, device_id_,
                   reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = CalPreprocess(num_input_, sel_idx, sel_boxes, input, output, index_buff, box_size_, row_mask, device_id_,
                         reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = CalNms(num_input_, iou_value_, output, sel_boxes, box_size_, row_mask, device_id_,
                  reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &NMSWithMaskFwdGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeBool),
     &NMSWithMaskFwdGpuKernelMod::LaunchKernel<float>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NMSWithMask, NMSWithMaskFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
