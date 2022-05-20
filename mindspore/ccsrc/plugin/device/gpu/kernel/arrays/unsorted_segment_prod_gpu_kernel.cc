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

#include "plugin/device/gpu/kernel/arrays/unsorted_segment_prod_gpu_kernel.h"

namespace mindspore {
namespace kernel {
#define UNSORTED_SEGMENT_PROD_GPU_REGISTER(T_DT, S_DT, T, S)              \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(S_DT).AddOutputAttr(T_DT), \
    &UnsortedSegmentProdGpuKernelMod::LaunchKernel<T, S>
#define UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(T_DT, S_DT, DT, T, S)                        \
  KernelAttr().AddInputAttr(T_DT).AddInputAttr(S_DT).AddInputAttr(DT).AddOutputAttr(T_DT), \
    &UnsortedSegmentProdGpuKernelMod::LaunchKernel<T, S>

void UnsortedSegmentProdGpuKernelMod::ResetResource() {
  input_dim0_ = 1;
  input_dim1_ = 1;
  output_dim0_ = 1;
  output_dim1_ = 1;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void UnsortedSegmentProdGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(input_dim0_ * input_dim1_ * data_unit_size_);
  input_size_list_.push_back(input_dim0_ * ids_unit_size_);
  output_size_list_.push_back(output_dim0_ * output_dim1_ * data_unit_size_);
}

bool UnsortedSegmentProdGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Got empty inputs or outputs, which is invalid.";
    return false;
  }

  kernel_name_ = base_operator->name();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  ids_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).first);
  return true;
}

int UnsortedSegmentProdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  ResetResource();

  auto input_shapes = inputs[kIndex0]->GetDeviceShapeAdaptively();
  auto ids_shapes = inputs[kIndex1]->GetDeviceShapeAdaptively();
  auto output_shapes = outputs[kIndex0]->GetDeviceShapeAdaptively();

  auto axis = ids_shapes.size();
  for (size_t i = 0; i < input_shapes.size(); i++) {
    if (i < axis) {
      input_dim0_ *= input_shapes[i];
    } else {
      input_dim1_ *= input_shapes[i];
    }
  }

  output_dim0_ = output_shapes[0];
  for (size_t j = 1; j < output_shapes.size(); j++) {
    output_dim1_ *= output_shapes[j];
  }

  InitSizeLists();
  return KRET_OK;
}

template <typename T, typename S>
bool UnsortedSegmentProdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices_addr = GetDeviceAddress<S>(inputs, kIndex1);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);

  UnsortedSegmentProd(input_dim0_, input_dim1_, output_dim0_, output_dim1_, input_addr, indices_addr, output_addr,
                      reinterpret_cast<cudaStream_t>(stream_ptr), device_id_);
  return true;
}

std::vector<std::pair<KernelAttr, UnsortedSegmentProdGpuKernelMod::UnsortedSegmentProdFunc>>
  UnsortedSegmentProdGpuKernelMod::func_list_ = {
    {UNSORTED_SEGMENT_PROD_GPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, double, int)},
    {UNSORTED_SEGMENT_PROD_GPU_REGISTER(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
    {UNSORTED_SEGMENT_PROD_GPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, float, int)},
    {UNSORTED_SEGMENT_PROD_GPU_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
    {UNSORTED_SEGMENT_PROD_GPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, half, int)},
    {UNSORTED_SEGMENT_PROD_GPU_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, half, int64_t)},
    {UNSORTED_SEGMENT_PROD_GPU_REGISTER(kNumberTypeInt32, kNumberTypeInt32, int, int)},
    {UNSORTED_SEGMENT_PROD_GPU_REGISTER(kNumberTypeInt32, kNumberTypeInt64, int, int64_t)},
    {UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt32, double, int)},
    {UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt64, double, int)},
    {UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt32, float, int)},
    {UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt64, float, int)},
    {UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, half, int)},
    {UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt64, half, int)},
    {UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int, int)},
    {UNSORTED_SEGMENT_PROD_GPU_DY_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt64, int, int)}};

std::vector<KernelAttr> UnsortedSegmentProdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UnsortedSegmentProdFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UnsortedSegmentProd, UnsortedSegmentProdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
