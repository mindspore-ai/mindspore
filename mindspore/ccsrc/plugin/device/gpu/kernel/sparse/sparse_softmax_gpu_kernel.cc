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

#include "plugin/device/gpu/kernel/sparse/sparse_softmax_gpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr int64_t INDICES_DIMS = 2;
constexpr int64_t VALUES_DIMS = 1;
constexpr int64_t SHAPE_MIN_SIZE = 2;
bool SparseSoftmaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::SparseSoftmax>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  kernel_func_ = func_list_[index].second;
  indices_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  values_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).first);
  shape_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).first);
  output_unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  return true;
}

int SparseSoftmaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> indices_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> values_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                           inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> shape_shape = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  int64_t indices_dims = indices_shape.size();
  if (indices_dims != INDICES_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'indices' should be 2-D, but got " << indices_dims
                  << "-D.";
    return KRET_RESIZE_FAILED;
  }
  int64_t values_dims = values_shape.size();
  if (values_dims != VALUES_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'values' should be 1-D, but got " << values_dims
                  << "-D.";
    return KRET_RESIZE_FAILED;
  }
  int64_t shape_dims = shape_shape.size();
  if (shape_dims != VALUES_DIMS) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'shape' should be 1-D, but got " << shape_dims
                  << "-D.";
    return KRET_RESIZE_FAILED;
  }
  indices_elements_ = std::accumulate(indices_shape.begin(), indices_shape.end(), 1, std::multiplies<int64_t>());
  indice_number_ = indices_shape[0];
  indice_dims_ = indices_shape[1];
  values_elements_ = values_shape[0];
  shape_elements_ = shape_shape[0];
  if (shape_elements_ < SHAPE_MIN_SIZE) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the size of 'shape' should not be less than 2, but got "
                  << shape_elements_ << ".";
    return KRET_RESIZE_FAILED;
  }
  if (indice_number_ != values_elements_) {
    MS_LOG(ERROR) << "For " << kernel_name_ << " the indices size[0] must equal to values number " << values_elements_
                  << ", but got " << indice_number_ << ".";
    return KRET_RESIZE_FAILED;
  }
  if (indice_dims_ != shape_elements_) {
    MS_LOG(ERROR) << "For " << kernel_name_ << " the indices size[1] must equal to shape number " << shape_elements_
                  << ", but got " << indice_dims_ << ".";
    return KRET_RESIZE_FAILED;
  }
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool SparseSoftmaxGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) {
  int64_t *indices = GetDeviceAddress<int64_t>(inputs, kIndex0);
  T *values = GetDeviceAddress<T>(inputs, kIndex1);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  int32_t *reorder_device = GetDeviceAddress<int32_t>(workspace, kIndex0);
  int64_t *indice_to_num_device = GetDeviceAddress<int64_t>(workspace, kIndex1);
  CalSparseSoftmax(indices, values, output, reorder_device, indice_to_num_device, indice_dims_, values_elements_,
                   device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, SparseSoftmaxGpuKernelMod::SparseSoftmaxFunc>> SparseSoftmaxGpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &SparseSoftmaxGpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat64),
    &SparseSoftmaxGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> SparseSoftmaxGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseSoftmaxFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseSoftmax, SparseSoftmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
