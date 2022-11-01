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

#include "plugin/device/gpu/kernel/sparse/sparse_matrix_softmax_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool SparseMatrixSoftmaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t inputs_num = 5;
  constexpr size_t outputs_num = 5;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), outputs_num, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  index_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex4).first);

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  InitSizeLists();
  return true;
}

int SparseMatrixSoftmaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    MS_LOG(ERROR) << kernel_name_ << " reinit failed.";
    return ret;
  }

  input_dense_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                           inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  dense_shape_elements_ =
    std::accumulate(input_dense_shape_.begin(), input_dense_shape_.end(), 1, std::multiplies<size_t>());

  input_batch_pointers_ = std::vector<size_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                              inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  batch_pointers_elements_ =
    std::accumulate(input_batch_pointers_.begin(), input_batch_pointers_.end(), 1, std::multiplies<size_t>());

  input_row_pointers_ = std::vector<size_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                            inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  row_pointers_elements_ =
    std::accumulate(input_row_pointers_.begin(), input_row_pointers_.end(), 1, std::multiplies<size_t>()) - 1;

  input_col_indices_ = std::vector<size_t>(inputs.at(kIndex3)->GetDeviceShapeAdaptively().begin(),
                                           inputs.at(kIndex3)->GetDeviceShapeAdaptively().end());
  col_indices_elements_ =
    std::accumulate(input_col_indices_.begin(), input_col_indices_.end(), 1, std::multiplies<size_t>());

  input_values_ = std::vector<size_t>(inputs.at(kIndex4)->GetDeviceShapeAdaptively().begin(),
                                      inputs.at(kIndex4)->GetDeviceShapeAdaptively().end());
  values_elements_ = std::accumulate(input_values_.begin(), input_values_.end(), 1, std::multiplies<size_t>());

  return KRET_OK;
}

template <typename DataType, typename IndexType>
bool SparseMatrixSoftmaxGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  MS_EXCEPTION_IF_NULL(cuda_stream);
  IndexType *x_dense_shape = GetDeviceAddress<IndexType>(inputs, kIndex0);
  IndexType *x_batch_pointers = GetDeviceAddress<IndexType>(inputs, kIndex1);
  IndexType *x_row_pointers = GetDeviceAddress<IndexType>(inputs, kIndex2);
  IndexType *x_col_indices = GetDeviceAddress<IndexType>(inputs, kIndex3);
  DataType *x_values = GetDeviceAddress<DataType>(inputs, kIndex4);
  IndexType *y_dense_shape = GetDeviceAddress<IndexType>(outputs, kIndex0);
  IndexType *y_batch_pointers = GetDeviceAddress<IndexType>(outputs, kIndex1);
  IndexType *y_row_pointers = GetDeviceAddress<IndexType>(outputs, kIndex2);
  IndexType *y_col_indices = GetDeviceAddress<IndexType>(outputs, kIndex3);
  DataType *y_values = GetDeviceAddress<DataType>(outputs, kIndex4);

  bool is_nullptr = (x_dense_shape == nullptr) || (x_batch_pointers == nullptr) || (x_row_pointers == nullptr) ||
                    (x_col_indices == nullptr) || (x_values == nullptr) || (y_dense_shape == nullptr) ||
                    (y_batch_pointers == nullptr) || (y_row_pointers == nullptr) || (y_col_indices == nullptr) ||
                    (y_values == nullptr);
  if (is_nullptr) {
    MS_LOG(ERROR) << " NULL pointers. ";
    return false;
  }
  SparseMatrixSoftmax(dense_shape_elements_, batch_pointers_elements_, row_pointers_elements_, col_indices_elements_,
                      x_dense_shape, x_batch_pointers, x_row_pointers, x_col_indices, x_values, y_dense_shape,
                      y_batch_pointers, y_row_pointers, y_col_indices, y_values, device_id_, cuda_stream);
  return true;
}

std::vector<std::pair<KernelAttr, SparseMatrixSoftmaxGpuKernelMod::SparseMatrixSoftmaxLaunchFunc>>
  SparseMatrixSoftmaxGpuKernelMod::func_list_ = {{
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseMatrixSoftmaxGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseMatrixSoftmaxGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseMatrixSoftmaxGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseMatrixSoftmaxGpuKernelMod::LaunchKernel<double, int32_t>},
  }};

std::vector<KernelAttr> SparseMatrixSoftmaxGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseMatrixSoftmaxGpuKernelMod::SparseMatrixSoftmaxLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseMatrixSoftmax, SparseMatrixSoftmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
