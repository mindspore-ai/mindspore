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

#include "plugin/device/gpu/kernel/arrays/embedding_lookup_gpu_kernel.h"
#include "utils/check_convert_utils.h"

namespace {
constexpr size_t kEmbeddingLookupInputsNum = 3;
constexpr size_t kEmbeddingLookupOutputsNum = 1;
}  // namespace
namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename S, typename G>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateEmbeddingLookupKernelPtr(const std::string &kernel_name,
                                                                              const uint32_t &device_id) {
  return std::make_unique<cukernel::EmbeddingLookupHelperGpuKernel<T, S, G>>(kernel_name, device_id);
}

using EmbeddingLookupPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, EmbeddingLookupPtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateEmbeddingLookupKernelPtr<double, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateEmbeddingLookupKernelPtr<double, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateEmbeddingLookupKernelPtr<float, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateEmbeddingLookupKernelPtr<float, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateEmbeddingLookupKernelPtr<float, int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateEmbeddingLookupKernelPtr<int32_t, int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateEmbeddingLookupKernelPtr<half, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateEmbeddingLookupKernelPtr<half, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   CreateEmbeddingLookupKernelPtr<bool, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   CreateEmbeddingLookupKernelPtr<bool, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateEmbeddingLookupKernelPtr<int, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateEmbeddingLookupKernelPtr<int, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   CreateEmbeddingLookupKernelPtr<int16_t, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   CreateEmbeddingLookupKernelPtr<int16_t, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   CreateEmbeddingLookupKernelPtr<int8_t, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   CreateEmbeddingLookupKernelPtr<int8_t, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   CreateEmbeddingLookupKernelPtr<uint8_t, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   CreateEmbeddingLookupKernelPtr<uint8_t, int64_t, int64_t>},
};
}  // namespace

bool EmbeddingLookupGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool EmbeddingLookupGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::EmbeddingLookup>(base_operator);
  kernel_name_ = kernel_ptr->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_ptr;
    return false;
  }

  if (inputs.size() != kEmbeddingLookupInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << inputs.size();
  }

  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));

  int ret = Resize(kernel_ptr, inputs, outputs);
  if (ret == KRET_RESIZE_FAILED) {
    return false;
  }

  return true;
}

int EmbeddingLookupGpuKernelMod::Resize(
  const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
  const std::vector<KernelTensorPtr> &outputs,
  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {  // check input size
  if (inputs.size() != kEmbeddingLookupInputsNum || outputs.size() != kEmbeddingLookupOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kEmbeddingLookupInputsNum
                  << " and " << kEmbeddingLookupOutputsNum << ", but got " << inputs.size() << " and "
                  << outputs.size();
    return KRET_RESIZE_FAILED;
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;

  std::vector<int64_t> input_params_shape = inputs[kIndex0]->GetShapeVector();
  (void)input_shapes.emplace_back(input_params_shape);
  std::vector<int64_t> input_indices_shape = inputs[kIndex1]->GetShapeVector();
  (void)input_shapes.emplace_back(input_indices_shape);
  std::vector<int64_t> input_offset_shape = inputs[kIndex2]->GetShapeVector();
  (void)input_shapes.emplace_back(input_offset_shape);
  std::vector<int64_t> output_shape = outputs[kIndex0]->GetShapeVector();
  (void)output_shapes.emplace_back(output_shape);

  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }

  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> EmbeddingLookupGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, EmbeddingLookupPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, EmbeddingLookup, EmbeddingLookupGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
