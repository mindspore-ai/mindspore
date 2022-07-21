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
#include "plugin/device/gpu/kernel/sparse_grad/sparse_add_grad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "mindspore/core/ops/grad/sparse_add_grad.h"

namespace mindspore {
namespace kernel {
constexpr auto kSparseAddGrad = "SparseAddGrad";
constexpr size_t kSparseAddGradInputsNum = 4;
constexpr size_t kSparseAddGradOutputsNum = 2;
constexpr size_t kSparseAddGradIndex0 = 0;
constexpr size_t kSparseAddGradIndex1 = 1;
constexpr size_t kSparseAddGradIndex2 = 2;

namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateSparseAddGradKernelPtr(const std::string &kernel_name,
                                                                            const uint32_t &device_id) {
  return std::make_unique<cukernel::SparseAddGradHelperGpuKernel<T, S>>(kernel_name, device_id);
}

#define GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(ms_index_type, ms_value_type, index_type, value_type) \
  {                                                                                               \
    KernelAttr()                                                                                  \
      .AddInputAttr(ms_value_type)                                                                \
      .AddInputAttr(ms_index_type)                                                                \
      .AddInputAttr(ms_index_type)                                                                \
      .AddInputAttr(ms_index_type)                                                                \
      .AddOutputAttr(ms_value_type)                                                               \
      .AddOutputAttr(ms_value_type),                                                              \
      CreateSparseAddGradKernelPtr<index_type, value_type>                                        \
  }

using SparseAddGradPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, SparseAddGradPtrCreatorFunc>> kernel_attr = {
  GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt8, int64_t, int8_t),
  GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt16, int64_t, int16_t),
  GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt32, int64_t, int32_t),
  GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t),
  GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat32, int64_t, float),
  GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeFloat64, int64_t, double),
  GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeComplex64, int64_t, cuComplex),
  GPU_SPARSE_ADD_GRAD_KERNEL_REGISTER(kNumberTypeInt64, kNumberTypeComplex128, int64_t, cuDoubleComplex),
};
}  // namespace

bool SparseAddGradGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

std::vector<KernelAttr> SparseAddGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseAddGradPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

bool SparseAddGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseAddGrad>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kSparseAddGradInputsNum || outputs.size() != kSparseAddGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kSparseAddGradInputsNum
                  << " and " << kSparseAddGradOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', support this kernel data type: " << kernel_attr;
    return false;
  }
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  is_need_retrieve_output_shape_ = true;
  return true;
}

int SparseAddGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  outputs_ = outputs;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_shapes),
                       [](KernelTensorPtr x) { return x->GetShapeVector(); });
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_shapes),
                       [](KernelTensorPtr x) { return x->GetShapeVector(); });

  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) != KRET_OK) {
    return KRET_UNKNOWN_SHAPE;
  }
  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  dx1_size_ = input_shapes.at(kSparseAddGradIndex1).at(kSparseAddGradIndex0);
  dx2_size_ = input_shapes.at(kSparseAddGradIndex2).at(kSparseAddGradIndex0);
  return KRET_OK;
}

void SparseAddGradGpuKernelMod::SyncData() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_),
                                     "For SparseAddGrad, cudaStreamSynchronized failed");
  std::vector<int64_t> dx1_shape = {dx1_size_};
  std::vector<int64_t> dx2_shape = {dx2_size_};

  outputs_[kSparseAddGradIndex0]->SetShapeVector(dx1_shape);
  outputs_[kSparseAddGradIndex1]->SetShapeVector(dx2_shape);
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SparseAddGrad,
                                 []() { return std::make_shared<SparseAddGradGpuKernelMod>(kSparseAddGrad); });
}  // namespace kernel
}  // namespace mindspore
