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

#include "kernel/gpu/cuda/arrays/non_zero_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "kernel/gpu/cuda_impl/cuda_ops/non_zero_impl.cuh"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kNonZeroInputsNum = 1;
constexpr int kNonZeroOutputsNum = 1;
}  // namespace

bool NonZeroGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNonZeroInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNonZeroOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  data_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  index_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);
  return true;
}

void NonZeroGpuKernelMod::ResetResource() noexcept {
  real_output_size_ = 0;
  input_shape_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

int NonZeroGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  auto shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(input_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies{});
  if (input_size_ == 0) {
    return KRET_UNKNOWN_SHAPE;
  }

  workspace_size_list_.push_back(sizeof(size_t));
  output_size_list_.push_back(input_size_ * input_shape_.size() * index_size_);
  return KRET_OK;
}

template <typename DataType, typename IndexType>
bool NonZeroGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (input_size_ == 0) {
    return true;
  }
  if (inputs[kIndex0]->GetShapeVector().size() == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'x' must be greater than or equal to 1, but got 0.";
    return false;
  }
  auto input_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  auto output_size_ptr = GetDeviceAddress<size_t>(workspace, kIndex0);
  auto output_ptr = GetDeviceAddress<IndexType>(outputs, kIndex0);

  if (input_ptr == nullptr || output_size_ptr == nullptr || output_ptr == nullptr) {
    return false;
  }

  auto status = NonZero(input_ptr, output_ptr, output_size_ptr, input_shape_, input_size_, device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);

  // Update the final output size of NonZero.
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&real_output_size_, output_size_ptr, sizeof(size_t), cudaMemcpyDeviceToHost, cuda_stream_),
    "NonZero cudaMemcpyAsync failed.");
  if (cudaStreamQuery(cuda_stream_) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "For 'NonZero', cuda Stream Sync Failed.");
  }
  return true;
}

void NonZeroGpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_), "NonZero cudaStreamSynchronized failed");
  std::vector<int64_t> new_output_shape = {SizeToLong(real_output_size_), SizeToLong(input_shape_.size())};
  outputs[kIndex0]->SetShapeVector(new_output_shape);
  outputs[kIndex0]->set_size(real_output_size_ * input_shape_.size() * index_size_);
}

std::vector<std::pair<KernelAttr, NonZeroGpuKernelMod::NonZeroLaunchFunc>> NonZeroGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<bool, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroGpuKernelMod::LaunchKernel<uint64_t, int64_t>}};

std::vector<KernelAttr> NonZeroGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, NonZeroGpuKernelMod::NonZeroLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NonZero, NonZeroGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
