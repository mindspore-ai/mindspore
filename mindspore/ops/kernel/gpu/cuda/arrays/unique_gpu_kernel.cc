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

#include "kernel/gpu/cuda/arrays/unique_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateUniqueKernelPtr(const std::string &kernel_name,
                                                                     const uint32_t &device_id) {
  return std::make_unique<cukernel::UniqueHelperGpuKernel<T, S>>(kernel_name, device_id);
}
using UniquePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, UniquePtrCreatorFunc>> kernel_attr = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<float, int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<half, int>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<double, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<int, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<int8_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<int16_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<uint8_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
   CreateUniqueKernelPtr<uint16_t, int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   CreateUniqueKernelPtr<int64_t, int64_t>}};
}  // namespace

bool UniqueGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto batch_rank = primitive_->HasAttr("batch_rank") ? GetValue<int64_t>(primitive_->GetAttr("batch_rank")) : 0;
  if (batch_rank < 0) {
    return false;
  }
  batch_rank_ = static_cast<size_t>(batch_rank);
  auto [is_match, index] = MatchKernelAttr(GetKernelAttrFromTensors(inputs, outputs), GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = kernel_attr[index].second(kernel_name_, device_id_);
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Invalid inputs is empty.";
    return false;
  }
  return true;
}

int UniqueGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  DestroyResource();
  ResetResource();
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<size_t> shape =
    std::vector<size_t>(inputs[0]->GetDeviceShapeVector().begin(), inputs[0]->GetDeviceShapeVector().end());
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (is_null_input_) {
    return KRET_OK;
  }
  input_shapes.emplace_back(inputs[0]->GetDeviceShapeVector());
  helper_ptr_->CalMemSize(input_shapes, output_shapes);
  InitSizeLists();
  return KRET_OK;
}

void UniqueGpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaStreamSynchronized failed");
  size_t output_num = outputs.size();
  for (size_t i = 0; i < output_num; ++i) {
    std::vector<int64_t> shape = outputs[i]->GetShapeVector();
    if (i == 0) {
      auto dyn_out = helper_ptr_->GetOutputTensorInfo();
      MS_EXCEPTION_IF_CHECK_FAIL(dyn_out.shapes.size() == 1 && dyn_out.shapes[0].size() == 1,
                                 "Unique output info error.");
      shape[0] = dyn_out.shapes[0][0];
    }
    outputs[i]->SetShapeVector(std::vector<int64_t>(shape.begin(), shape.end()));
    outputs[i]->set_size(LongToSize(std::accumulate(shape.begin(), shape.end(), UnitSizeInBytes(outputs[i]->dtype_id()),
                                                    std::multiplies<int64_t>())));
  }
}

std::vector<KernelAttr> UniqueGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UniquePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Unique, UniqueGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
