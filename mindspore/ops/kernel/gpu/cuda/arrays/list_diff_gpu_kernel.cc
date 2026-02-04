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

#include "kernel/gpu/cuda/arrays/list_diff_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateListDiffKernelPtr(const std::string &kernel_name,
                                                                       const uint32_t &device_id) {
  return std::make_unique<cukernel::ListDiffHelperGpuKernel<T, S>>(kernel_name, device_id);
}
using ListDiffPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, ListDiffPtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<uint8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<uint16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<int16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateListDiffKernelPtr<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<half, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<uint8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<uint16_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<int8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<int16_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateListDiffKernelPtr<int64_t, int32_t>}};
}  // namespace

bool ListDiffGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto [is_match, index] = MatchKernelAttr(GetKernelAttrFromTensors(inputs, outputs), GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = kernel_attr[index].second(kernel_name_, device_id_);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For 'ListDiff' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int ListDiffGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  input_shapes.emplace_back(inputs[kIndex0]->GetDeviceShapeVector());
  input_shapes.emplace_back(inputs[kIndex1]->GetDeviceShapeVector());
  helper_ptr_->CalMemSize(input_shapes, output_shapes);
  ResetResource();
  InitSizeLists();
  return KRET_OK;
}

void ListDiffGpuKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaStreamSynchronized failed");
  size_t output_num = outputs.size();
  auto dyn_out = helper_ptr_->GetOutputTensorInfo();
  auto num_out = dyn_out.shapes[kIndex0][kIndex0];
  std::vector<int64_t> shape = outputs[kIndex0]->GetShapeVector();
  shape[kIndex0] = num_out;
  for (size_t i = 0; i < output_num; ++i) {
    outputs[i]->SetShapeVector(std::vector<int64_t>(shape.begin(), shape.end()));
    outputs[i]->set_size(LongToSize(std::accumulate(shape.begin(), shape.end(), UnitSizeInBytes(outputs[i]->dtype_id()),
                                                    std::multiplies<int64_t>())));
  }
}

std::vector<KernelAttr> ListDiffGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ListDiffPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ListDiff, ListDiffGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
