/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <utility>
#include <string>
#include <algorithm>
#include "kernel/gpu/cuda/arrays/bincount_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kOutputIndex0 = 0;

template <typename T>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateBincountKernelPtr(const std::string &kernel_name,
                                                                       const uint32_t &device_id) {
  return std::make_unique<cukernel::BincountHelperGpuKernel<T>>(kernel_name, device_id);
}
using BincountPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, BincountPtrCreatorFunc>> kernel_attr = {{KernelAttr()
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddInputAttr(kNumberTypeFloat32)
                                                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                                                 CreateBincountKernelPtr<float>},
                                                                                {KernelAttr()
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddInputAttr(kNumberTypeFloat64)
                                                                                   .AddOutputAttr(kNumberTypeFloat64),
                                                                                 CreateBincountKernelPtr<double>},
                                                                                {KernelAttr()
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddOutputAttr(kNumberTypeInt32),
                                                                                 CreateBincountKernelPtr<int32_t>},
                                                                                {KernelAttr()
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddInputAttr(kNumberTypeInt32)
                                                                                   .AddInputAttr(kNumberTypeInt64)
                                                                                   .AddOutputAttr(kNumberTypeInt64),
                                                                                 CreateBincountKernelPtr<int64_t>}};
}  // namespace

bool BincountGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool BincountGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  Resize(inputs, outputs);
  return true;
}

int BincountGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  for (const auto &output : outputs) {
    auto output_shape = output->GetShapeVector();
    if (!IsValidShape(output_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> array_shape = inputs[kInputIndex0]->GetShapeVector();
  std::vector<int64_t> size_shape = inputs[kInputIndex1]->GetShapeVector();
  std::vector<int64_t> weights_shape = inputs[kInputIndex2]->GetShapeVector();
  std::vector<int64_t> out_shape = outputs[kOutputIndex0]->GetShapeVector();

  input_shapes.emplace_back(array_shape);
  input_shapes.emplace_back(size_shape);
  input_shapes.emplace_back(weights_shape);
  output_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> BincountGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BincountPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Bincount, BincountGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
