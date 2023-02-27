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

#include "plugin/device/gpu/kernel/arrays/unique_consecutive_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "runtime/device/ms_device_shape_transfer.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename S>
std::unique_ptr<cukernel::UniqueConsecutiveHelperBase> CreateUniqueConsecutiveKernelPtr(const std::string &kernel_name,
                                                                                        const uint32_t &device_id) {
  return std::make_unique<cukernel::UniqueConsecutiveHelperGpuKernel<T, S>>(kernel_name, device_id);
}

using UniqueConsecutivePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::UniqueConsecutiveHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, UniqueConsecutivePtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateUniqueConsecutiveKernelPtr<float, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateUniqueConsecutiveKernelPtr<half, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   CreateUniqueConsecutiveKernelPtr<int, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   CreateUniqueConsecutiveKernelPtr<int64_t, int64_t>}};
}  // namespace

bool UniqueConsecutiveGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  stream_ptr_ = stream_ptr;
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

ValuePtr GetBaseOperatorAttr(const BaseOperatorPtr &op, const std::string &key) {
  ValuePtr attr = op->GetPrim()->GetAttr(key);
  if (attr == nullptr) {
    MS_LOG(EXCEPTION) << "The attr(" << key << ") of operator(" << op->name() << ") not exist";
  }
  return attr;
}

void UniqueConsecutiveGpuKernelMod::InitUniqueConsecutiveAttrs(const BaseOperatorPtr &base_operator,
                                                               const std::vector<KernelTensorPtr> &inputs) {
  // Get attrs from primitive.
  base_operator_ = base_operator;
  auto attr_idx = GetBaseOperatorAttr(base_operator_, "return_idx");
  auto attr_counts = GetBaseOperatorAttr(base_operator_, "return_counts");
  auto attr_axis = GetBaseOperatorAttr(base_operator_, "axis");
  return_idx_ = GetValue<bool>(attr_idx);
  return_counts_ = GetValue<bool>(attr_counts);
  constexpr int64_t kAxisIsNone = 1000;
  if (attr_axis->isa<None>() || GetValue<int64_t>(attr_axis) == kAxisIsNone) {
    is_flattend_ = true;
  } else {
    axis_ = GetValue<int64_t>(attr_axis);
    is_flattend_ = false;
  }
}

bool UniqueConsecutiveGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  InitUniqueConsecutiveAttrs(base_operator, inputs);
  // Initialize.
  inputs_ = inputs;
  outputs_ = outputs;
  auto [is_match, index] = MatchKernelAttr(GetKernelAttrFromTensors(inputs, outputs), GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = kernel_attr[index].second(kernel_name_, device_id_);
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Invalid inputs is empty.";
    return false;
  }
  is_need_retrieve_output_shape_ = true;
  return true;
}

int UniqueConsecutiveGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just
    // return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  DestroyResource();
  ResetResource();
  base_operator_ = base_operator;
  inputs_ = inputs;
  outputs_ = outputs;

  auto input_shape = inputs[0]->GetDeviceShapeAdaptively();
  int64_t dims = SizeToLong(input_shape.size());
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "For '" << base_operator_->name() << "', the 'axis' must be in the range [-" << dims << ","
                      << dims << "), but got " << axis_ << ".";
  }
  axis_ = axis_ >= 0 ? axis_ : axis_ + dims;
  is_flattend_ = input_shape.size() <= 1;

  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;

  // Check if shape contains 0.
  std::vector<size_t> shape =
    std::vector<size_t>(inputs[0]->GetDeviceShapeAdaptively().begin(), inputs[0]->GetDeviceShapeAdaptively().end());
  is_null_input_ = CHECK_SHAPE_NULL(shape, kernel_name_, "input");
  if (is_null_input_) {
    InitSizeLists();
    return true;
  }

  input_shapes.emplace_back(inputs[0]->GetDeviceShapeAdaptively());
  helper_ptr_->set_return_idx(return_idx_);
  helper_ptr_->set_return_counts(return_counts_);
  helper_ptr_->set_is_flattend(is_flattend_);
  helper_ptr_->set_axis(axis_);
  helper_ptr_->CalMemSize(input_shapes, output_shapes);
  InitSizeLists();
  return 0;
}

void UniqueConsecutiveGpuKernelMod::SyncData() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaStreamSynchronized failed");
  size_t output_num = outputs_.size();
  auto dyn_out = helper_ptr_->GetOutputTensorInfo();
  for (size_t i = 0; i < output_num; ++i) {
    outputs_[i]->SetShapeVector(std::vector<int64_t>(dyn_out.shapes[i].begin(), dyn_out.shapes[i].end()));
  }
}

std::vector<KernelAttr> UniqueConsecutiveGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UniqueConsecutivePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UniqueConsecutive, UniqueConsecutiveGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
