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

#include "plugin/device/cpu/kernel/unique_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
constexpr size_t kBucketSortThreshold = 100000;
void UniqueCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should be 1D, but got "
                      << input_shape.size() << "D";
  }
  input_size_ = input_shape[0];
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (common::AnfAlgo::HasNodeAttr(SORTED, kernel_node)) {
    sorted_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, SORTED);
  }
}

void UniqueCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  (void)workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
  (void)workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
  (void)workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
}

bool UniqueCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
                                const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int, int>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t, int64_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float, int>(inputs, workspace, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input should be float16, float32, int32, or int64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  if (!node_wpt_.expired()) {
    auto node_ = node_wpt_.lock();
    if (!node_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', node_wpt_(kernel_node) is expired. Error no: " << node_;
    }
    std::vector<size_t> out_shape;
    (void)out_shape.emplace_back(output_size_);
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(node_);
    std::vector<TypeId> dtypes(output_num);
    for (size_t i = 0; i < output_num; i++) {
      dtypes[i] = AnfAlgo::GetOutputDeviceDataType(node_, i);
    }
    common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {out_shape, common::AnfAlgo::GetOutputInferShape(node_, 1)},
                                                node_.get());
  }
  return true;
}

template <typename DataType, typename IndexType>
void UniqueCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  if (input_size_ == 0) {
    return;
  }
  if (inputs.size() < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the number of inputs should be greater than 0, but got: " << inputs.size();
  }
  if (workspace.size() < 3) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the number of workspaces should be greater than 2, but got: " << workspace.size();
  }
  if (outputs.size() < 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the number of outputs should be greater than 1, but got: " << outputs.size();
  }
  auto params = std::make_shared<UniqueParam<DataType, IndexType>>();
  params->input_ = reinterpret_cast<DataType *>(inputs[0]->addr);
  params->input_idx_ = reinterpret_cast<IndexType *>(workspace[0]->addr);
  params->workspace_ = reinterpret_cast<DataType *>(workspace[1]->addr);
  params->workspace_idx_ = reinterpret_cast<IndexType *>(workspace[2]->addr);
  params->output_ = reinterpret_cast<DataType *>(outputs[0]->addr);
  params->inverse_idx_ = reinterpret_cast<IndexType *>(outputs[1]->addr);
  params->input_size_ = input_size_;
  params->output_size_ = 0;

  params->thread_num_ = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  if (sorted_) {
    params->need_sort_ = true;
    if (input_size_ < kBucketSortThreshold) {
      Unique(params);
    } else {
      BucketUnique(params);
    }
  } else {
    params->need_sort_ = false;
    Unique(params);
  }
  output_size_ = static_cast<size_t>(params->output_size_);
}
}  // namespace kernel
}  // namespace mindspore
