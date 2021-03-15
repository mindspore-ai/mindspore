/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/unique_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
const size_t kUseBucketUniqueSize = 100000;
void UniqueCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  node_wpt_ = kernel_node;
  CheckParam(kernel_node);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_size_ = input_shape[0];
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

void UniqueCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
  workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
  workspace_size_list_.emplace_back(input_size_ * sizeof(int64_t));
}

bool UniqueCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> &workspace,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int, int>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t, int>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float, int>(inputs, workspace, outputs);
  }
  if (!node_wpt_.expired()) {
    auto node_ = node_wpt_.lock();
    if (!node_) {
      MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
    }
    std::vector<size_t> out_shape;
    out_shape.emplace_back(output_size_);
    std::vector<TypeId> dtypes;
    size_t output_num = AnfAlgo::GetOutputTensorNum(node_);
    for (size_t i = 0; i < output_num; i++) {
      dtypes.push_back(AnfAlgo::GetOutputInferDataType(node_, i));
    }
    AnfAlgo::SetOutputInferTypeAndShape(dtypes, {out_shape, AnfAlgo::GetOutputInferShape(node_, 1)}, node_.get());
  }
  return true;
}

template <typename DataType, typename IndexType>
void UniqueCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs) {
  if (input_size_ == 0) {
    return;
  }
  if (inputs.size() < 1) {
    MS_LOG(EXCEPTION) << "Input size should be large than 0";
  }
  if (workspace.size() < 3) {
    MS_LOG(EXCEPTION) << "workspace size should be large than 2";
  }
  if (outputs.size() < 2) {
    MS_LOG(EXCEPTION) << "Output size should be large than 1";
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
  params->need_sort_ = true;
  params->thread_num_ = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  if (input_size_ < kUseBucketUniqueSize) {
    Unique(params);
  } else {
    BucketUnique(params);
  }
  output_size_ = params->output_size_;
}

void UniqueCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but UniqueCPUKernel only support 1d.";
  }
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but UniqueCPUKernel needs 1 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but UniqueCPUKernel needs 2 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
