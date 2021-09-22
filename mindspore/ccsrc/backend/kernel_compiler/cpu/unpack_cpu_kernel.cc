/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/unpack_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUnpackInputsNum = 1;
constexpr size_t kUnpackOutputsMinNum = 1;
constexpr size_t kUnpackWorkspaceMinNum = 1;
}  // namespace

template <typename T>
void UnpackCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  int64_t axis_tmp = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (axis_tmp < 0) {
    axis_tmp += SizeToLong(input_shape.size());
  }
  output_num_ = LongToSize(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "num"));
  unstack_param_.num_ = SizeToInt(output_num_);
  unstack_param_.axis_ = LongToInt(axis_tmp);
  unstack_param_.pre_dims_ = 1;
  unstack_param_.axis_dim_ = 1;
  unstack_param_.after_dims_ = 1;

  for (size_t i = 0; i < input_shape.size(); i++) {
    if (i < IntToSize(unstack_param_.axis_)) {
      unstack_param_.pre_dims_ *= SizeToInt(input_shape[i]);
    } else if (i > IntToSize(unstack_param_.axis_)) {
      unstack_param_.after_dims_ *= SizeToInt(input_shape[i]);
    } else {
      unstack_param_.axis_dim_ = SizeToInt(input_shape[i]);
    }
  }
}

template <typename T>
void UnpackCPUKernel<T>::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  (void)workspace_size_list_.emplace_back(sizeof(T *) * output_num_);
}

template <typename T>
bool UnpackCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUnpackInputsNum, kernel_name_);
  if (outputs.size() < kUnpackOutputsMinNum || workspace.size() < kUnpackWorkspaceMinNum) {
    MS_LOG(EXCEPTION) << "unpack error output or workspace size.";
  }
  LaunchKernel(inputs, workspace, outputs);
  return true;
}

template <typename T>
void UnpackCPUKernel<T>::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  const void *input = reinterpret_cast<void *>(inputs[0]->addr);
  void **outputs_host = reinterpret_cast<void **>(workspace[0]->addr);
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs_host[i] = reinterpret_cast<T *>(outputs[i]->addr);
  }
  int data_size = SizeToInt(sizeof(T));
  Unstack(input, outputs_host, &unstack_param_, data_size);
}
}  // namespace kernel
}  // namespace mindspore
