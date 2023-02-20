/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/inc/cpu_context.h"
#include "cpu_kernel/common/cpu_node_def.h"
#include "cpu_kernel/common/device.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "proto/cpu_attr.pb.h"
#include "proto/cpu_node_def.pb.h"
#include "cpu_kernel/common/status.h"

namespace aicpu {
CpuKernelContext::CpuKernelContext(DeviceType type) {
  Device *device = new (std::nothrow) Device(type);
  if (device != nullptr) {
    device_.reset(device);
  }
}

uint32_t CpuKernelContext::Init(NodeDef *node_def) {
  KERNEL_CHECK_NULLPTR(node_def, KERNEL_STATUS_PARAM_INVALID, "Node def is null.")
  op_ = node_def->GetOpType();
  KERNEL_LOG_DEBUG("Construct the ctx of the op[%s] begin.", op_.c_str());
  for (int32_t i = 0; i < node_def->InputsSize(); i++) {
    auto input = node_def->MutableInputs(i);
    KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_PARAM_INVALID, "Get input[%d] tensor failed in op[%s].", i, op_.c_str())
    inputs_.emplace_back(std::move(input));
  }

  for (int32_t i = 0; i < node_def->OutputsSize(); i++) {
    auto output = node_def->MutableOutputs(i);
    KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output[%d] tensor failed in op[%s].", i, op_.c_str())
    outputs_.emplace_back(std::move(output));
  }

  auto attrMap = node_def->Attrs();
  for (auto iter = attrMap.begin(); iter != attrMap.end(); ++iter) {
    auto attr_value_ptr = iter->second;
    KERNEL_CHECK_NULLPTR(attr_value_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr[%s] failed in op[%s].",
                         iter->first.c_str(), op_.c_str())
    auto ret = attrs_.insert(std::make_pair(iter->first, std::move(attr_value_ptr)));
    if (!ret.second) {
      KERNEL_LOG_ERROR("Insert attr[%s] failed in op[%s].", iter->first.c_str(), op_.c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

  KERNEL_LOG_DEBUG("Construct the ctx of the op[%s] success.", op_.c_str());
  return KERNEL_STATUS_OK;
}

/*
 * get op type.
 * @return string: op type
 */
std::string CpuKernelContext::GetOpType() const { return op_; }

/*
 * get input tensor.
 * @return Tensor *: not null->success, null->failed
 */
Tensor *CpuKernelContext::Input(uint32_t index) const {
  if (index >= inputs_.size()) {
    KERNEL_LOG_WARN(
      "Input index[%u] should be less than input tensors total "
      "size[%zu].",
      index, inputs_.size());
    return nullptr;
  }

  return inputs_[index].get();
}

/*
 * get output tensor.
 * @return Tensor *: not null->success, null->failed
 */
Tensor *CpuKernelContext::Output(uint32_t index) const {
  if (index >= outputs_.size()) {
    KERNEL_LOG_WARN(
      "Output index[%u] should be less than output tensors total "
      "size[%zu].",
      index, outputs_.size());
    return nullptr;
  }

  return outputs_[index].get();
}

/*
 * get attr.
 * @return AttrValue *: not null->success, null->failed
 */
AttrValue *CpuKernelContext::GetAttr(std::string name) const {
  auto it = attrs_.find(name);
  if (it == attrs_.end()) {
    KERNEL_LOG_WARN("Attr[%s] is not exist.", name.c_str());
    return nullptr;
  }

  return (it->second).get();
}

/*
 * get input size.
 * @return uint32_t: input size
 */
uint32_t CpuKernelContext::GetInputsSize() const { return inputs_.size(); }

/*
 * get output size.
 * @return uint32_t: output size
 */
uint32_t CpuKernelContext::GetOutputsSize() const { return outputs_.size(); }
}  // namespace aicpu
