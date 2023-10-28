/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include "ir/tensor.h"
#include "runtime/stream.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "transform/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

bool AclnnKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "AclnnKernelMod Init";
  return true;
}

int AclnnKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  GetWorkSpaceInfo(inputs, outputs);
  return KernelMod::Resize(inputs, outputs);
}

bool AclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  return true;
}

void AclnnKernelMod::RunOp(const std::string &op_type, void *stream_ptr, const std::vector<KernelTensor *> &workspace) {
  if (workspace_size_list_.empty()) {
    RUN_OP_API(op_type, stream_ptr, nullptr, 0, executor_, after_launch_func_);
  } else {
    if (workspace.empty()) {
      MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
    }
    auto workspace_tensor = workspace[0];
    if (workspace_tensor->size() != workspace_size_list_[0]) {
      MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                        << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
    }
    RUN_OP_API(op_type, stream_ptr, workspace_tensor->device_ptr(), workspace_size_list_[0], executor_,
               after_launch_func_);
  }
}

void AclnnKernelMod::RunOpAsync(const std::string &op_type, void *stream_ptr,
                                const std::vector<KernelTensor *> &workspace) {
  if (workspace_size_list_.empty()) {
    RUN_OP_API_SYNC(op_type, stream_ptr, nullptr, 0, executor_);
  } else {
    if (workspace.empty()) {
      MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
    }
    const auto &workspace_tensor = workspace[0];
    if (workspace_tensor->size() != workspace_size_list_[0]) {
      MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                        << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
    }
    RUN_OP_API_SYNC(op_type, stream_ptr, workspace_tensor->device_ptr(), workspace_size_list_[0], executor_);
  }
}

void AclnnKernelMod::UpdateWorkspace(const uint64_t workspace_size) {
  auto real_workspace_size = static_cast<size_t>(workspace_size);
  if (real_workspace_size != 0) {
    std::vector<size_t> workspace_size_list = {real_workspace_size};
    SetWorkspaceSizeList(workspace_size_list);
  }
}

void AclnnKernelMod::ParseGenExecutor(const std::tuple<aclOpExecutor *, CallBackFunc> &args) {
  executor_ = std::get<0>(args);
  if (executor_ == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Please check op api's generate!";
  }
  after_launch_func_ = std::get<1>(args);
  if (after_launch_func_ == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Please check op api's call back func!";
  }
}

void AclnnKernelMod::SetInputsInfo(const std::vector<TypeId> &type_ids, const ShapeArray &shapes) {
  input_size_list_.resize(type_ids.size());
  if (type_ids.size() != shapes.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Aclnn kernel's input type size is not equal with shape size:" << shapes.size()
                               << " and type's size:" << type_ids.size();
  }
  input_size_list_.resize(type_ids.size(), 0);
}

void AclnnKernelMod::SetOutputsInfo(const std::vector<TypeId> &type_ids, const ShapeArray &shapes) {
  if (type_ids.size() != shapes.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Aclnn kernel's output type size is not equal with shape size:" << shapes.size()
                               << " and type's size:" << type_ids.size();
  }
  output_size_list_.resize(type_ids.size());
  for (size_t i = 0; i < type_ids.size(); i++) {
    size_t type_size = GetTypeByte(TypeIdToType(type_ids[i]));
    size_t tensor_size = shapes[i].empty()
                           ? type_size
                           : std::accumulate(shapes[i].begin(), shapes[i].end(), type_size, std::multiplies<size_t>());
    tensor_size = std::max(tensor_size, type_size);
    output_size_list_[i] = tensor_size;
  }
}

}  // namespace kernel
}  // namespace mindspore
