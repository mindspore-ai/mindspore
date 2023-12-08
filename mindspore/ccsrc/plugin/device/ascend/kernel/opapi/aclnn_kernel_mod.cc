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
#include "transform/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {

bool AclnnKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "AclnnKernelMod Init";
  return true;
}

int AclnnKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  GetWorkSpaceInfo(inputs, outputs);
  return ret;
}

bool AclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  return true;
}

void AclnnKernelMod::RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace) {
  if (workspace_size_list_.empty()) {
    RUN_OP_API_ASYNC(op_type_, nullptr, 0, executor_, stream_ptr, release_func_);
  } else {
    if (workspace.empty()) {
      MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
    }
    auto workspace_tensor = workspace[0];
    if (workspace_tensor->size() != workspace_size_list_[0]) {
      MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                        << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
    }
    RUN_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor_, stream_ptr,
                     release_func_);
  }
}

void AclnnKernelMod::RunOpSync(void *stream_ptr, const std::vector<KernelTensor *> &workspace) {
  if (workspace_size_list_.empty()) {
    RUN_OP_API_SYNC(op_type_, nullptr, 0, executor_, stream_ptr);
  } else {
    if (workspace.empty()) {
      MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
    }
    const auto &workspace_tensor = workspace[0];
    if (workspace_tensor->size() != workspace_size_list_[0]) {
      MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                        << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
    }
    RUN_OP_API_SYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor_, stream_ptr);
  }
}

void AclnnKernelMod::UpdateWorkspace(const std::tuple<uint64_t, aclOpExecutor *, CallBackFunc> &args) {
  auto call_back_func = std::get<2>(args);
  if (call_back_func != nullptr) {
    call_back_func();
  }
  auto real_workspace_size = static_cast<size_t>(std::get<0>(args));
  if (real_workspace_size != 0) {
    std::vector<size_t> workspace_size_list = {real_workspace_size};
    SetWorkspaceSizeList(workspace_size_list);
  }
}

void AclnnKernelMod::ParseGenExecutor(const std::tuple<uint64_t, aclOpExecutor *, CallBackFunc> &args) {
  executor_ = std::get<1>(args);
  if (executor_ == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Please check op api's generate!";
  }
  release_func_ = std::get<2>(args);
}

void AclnnKernelMod::SetDTypes(const std::string &op_name) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(WARNING) << "Not find op:" << op_name << " in OpDef. The inputs/outputs types maybe empty.";
    return;
  }
  auto &args = op_def->args_;
  auto &returns = op_def->returns_;
  (void)std::transform(args.begin(), args.end(), std::back_inserter(inputs_dtypes_),
                       [](const mindspore::ops::OpInputArg &arg) { return arg.arg_dtype_; });
  (void)std::transform(returns.begin(), returns.end(), std::back_inserter(outputs_dtypes_),
                       [](const mindspore::ops::OpOutputArg &arg) { return arg.arg_dtype_; });
}

}  // namespace kernel
}  // namespace mindspore
