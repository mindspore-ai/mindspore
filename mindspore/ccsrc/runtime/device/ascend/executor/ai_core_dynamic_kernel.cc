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

#include "runtime/device/ascend/executor/ai_core_dynamic_kernel.h"

#include <algorithm>
#include "framework/common/debug/log.h"
#include "utils/log_adapter.h"
#include "register/op_tiling.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "runtime/device/ascend/executor/tiling/op_tiling_adapter.h"
#include "common/trans.h"
#include "backend/kernel_compiler/tbe/tbe_utils.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace device {
namespace ascend {
AiCoreDynamicKernel::~AiCoreDynamicKernel() {
  if (tiling_data_ptr_ != nullptr) {
    auto ret = rtFree(tiling_data_ptr_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "rtFree tiling_data_ptr_ failed";
    }
  }
}

bool AiCoreDynamicKernel::NeedSkipExecute(const CNodePtr &cnode) {
  // Skip run ReduceSum when axis is a Empty Tensor
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  if (op_name != "ReduceSum") {
    return false;
  }

  const size_t axes_index = 1;
  if (cnode->inputs().size() <= axes_index + 1) {
    return false;
  }
  auto input_axes = cnode->input(axes_index + 1);
  auto axes_abs = input_axes->abstract();
  MS_EXCEPTION_IF_NULL(axes_abs);
  auto axes_shape = AnfAlgo::GetInputDeviceShape(cnode, axes_index);
  if (axes_abs->isa<abstract::AbstractTensor>()) {
    if (std::any_of(axes_shape.begin(), axes_shape.end(), [](ssize_t shape) { return shape == 0; })) {
      return true;
    }
  }
  return false;
}

void AiCoreDynamicKernel::Execute() {
  if (stream_ == nullptr) {
    MS_LOG(EXCEPTION) << "stream_ptr should not be nullptr.";
  }
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  auto node_info = cnode->fullname_with_scope();
  MS_LOG(INFO) << "Start Execute node:" << node_info;

  if (NeedSkipExecute(cnode)) {
    // Skip reduce if axis is a empty Tensor (shape = 0)
    MS_LOG(INFO) << "The node " << node_info << "Need Skip.";
    rtError_t status = rtMemcpyAsync(kernel_outputs_[0]->addr, kernel_inputs_[0]->size, kernel_inputs_[0]->addr,
                                     kernel_inputs_[0]->size, RT_MEMCPY_DEVICE_TO_DEVICE, stream_);
    if (status != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "rtMemcpyAsync failed for " << node_info;
    }
    std::vector<TypeId> dtypes{AnfAlgo::GetOutputInferDataType(cnode, 0)};
    AnfAlgo::SetOutputInferTypeAndShape(dtypes, {AnfAlgo::GetInputDeviceShape(cnode, 0)}, cnode.get());

    MS_LOG(INFO) << "Execute node:" << cnode->fullname_with_scope() << " success.";
    return;
  }

  rtL2Ctrl_t *l2ctrl = nullptr;
  auto args_size = static_cast<uint32_t>(UlongToUint(sizeof(void *)) * runtime_args_.size());
  if (handle_ != nullptr) {
    const auto dev_func =
      origin_key_.find("kernel0") != origin_key_.npos ? origin_key_ : origin_key_ + "_" + std::to_string(tiling_key_);
    const auto kernel_info = node_info + "/" + std::to_string(tiling_key_);
    if (RT_ERROR_NONE != rtKernelLaunchWithHandle(handle_, dev_func.c_str(), block_dim_, runtime_args_.data(),
                                                  args_size, l2ctrl, stream_, kernel_info.c_str())) {
      MS_LOG(EXCEPTION) << "Call runtime rtKernelLaunchWithHandle error. Node info: " << node_info;
    }
  } else {
    if (RT_ERROR_NONE != rtKernelLaunch(stub_func_, block_dim_, runtime_args_.data(), args_size, l2ctrl, stream_)) {
      MS_LOG(EXCEPTION) << "Call runtime rtKernelLaunch error. Node info: " << node_info;
    }
  }
  MS_LOG(INFO) << "Execute node:" << cnode->fullname_with_scope() << " success.";
}

void AiCoreDynamicKernel::ParseCompileJson() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::IsDynamicShape(cnode)) {
    return;
  }
  bool get_flag = true;
  TbeUtils::GetCompileInfo(cnode, &op_compile_info_, &get_flag);
  if (!get_flag) {
    MS_LOG(EXCEPTION) << "Get compile_info failed. The compile result of [" << cnode->fullname_with_scope()
                      << "] maybe not in the json file(kernel_meta/) or the file had been deleted.";
  }
  MS_LOG(INFO) << "Node: " << cnode->fullname_with_scope() << " get compile_info: " << op_compile_info_;
}

void AiCoreDynamicKernel::Initialize() {
  DynamicKernel::Initialize();
  ParseCompileJson();
}

void AiCoreDynamicKernel::UpdateArgs() {
  ComputeTiling();
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!CopyTilingToDevice()) {
    MS_LOG(EXCEPTION) << "Copy tiling to device failed. op name: " << cnode->fullname_with_scope();
  }

  AllocateWorkspace();
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  KernelLaunchInfo kernel_launch_info;
  KernelRuntime::GenLaunchArgs(*kernel_mod, cnode, &kernel_launch_info);

  runtime_args_.clear();
  kernel_inputs_ = kernel_launch_info.inputs_;
  kernel_outputs_ = kernel_launch_info.outputs_;

  (void)std::transform(std::begin(kernel_inputs_), std::end(kernel_inputs_), std::back_inserter(runtime_args_),
                       [](const AddressPtr &input) { return input->addr; });
  (void)std::transform(std::begin(kernel_outputs_), std::end(kernel_outputs_), std::back_inserter(runtime_args_),
                       [](const AddressPtr &output) { return output->addr; });
  // Update workspace
  if (!workspace_addr_.empty()) {
    (void)std::transform(std::begin(workspace_addr_), std::end(workspace_addr_), std::back_inserter(runtime_args_),
                         [](const DeviceAddressPtr &address_ptr) { return address_ptr->GetMutablePtr(); });
  }

  if (is_dynamic_shape_ && !tiling_data_.empty() && tiling_data_ptr_ != nullptr) {
    runtime_args_.push_back(tiling_data_ptr_);
  }
}

void AiCoreDynamicKernel::ComputeTiling() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Start compute tiling of: " << cnode->fullname_with_scope();
  // start compute tiling
  optiling::utils::OpRunInfo op_run_info_v2(-1, true, 0);
  tiling::OpTilingCalculateAdapter converter;
  ge::ComputeGraphPtr ge_graph = std::make_shared<ge::ComputeGraph>("default");
  auto ge_node = converter.AnfNodeToGeNodeAdapter(cnode, &ge_graph, depend_tensor_map_, op_compile_info_);
  (void)optiling::OpParaCalculateV2(ge_node, op_run_info_v2);

  block_dim_ = op_run_info_v2.GetBlockDim();
  op_run_info_v2.GetAllWorkspaces(workspaces_size_);
  tiling_data_ = op_run_info_v2.GetAllTilingData().str();
  tiling_key_ = op_run_info_v2.GetTilingKey();
}

void AiCoreDynamicKernel::AllocateWorkspace() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = KernelRuntimeManager::Instance().GetSingleKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);

  workspace_addr_.clear();
  for (auto size : workspaces_size_) {
    auto device_address_ptr = std::make_shared<AscendDeviceAddress>(nullptr, size, kAscendDevice, device_id);
    device_address_ptr->set_is_ptr_persisted(true);
    auto device_ptr = runtime_instance->MallocMem(MemType::kDynamicMem, size, device_address_ptr);
    if (device_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "MallocMem from memory pool failed. Node info :" << cnode->fullname_with_scope();
    }
    workspace_addr_.emplace_back(device_address_ptr);
  }
}

bool AiCoreDynamicKernel::CopyTilingToDevice() {
  if (tiling_data_.size() > op_para_size_) {
    MS_LOG(EXCEPTION) << "Compute tiling size:" << tiling_data_.size()
                      << " larger than tbe build op_para_size:" << op_para_size_;
  }

  if (tiling_data_.empty() || tiling_data_ptr_ == nullptr) {
    MS_LOG(INFO) << "Tiling size is 0, skip aclrtMemcpyAsync";
    return true;
  }

  auto ret = aclrtMemcpyAsync(tiling_data_ptr_, tiling_data_.size(), tiling_data_.c_str(), tiling_data_.size(),
                              ACL_MEMCPY_HOST_TO_DEVICE, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Tiling aclrtMemcpyAsync failed, ret:" << ret;
  }
  return true;
}

void AiCoreDynamicKernel::PostExecute() {}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
