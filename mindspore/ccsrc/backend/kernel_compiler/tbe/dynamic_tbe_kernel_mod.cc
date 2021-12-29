/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/tbe/dynamic_tbe_kernel_mod.h"

#include <algorithm>
#include <stack>
#include "acl/acl_rt.h"
#include "utils/ms_context.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_runtime.h"
#include "backend/optimizer/common/helper.h"
#include "framework/common/debug/log.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "runtime/device/ascend/executor/tiling/op_tiling_adapter.h"
#include "utils/ms_device_shape_transfer.h"
#include "utils/utils.h"
#include "register/op_tiling.h"
#include "nlohmann/json.hpp"

namespace mindspore {
namespace kernel {
using TbeTaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::TbeTaskInfo>;
using tbe::KernelManager;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;

DynamicTbeKernelMod::DynamicTbeKernelMod(KernelPackPtr kernel_pack, const AnfNodePtr &anf_node_ptr)
    : TbeKernelMod(std::move(kernel_pack), anf_node_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  auto cnode = anf_node_ptr->cast<CNodePtr>();
  if (cnode != nullptr) {
    op_compile_info_ = ParseCompileJson(cnode);
  }
}

DynamicTbeKernelMod::~DynamicTbeKernelMod() {
  if (tiling_data_ptr_ != nullptr) {
    (void)rtFree(tiling_data_ptr_);
  }
}

void DynamicTbeKernelMod::InferOp() {
  if (AnfAlgo::IsDynamicShape(anf_node_.lock())) {
    auto node = anf_node_.lock();
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    need_skip_execute_ = NeedSkipExecute(cnode);
    if (need_skip_execute_) {
      std::vector<TypeId> dtypes{AnfAlgo::GetOutputInferDataType(cnode, 0)};
      AnfAlgo::SetOutputInferTypeAndShape(dtypes, {AnfAlgo::GetInputDeviceShape(cnode, 0)}, cnode.get());
    } else {
      KernelMod::InferShape();
    }
  }
}

void DynamicTbeKernelMod::InitOp() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (!AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The node is not dynamic shape: " << cnode->fullname_with_scope();
  }

  if (!atomic_clean_nodes_.empty()) {
    for (const auto &atomic_clean_node : atomic_clean_nodes_) {
      AnfAlgo::GetKernelMod(atomic_clean_node)->InitOp();
    }
  }

  if (need_skip_execute_) {
    return;
  }

  // gen FuncStub
  if (handle_ == nullptr) {
    auto func_stub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim_, true, &handle_, &origin_key_);
    if (func_stub != 1) {
      MS_LOG(EXCEPTION) << "GenFuncStub failed.";
    }
  }

  // start compute tiling
  MS_LOG(INFO) << "Start compute tiling of: " << cnode->fullname_with_scope();
  optiling::utils::OpRunInfo op_run_info_v2(-1, true, 0);
  device::tiling::OpTilingCalculateAdapter converter;
  ::ge::ComputeGraphPtr ge_graph = std::make_shared<::ge::ComputeGraph>("default");
  auto ge_node = converter.AnfNodeToGeNodeAdapter(cnode, &ge_graph, depend_tensor_map_, op_compile_info_);
  (void)optiling::OpParaCalculateV2(ge_node, op_run_info_v2);

  block_dim_ = op_run_info_v2.GetBlockDim();
  std::vector<int64_t> workspace_size_list;
  op_run_info_v2.GetAllWorkspaces(workspace_size_list);
  tiling_data_ = op_run_info_v2.GetAllTilingData().str();
  tiling_key_ = op_run_info_v2.GetTilingKey();

  workspace_size_list_.clear();
  workspace_size_list_.resize(workspace_size_list.size());
  std::transform(workspace_size_list.begin(), workspace_size_list.end(), workspace_size_list_.begin(),
                 [](int64_t size) { return static_cast<size_t>(size); });
}

std::string DynamicTbeKernelMod::ParseCompileJson(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);

  bool get_flag = true;
  std::string op_compile_info = "";
  TbeUtils::GetCompileInfo(cnode, &op_compile_info, &get_flag);
  if (!get_flag) {
    MS_LOG(EXCEPTION) << "Get compile_info failed. The compile result of [" << cnode->fullname_with_scope()
                      << "] maybe not in the json file(kernel_meta/) or the file had been deleted.";
  }
  MS_LOG(INFO) << "Node: " << cnode->fullname_with_scope() << " get compile_info: " << op_compile_info;
  return op_compile_info;
}

void DynamicTbeKernelMod::InitTilingDataPtr() {
  if (tiling_data_ptr_ != nullptr) {
    return;
  }
  auto kernel_json_info = kernel_pack_->kernel_json_info();
  auto op_para_size = kernel_json_info.op_para_size;
  if (op_para_size > 0) {
    auto ret = rtMalloc(&tiling_data_ptr_, op_para_size, RT_MEMORY_HBM);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "rtMalloc tiling data failed";
    }
  }
}

bool DynamicTbeKernelMod::CopyTilingToDevice(void *stream_ptr) {
  InitTilingDataPtr();
  MS_EXCEPTION_IF_NULL(kernel_pack_);
  auto kernel_json_info = kernel_pack_->kernel_json_info();

  auto op_para_size = kernel_json_info.op_para_size;
  if (tiling_data_.size() > op_para_size) {
    MS_LOG(EXCEPTION) << "Compute tiling size:" << tiling_data_.size()
                      << " larger than tbe build op_para_size:" << op_para_size;
  }

  if (tiling_data_.empty() || tiling_data_ptr_ == nullptr) {
    MS_LOG(INFO) << "Tiling size is 0, skip aclrtMemcpyAsync";
    return true;
  }
  // cppcheck-suppress unreadVariable
  auto lock = AscendKernelMod::LockRuntime();
  auto ret = aclrtMemcpyAsync(tiling_data_ptr_, op_para_size, tiling_data_.c_str(), tiling_data_.size(),
                              ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Tiling aclrtMemcpyAsync failed, ret:" << ret;
  }
  return true;
}

bool DynamicTbeKernelMod::NeedSkipExecute(const CNodePtr &cnode) {
  // Skip run ReduceSum when axis is a Empty Tensor
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  if (op_name != kReduceSumOpName) {
    return false;
  }

  const size_t axes_index = 1;
  if (cnode->inputs().size() <= axes_index + 1) {
    return false;
  }
  auto input_axes = cnode->input(axes_index + 1);
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(input_axes.get());
  auto axes_abs = input_axes->abstract()->Clone();
  MS_EXCEPTION_IF_NULL(axes_abs);
  auto axes_shape = AnfAlgo::GetInputDeviceShape(cnode, axes_index);
  if (axes_abs->isa<abstract::AbstractTensor>()) {
    if (std::any_of(axes_shape.begin(), axes_shape.end(), [](ssize_t shape) { return shape == 0; })) {
      return true;
    }
  }
  return false;
}

bool DynamicTbeKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr.";
    return false;
  }
  if (stream_ == nullptr) {
    stream_ = stream_ptr;
  }

  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "anfnode is not a cnode";
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  // is dynamic shape
  if (!AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The cnode is not dynamic shape:" << cnode->fullname_with_scope();
  }

  if (!atomic_clean_nodes_.empty()) {
    for (auto atomic_clean_node : atomic_clean_nodes_) {
      KernelLaunchInfo kernel_launch_info;
      auto kernel_mod = AnfAlgo::GetKernelMod(atomic_clean_node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      device::KernelRuntime::GenLaunchArgs(*kernel_mod, atomic_clean_node, &kernel_launch_info);
      auto atomic_inputs = kernel_launch_info.inputs_;
      std::vector<AddressPtr> atomic_outputs;
      std::vector<AddressPtr> atomic_workspace;
      kernel_mod->Launch(atomic_inputs, atomic_workspace, atomic_outputs, stream_ptr);
    }
  }

  // need skip, for reducesum empty input axis
  if (need_skip_execute_) {
    // Skip reduce if axis is a empty Tensor (shape = 0)
    MS_LOG(INFO) << "The node " << cnode->fullname_with_scope() << "Need Skip.";
    // cppcheck-suppress unreadVariable
    auto lock = AscendKernelMod::LockRuntime();
    rtError_t status = aclrtMemcpyAsync(outputs[0]->addr, inputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
    if (status != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "aclrtMemcpyAsync failed for " << cnode->fullname_with_scope();
    }

    MS_LOG(INFO) << "Execute node:" << cnode->fullname_with_scope() << " success.";
    return true;
  }

  // copy tiling to device
  if (!CopyTilingToDevice(stream_ptr)) {
    MS_LOG(EXCEPTION) << "Copy tiling to device failed. op name: " << cnode->fullname_with_scope();
  }

  // pack all addresses into a vector.
  std::vector<void *> runtimeargs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) -> void * { return output->addr; });
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtimeargs),
                         [](const AddressPtr &addr) -> void * { return addr->addr; });
  }

  if (!tiling_data_.empty() && tiling_data_ptr_ != nullptr) {
    runtimeargs.push_back(tiling_data_ptr_);
  }

  rtL2Ctrl_t *l2ctrl = nullptr;
  auto args_size = static_cast<uint32_t>(UlongToUint(sizeof(void *)) * runtimeargs.size());
  auto node_info = cnode->fullname_with_scope();
  const auto dev_func =
    origin_key_.find("kernel0") != origin_key_.npos ? origin_key_ : origin_key_ + "_" + std::to_string(tiling_key_);
  const auto kernel_info = node_info + "/" + std::to_string(tiling_key_);
  // cppcheck-suppress unreadVariable
  auto lock = AscendKernelMod::LockRuntime();
  auto ret = rtKernelLaunchWithHandle(handle_, dev_func.c_str(), block_dim_, runtimeargs.data(), args_size, l2ctrl,
                                      stream_ptr, kernel_info.c_str());
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call runtime rtKernelLaunchWithHandle error. Node info: " << node_info;
    return false;
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
