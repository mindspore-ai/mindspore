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

#include "plugin/device/ascend/kernel/tbe/dynamic_tbe_kernel_mod.h"

#include <algorithm>
#include <stack>
#include <utility>
#include "acl/acl_rt.h"
#include "utils/ms_context.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_runtime.h"
#include "backend/common/optimizer/helper.h"
#include "framework/common/debug/log.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "plugin/device/ascend/kernel/tbe/tiling/op_tiling_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/utils.h"
#include "register/op_tiling.h"
#include "nlohmann/json.hpp"
#include "runtime/device/memory_manager.h"

namespace mindspore {
namespace kernel {
using TbeTaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::TbeTaskInfo>;
using tbe::KernelManager;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;
using tbe::TbeUtils;

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
    auto mem_manager = std::make_shared<device::ascend::AscendMemoryManager>();
    mem_manager->FreeMemFromMemPool(tiling_data_ptr_);
  }
}

void DynamicTbeKernelMod::SyncData() {
  if (need_skip_execute_) {
    AscendKernelMod::SyncData();
  }
}

void DynamicTbeKernelMod::GenFuncStub() {
  if (func_stub_ == nullptr && handle_ == nullptr) {
    MS_EXCEPTION_IF_NULL(kernel_pack_);
    auto func_stub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim_, &handle_, &origin_key_);
    if (kernel_pack_->kernel_json_info().has_kernel_list) {
      if (func_stub != 1) {
        MS_LOG(EXCEPTION) << "GenFuncStub failed.";
      }
    } else {
      if (func_stub == 0) {
        MS_LOG(EXCEPTION) << "GenFuncStub failed.";
      }
      func_stub_ = reinterpret_cast<void *>(func_stub);
    }
  }
}

int DynamicTbeKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The node is not dynamic shape: " << cnode->fullname_with_scope();
  }

  // update output size after InferShape.
  // avoid atomic_clean memory violation, we need dynamic atomic_clean op.
  AscendKernelMod::UpdateOutputSizeList();

  need_skip_execute_ = AnfAlgo::IsDynamicShapeSkipExecute(cnode);
  if (need_skip_execute_) {
    return 0;
  }

  GenFuncStub();
  // start compute tiling
  device::tiling::OpTilingCalculateAdapter converter;
  const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map = inputsOnHost;
  ::ge::ComputeGraphPtr ge_graph = std::make_shared<::ge::ComputeGraph>("default");
  optiling::utils::OpRunInfo op_run_info_v2(-1, true, 0);
  MS_LOG(INFO) << "Start compute tiling of: " << cnode->fullname_with_scope();
  if (!atomic_clean_nodes_.empty()) {
    atomic_compile_info_ = ParseCompileJson(atomic_clean_nodes_[0].lock());
  }
  auto ge_node = converter.AnfNodeToGeNodeAdapter(cnode, &ge_graph, depend_tensor_map, op_compile_info_);
  MS_EXCEPTION_IF_NULL(ge_node);
  auto ge_op = converter.GeNodeToGeOperatorAdapter(ge_node);
  auto ret = optiling::OpParaCalculateV2(ge_op, op_run_info_v2);
  if (ret != ::ge::GRAPH_SUCCESS) {
    MS_LOG(EXCEPTION) << "The node: " << cnode->fullname_with_scope() << " compute tiling failed!";
  }

  block_dim_ = op_run_info_v2.GetBlockDim();
  std::vector<int64_t> workspace_size_list;
  op_run_info_v2.GetAllWorkspaces(workspace_size_list);
  tiling_data_ = op_run_info_v2.GetAllTilingData().str();
  tiling_key_ = op_run_info_v2.GetTilingKey();

  workspace_size_list_.clear();
  workspace_size_list_.resize(workspace_size_list.size());
  std::transform(workspace_size_list.begin(), workspace_size_list.end(), workspace_size_list_.begin(),
                 [](int64_t size) { return static_cast<size_t>(size); });

  // compute tiling of atomic_clean op.
  if (!atomic_clean_nodes_.empty()) {
    // Update workspace size
    converter.UpdateWorkspace(ge_node, workspace_size_list);

    optiling::utils::OpRunInfo atomic_op_info(-1, true, 0);
    ret = optiling::OpAtomicCalculateV2(*ge_node, atomic_op_info);
    if (ret != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "The node: " << cnode->fullname_with_scope() << " compute atomic tiling failed!";
    }
    for (const auto &atomic_clean_node : atomic_clean_nodes_) {
      auto dynamic_kernel_mod = dynamic_cast<DynamicTbeKernelMod *>(AnfAlgo::GetKernelMod(atomic_clean_node.lock()));
      MS_EXCEPTION_IF_NULL(dynamic_kernel_mod);
      dynamic_kernel_mod->InitAtomicOps(atomic_op_info);
    }
  }
  return 0;
}

std::string DynamicTbeKernelMod::ParseCompileJson(const CNodePtr &cnode) const {
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
    auto mem_manager = std::make_shared<device::ascend::AscendMemoryManager>();
    tiling_data_ptr_ = mem_manager->MallocMemFromMemPool(op_para_size, false);
    if (tiling_data_ptr_ == nullptr) {
      MS_LOG(EXCEPTION) << "RtMalloc tiling data failed.";
    }
  }
}

void DynamicTbeKernelMod::CopyTilingToDevice(void *stream_ptr) {
  InitTilingDataPtr();
  MS_EXCEPTION_IF_NULL(kernel_pack_);
  auto kernel_json_info = kernel_pack_->kernel_json_info();

  auto op_para_size = kernel_json_info.op_para_size;
  if (tiling_data_.size() > op_para_size) {
    MS_LOG(EXCEPTION) << "Compute tiling size:" << tiling_data_.size()
                      << " larger than tbe build op_para_size:" << op_para_size;
  }

  if (tiling_data_.empty() || tiling_data_ptr_ == nullptr) {
    MS_LOG(INFO) << "Tiling size is 0, skip aclrtMemcpyAsync.";
    return;
  }
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  auto ret = rtMemcpyAsync(tiling_data_ptr_, op_para_size, tiling_data_.c_str(), tiling_data_.size(),
                           RT_MEMCPY_HOST_TO_DEVICE_EX, stream_ptr);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Tiling aclrtMemcpyAsync failed, ret:" << ret;
  }
}

bool DynamicTbeKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "The stream_ptr should not be nullptr.";
    return false;
  }

  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "The kernel_pack should not be nullptr.";
    return false;
  }

  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  // is dynamic shape
  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The cnode is not dynamic shape:" << cnode->fullname_with_scope();
  }

  // need skip, for reducesum empty input axis
  if (need_skip_execute_) {
    // Skip reduce if axis is a empty Tensor (shape = 0)
    MS_LOG(INFO) << "The node " << cnode->fullname_with_scope() << "Need Skip.";
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
    rtError_t status = aclrtMemcpyAsync(outputs[0]->addr, inputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
    if (status != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "AclrtMemcpyAsync failed for " << cnode->fullname_with_scope();
    }

    MS_LOG(INFO) << "Execute node:" << cnode->fullname_with_scope() << " success.";
    return true;
  }

  if (!atomic_clean_nodes_.empty()) {
    for (const auto &atomic_clean_node : atomic_clean_nodes_) {
      KernelLaunchInfo kernel_launch_info;
      auto kernel_mod = AnfAlgo::GetKernelMod(atomic_clean_node.lock());
      MS_EXCEPTION_IF_NULL(kernel_mod);
      device::KernelRuntime::GenLaunchArgs(*kernel_mod, atomic_clean_node.lock(), &kernel_launch_info);
      auto atomic_inputs = kernel_launch_info.inputs_;
      std::vector<AddressPtr> atomic_outputs;
      std::vector<AddressPtr> atomic_workspace;
      kernel_mod->Launch(atomic_inputs, atomic_workspace, atomic_outputs, stream_ptr);
    }
  }

  // copy tiling to device
  CopyTilingToDevice(stream_ptr);

  // pack all addresses into a vector.
  std::vector<void *> runtimeargs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &input) { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtimeargs),
                       [](const AddressPtr &output) { return output->addr; });
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtimeargs),
                         [](const AddressPtr &addr) { return addr->addr; });
  }

  if (!tiling_data_.empty() && tiling_data_ptr_ != nullptr) {
    runtimeargs.push_back(tiling_data_ptr_);
  }

  AddressPtr overflow_address_ptr = GetOverflowAddress();
  if (overflow_address_ptr != nullptr) {
    runtimeargs.emplace_back(overflow_address_ptr->addr);
    MS_LOG(DEBUG) << "Assign overflow memory for node " << node->fullname_with_scope() << ", addr is "
                  << overflow_address_ptr->addr;
  }

  rtL2Ctrl_t *l2ctrl = nullptr;
  auto args_size = static_cast<uint32_t>(UlongToUint(sizeof(void *)) * runtimeargs.size());
  auto node_info = cnode->fullname_with_scope();
  if (kernel_pack_->kernel_json_info().has_kernel_list) {
    const auto kernel_info = node_info + "/" + std::to_string(tiling_key_);
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
    rtArgsEx_t args_info = {};
    args_info.args = runtimeargs.data();
    args_info.argsSize = args_size;
    auto ret =
      rtKernelLaunchWithHandle(handle_, tiling_key_, block_dim_, &args_info, l2ctrl, stream_ptr, kernel_info.c_str());
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call runtime rtKernelLaunchWithHandle error. Node info: " << node_info;
      return false;
    }
  } else {
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
    auto ret = rtKernelLaunch(func_stub_, block_dim_, runtimeargs.data(), args_size, l2ctrl, stream_ptr);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call runtime rtKernelLaunch error. Node info: " << node_info;
      return false;
    }
  }

  return true;
}

void DynamicTbeKernelMod::InitAtomicOps(const optiling::utils::OpRunInfo &op_info) {
  GenFuncStub();
  AscendKernelMod::UpdateOutputSizeList();
  block_dim_ = op_info.GetBlockDim();
  std::vector<int64_t> workspace_size_list;
  op_info.GetAllWorkspaces(workspace_size_list);
  tiling_data_ = op_info.GetAllTilingData().str();
  tiling_key_ = op_info.GetTilingKey();

  workspace_size_list_.clear();
  workspace_size_list_.resize(workspace_size_list.size());
  std::transform(workspace_size_list.begin(), workspace_size_list.end(), workspace_size_list_.begin(), LongToSize);
}
}  // namespace kernel
}  // namespace mindspore
