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
#include "plugin/device/ascend/hal/hardware/ge_kernel_executor.h"
#include <utility>
#include <algorithm>
#include "acl/acl_rt.h"
#include "acl/acl_op_compiler.h"
#include "include/common/profiler.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_metadata.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_build.h"

#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_task.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_build.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_metadata.h"
#include "kernel/kernel_build_info.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_util.h"
#include "transform/acl_ir/ge_adapter_info.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/debug/data_dump/overflow_dumper.h"
#include "include/backend/debug/profiler/profiling.h"
#include "utils/anf_utils.h"
#endif

namespace mindspore::device::ascend {
namespace {
bool GenerateKernelMod(const std::vector<CNodePtr> &kernels) {
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetKernelMod(kernel)) {
      continue;
    }
    if (AnfAlgo::IsKernelSelectBackoffOp(kernel)) {
      continue;
    }
    kernel::KernelModPtr kernel_mod_ptr = nullptr;
    if (AnfAlgo::GetKernelType(kernel) == KernelType::ACL_KERNEL) {
      kernel_mod_ptr = kernel::AclOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::HOST_KERNEL) {
      kernel_mod_ptr = kernel::HostOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::HCCL_KERNEL) {
      kernel_mod_ptr = kernel::HcclOpBuild(kernel);
    } else if (AnfAlgo::GetKernelType(kernel) == KernelType::OPAPI_KERNEL) {
      kernel_mod_ptr = kernel::AclnnOpBuild(kernel);
    } else {
      MS_LOG(EXCEPTION) << "The kernel: " << kernel->fullname_with_scope() << " kernel build failed, kernel type: "
                        << kernel::KernelTypeLabel(AnfAlgo::GetKernelType(kernel));
    }
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, kernel.get());
  }
  return true;
}

bool GraphWithNoRealKernel(const KernelGraphPtr &kernel_graph) {
  const auto &nodes = kernel_graph->execution_order();
  for (auto &node : nodes) {
    if (AnfUtils::IsRealKernel(node)) {
      return false;
    }
  }
  return true;
}

pynative::KernelTaskPtr GetTaskByTaskType(const pynative::KernelTaskType &task_type,
                                          const std::shared_ptr<pynative::KernelTaskContext> &context) {
  switch (task_type) {
    case pynative::KernelTaskType::kCONTIGUOUS_TASK:
      return std::make_shared<AscendContiguousKernelTask>(context);
    case pynative::KernelTaskType::kCOPY_TASK:
      return std::make_shared<AscendCopyWithSliceKernelTask>(context);
    default:
      MS_LOG(EXCEPTION) << "KernelTaskType is invalid, task_type:" << task_type;
  }
}

void SetAclOpPrecisionMode() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto op_precision_mode = ms_context->get_param<std::string>(MS_CTX_OP_PRECISION_MODE);
  if (op_precision_mode.empty()) {
    return;
  }
  MS_LOG(INFO) << "Set ACL_OP_PRECISION_MODE: " << op_precision_mode;
  auto ret = aclSetCompileopt(aclCompileOpt::ACL_OP_PRECISION_MODE, op_precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set op precision mode failed! Error flag is " << ret;
  }
}
}  // namespace

void GeKernelExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  auto ret = aclInit(nullptr);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(WARNING) << "Call aclInit failed. Error flag is " << ret;
  }
  MS_LOG(INFO) << "Call aclInit successfully.";
  MS_EXCEPTION_IF_NULL(device_context_);
  res_manager_ = device_context_->device_res_manager_.get();
  MS_EXCEPTION_IF_NULL(res_manager_);
  graph_executor_ = dynamic_cast<GeGraphExecutor *>(device_context_->graph_executor_.get());
  // not check graph executor, may use in ascend device context
  SetAclOpPrecisionMode();
  initialized_ = true;
}

void GeKernelExecutor::Destroy() {
  if (!initialized_) {
    return;
  }
  res_manager_ = nullptr;
  graph_executor_ = nullptr;
  initialized_ = false;
}

void GeKernelExecutor::UnifyMindIR(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  GEGraphOptimization::GetInstance().UnifyMindIR(graph);
}

void GeKernelExecutor::AddMindIRPass(const KernelGraphPtr &graph) const {
  GEGraphOptimization::GetInstance().GEMindIRPass(graph);
}

void GeKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  // will be cached by OpCompileInfo
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // GE graph run mode do optimize in ProcessBeforeRun
  if (kernel_graph->is_graph_run_mode() && IsEnableRefMode()) {
    return;
  }
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph", 1, 0, 0);
  GEGraphOptimization::GetInstance().OptimizeACLGraph(kernel_graph);
  bool aclnn_can_used = !kernel_graph->is_from_single_op();
  // select kernel
  const auto &kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    auto [select_res, msg, etype] =
      device::ascend::SelectKernelInfoWithMsg(kernel, aclnn_can_used && kernel::IsEnabledAclnn(kernel));
    if (!select_res) {
      MS_LOG(INFO) << "node is " << kernel->fullname_with_scope() << " should backoff";
      std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
      device::ascend::HandleKernelSelectFailure(kernel_graph, kernel, failure_info);
    }
  }
  if (!kernel_graph->is_from_single_op()) {
    kernel_graph->SetKernelObjectTypesForUnrealNodes();
  }

  GEGraphOptimization::GetInstance().OptimizeACLGraphAfterKernelSelect(kernel_graph);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph", 1, 0, 1);
}

void GeKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  if (!nodes.empty() && IsEnableRefMode()) {
    auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(nodes[0]->func_graph());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    // Not create kernel when use GE
    if (!kernel_graph->is_from_single_op() && kernel_graph->is_graph_run_mode()) {
      return;
    }
  }
  // build kernel mod
  MS_LOG(DEBUG) << "Status record: start create kernel.";
  profiler::CollectHostInfo("Ascend", "CreateKernel", "CreateGeKernel", 1, 0, 0);
  PROF_START(create_kernel);
  device::ascend::SetKernelInfoBeforeCreateKernel(nodes);
  auto ret = GenerateKernelMod(nodes);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  PROF_END(create_kernel);
  profiler::CollectHostInfo("Ascend", "CreateKernel", "CreateGeKernel", 1, 0, 1);
  MS_LOG(DEBUG) << "Status record: end create kernel.";
}

void GeKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 0);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  // use GE
  if (kernel_graph->is_graph_run_mode() && IsEnableRefMode()) {
    if (GraphWithNoRealKernel(kernel_graph)) {
      return;
    }
    MS_EXCEPTION_IF_NULL(graph_executor_);
    graph_executor_->PreprocessBeforeRun(kernel_graph);
    profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 1);
    return;
  }

  // nop op -> memcpy
  const auto &nodes = kernel_graph->execution_order();
  for (const auto &node : nodes) {
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    // If the 2nd input of reshape is not a value node, then there are two inputs to select the host reshape operator
    bool is_host_reshape_op = false;
    if (op_name == prim::kPrimReshape->name()) {
      auto kernel_mod = AnfAlgo::GetKernelMod(node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      is_host_reshape_op = kernel_mod->GetKernelModType() == kernel::KernelModType::HostKernelMod;
    }
    bool is_nop_op = transform::AclHelper::IsNopNode(node);
    bool is_transpose_nop = (op_name == prim::kPrimTranspose->name() || op_name == prim::kPrimTransposeD->name()) &&
                            common::AnfAlgo::HasNodeAttr(kAttrNopOp, node);
    if (is_transpose_nop || (is_nop_op && !is_host_reshape_op)) {
      nop_op_to_memcpy_.insert(node);
    }
  }

  profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", 1, 0, 1);
}

bool GeKernelExecutor::PySyncRuning() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(res_manager_);
  if ((ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) &&
      !res_manager_->SyncStream(kDefaultStreamIndex)) {
    return false;
  }
  return true;
}

bool GeKernelExecutor::MemoryCopyAsync(const CNodePtr &node, const vector<KernelTensor *> &inputs,
                                       const vector<KernelTensor *> &outputs) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Launch MemoryCopyAsync instead for kernel " << node->fullname_with_scope();
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(DEBUG) << "Kernel " << node->fullname_with_scope() << " input output size should be 1 but"
                  << " input size is:" << inputs.size() << " output size is:" << outputs.size();
  }

  const auto stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(stream);
  aclError status = aclrtMemcpyAsync(outputs[0]->device_ptr(), outputs[0]->size(), inputs[0]->device_ptr(),
                                     inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status << " destMax:" << outputs[0]->size()
                  << " count:" << inputs[0]->size();
    return false;
  }
  return true;
}

bool GeKernelExecutor::LaunchKernel(const CNodePtr &kernel, const vector<KernelTensor *> &inputs,
                                    const vector<KernelTensor *> &workspace, const vector<KernelTensor *> &outputs,
                                    size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(res_manager_);
  (void)res_manager_->BindDeviceToCurrentThread(false);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  // Stream id may not be assigned in some scenarios, such as PyNative. Use the default stream in those cases.
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_EXCEPTION_IF_NULL(stream);
#ifdef ENABLE_DEBUGGER
  if (DumpJsonParser::GetInstance().async_dump_enabled()) {
    MS_LOG(WARNING) << "Dump is currently not support for pynative mode or kernelbykernel mode, skip dump kernel: "
                    << kernel->fullname_with_scope();
  }
#endif

  // launch kernel
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream);
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    if (!MemoryCopyAsync(kernel, inputs, outputs)) {
      MS_LOG(ERROR) << "Memory copy failed for kernel " << kernel->fullname_with_scope();
      return false;
    }
  } else {
    MS_LOG(DEBUG) << "Begin launch kernel: " << kernel->fullname_with_scope();
    bool ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
    MS_LOG(DEBUG) << "End launch kernel: " << kernel->fullname_with_scope();
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      res_manager_->ResetStreamAndCtx();
      return false;
    }
  }
  // for PyNative Sync Run mode
  auto ret = PySyncRuning();
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
               kernel->fullname_with_scope(), false);

  return ret;
}

bool GeKernelExecutor::ExecuteKernelTask(const pynative::KernelTaskType &task_type,
                                         const device::DeviceAddressPtrList &input_addr_list,
                                         const TensorStorageInfoPtrList &input_storage_list,
                                         const device::DeviceAddressPtrList &output_addr_list) const {
  auto stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);

  auto task_context = std::make_shared<pynative::KernelTaskContext>(device_context_, input_addr_list,
                                                                    input_storage_list, output_addr_list, stream);

  auto task = GetTaskByTaskType(task_type, task_context);
  MS_EXCEPTION_IF_NULL(task);
  auto ret = task->RunWithRet();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Exec task failed, task_type:" << task_type;
  }
  return ret;
}

}  // namespace mindspore::device::ascend
