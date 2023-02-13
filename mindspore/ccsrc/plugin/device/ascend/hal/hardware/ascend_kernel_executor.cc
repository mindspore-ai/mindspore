/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ascend_kernel_executor.h"
#include <algorithm>
#include <utility>
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hardware/ascend_graph_optimization.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/hal/device/kernel_build_ascend.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "include/common/utils/parallel_context.h"
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/hal/device/kernel_adjust.h"

#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "toolchain/adx_datadump_server.h"
#include "toolchain/adx_datadump_callback.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "debug/data_dump/e2e_dump.h"
#include "debug/debugger/debugger_utils.h"
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"
#include "utils/anf_utils.h"
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"

using Adx::AdxRegDumpProcessCallBack;
using mindspore::device::ascend::ProfilingManager;
using mindspore::profiler::ascend::MemoryProfiling;
#endif

namespace mindspore {
namespace device {
namespace ascend {
constexpr size_t kAtomicCleanInputSize = 2;
namespace {
/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: MindRT.
 * Description: Parse config json file and register callback to adx.
 */
#ifndef ENABLE_SECURITY
void DumpInit(uint32_t device_id) {
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  json_parser.CopyDumpJsonToDir(device_id);
  json_parser.CopyHcclJsonToDir(device_id);
  json_parser.CopyMSCfgJsonToDir(device_id);
  if (json_parser.async_dump_enabled()) {
#if !(defined(ENABLE_TEST) || defined(ENABLE_TESTCASES))
    // register callback to adx
    if (json_parser.FileFormatIsNpy()) {
      (void)AdxRegDumpProcessCallBack(mindspore::ascend::DumpDataCallBack);
    }
#endif
    if (AdxDataDumpServerInit() != 0) {
      MS_LOG(EXCEPTION) << "Adx data dump server init failed";
    }
  }
}
#endif
}  // namespace

void AscendKernelExecutor::Initialize() {
  kernel::ascend::TbeKernelCompileManager::GetInstance().TbeInitialize();
  res_manager_ = dynamic_cast<AscendDeviceResManager *>(device_context_->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(res_manager_);
  graph_executor_ = dynamic_cast<AscendGraphExecutor *>(device_context_->graph_executor_.get());
  MS_EXCEPTION_IF_NULL(graph_executor_);
#ifndef ENABLE_SECURITY
  DumpInit(res_manager_->rank_id_);
#endif
}

void AscendKernelExecutor::Destroy() {
  AscendGraphOptimization::GetInstance().Reset();
  res_manager_ = nullptr;
  graph_executor_ = nullptr;
}

void AscendKernelExecutor::UnifyMindIR(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  AscendGraphOptimization::GetInstance().UnifyMindIR(graph);
}

void AscendKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  AscendGraphOptimization::GetInstance().OpAdaptation(kernel_graph);
  if (kernel_graph->is_from_single_op()) {
    AscendGraphOptimization::GetInstance().OptimizeSingleOpGraph(kernel_graph);
  } else {
    AscendGraphOptimization::GetInstance().OptimizeGraph(kernel_graph);
  }
}

// Before creating the kernel, check whether the node has completed the operator selection. If not, the operator
// selection needs to be performed to set kernel info.
void SetKernelInfoBeforeCreateKernel(const std::vector<CNodePtr> &nodes) {
  // Check whether the node has completed kernel selection.
  for (const auto &node : nodes) {
    if (AnfAlgo::GetSelectKernelBuildInfo(node) != nullptr) {
      continue;
    }

    // Kernel selection process.
    auto [status, msg, etype] = SelectKernelInfoWithMsg(node);
    if (status == device::ascend::kNoMatched) {
      auto graph = AnfAlgo::FetchKernelGraph(node.get());
      if (graph == nullptr || graph->is_from_single_op() || graph->is_graph_run_mode()) {
        MS_EXCEPTION(etype) << msg;
      }
      MS_LOG(INFO) << "Try to use backoff CPU kernel, node:" << node->fullname_with_scope();
      std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
      AnfAlgo::SetKernelSelectBackoffInfo(node, failure_info);
      continue;
    }
  }
}

void AscendKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  SetKernelInfoBeforeCreateKernel(nodes);

  MS_LOG(INFO) << "Status record: start create kernel.";
  PROF_START(create_kernel);
  auto ret = device::ascend::KernelBuild(nodes);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }
  PROF_END(create_kernel);
  MS_LOG(INFO) << "Status record: end create kernel.";
}

void AscendKernelExecutor::LaunchDeviceLibrary() const {
  MS_LOG(INFO) << "Status record: start launch device library.";
  auto ret = mindspore::kernel::AicpuOpKernelLoad::GetInstance().LaunchAicpuKernelSo();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Cust aicpu kernel so load failed.";
  }
  MS_LOG(INFO) << "Status record: end launch device library.";
}

void AscendKernelExecutor::SetAtomicCleanToNodes(const KernelGraphPtr &graph,
                                                 const std::map<CNodePtr, std::vector<CNodePtr>> &atomics_node) const {
  // don't clear node_atomics_ in the end, since atomic_clean_nodes_ in kernel.h is weakptr
  MS_EXCEPTION_IF_NULL(graph);
  auto nodes = graph->execution_order();
  for (const auto &node : nodes) {
    auto it = atomics_node.find(node);
    if (it != atomics_node.end()) {
      const auto &atomics = it->second;
      auto kernel_mod = AnfAlgo::GetKernelMod(node);
      auto ascend_kernel_mod = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
      if (ascend_kernel_mod != nullptr) {
        ascend_kernel_mod->SetAtomicCleanNodes(atomics);
      }
    }
  }
}

void AscendKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  device::KernelAdjust::GetInstance().InsertDeviceLoopCtrl(kernel_graph);
  device::KernelAdjust::GetInstance().AssignLoopCtrlMemory(*kernel_graph);
  if (kernel_graph->is_from_single_op()) {
    PreprocessBeforeRunSingleOpGraph(kernel_graph);
  } else {
    PreprocessBeforeRunGraph(kernel_graph);
  }
}

void AscendKernelExecutor::PreprocessBeforeRunGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start preprocess before run graph. graph id: " << graph->graph_id();
  PROF_START(preprocess_before_run_graph);
  try {
    if (graph->is_graph_run_mode()) {
      graph_executor_->PreprocessBeforeRun(graph);
    } else if (graph->is_dynamic_shape() && (IsGraphMode() || graph->has_flag(kFlagPyNativeRunInGraph))) {
      device::ascend::InsertAtomicCleanOps(graph->execution_order(), &node_atomics_);
      SetAtomicCleanToNodes(graph, node_atomics_);  // graph mode may can do it too, instead of update execorder
      AscendStreamAssign::GetInstance().AssignStream(NOT_NULL(graph));
      AssignOutputNopNodeDeviceAddress(graph, device_context_);
      LaunchDeviceLibrary();
    } else {
      PreprocessBeforeRunSingleOpGraph(graph);
      AscendStreamAssign::GetInstance().AssignStream(NOT_NULL(graph));
      const auto &kernels = graph->execution_order();
      CreateKernel(kernels);
      DoSomas(NOT_NULL(graph));
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Preprocess failed before run graph " << graph->graph_id()
                      << ".#dmsg#Framework Error Message:#dmsg#" << e.what();
  }

  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    common::AnfAlgo::SetNodeAttr(kAttrMSFunction, MakeValue(true), kernel);
  }

  PROF_END(preprocess_before_run_graph);
  MS_LOG(INFO) << "Status record: end preprocess before run graph. graph id: " << graph->graph_id();
}

void AscendKernelExecutor::DoSomas(const KernelGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // somas
  if (ms_context->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) != kOptimizeO0) {
    auto somas = std::make_shared<AscendSomas>();
    bool ret = somas->Assign(graph);
    if (ret) {
      MS_LOG(INFO) << "Somas allocate success for graph " << graph->graph_id()
                   << " somas size: " << graph->somas_whole_block_size();
    } else {
      MS_LOG(WARNING) << "Somas allocate failed for graph " << graph->graph_id();
    }
  }
  MS_LOG(INFO) << "Status record: end preprocess before run graph. graph id: " << graph->graph_id();
}

void AscendKernelExecutor::PreprocessBeforeRunSingleOpGraph(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &nodes = graph->execution_order();

  for (const auto &node : nodes) {
    // Remove placeholder
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    static const std::set<std::string> place_holder_nodes = {kDynamicRNNOpName, kDynamicGRUV2OpName};
    auto iter = place_holder_nodes.find(op_name);
    if (iter != place_holder_nodes.end()) {
      auto none_index = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrPlaceHolderIndex);
      // Remove seq_length
      auto input_num = common::AnfAlgo::GetInputTensorNum(node);
      std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(node)};
      for (size_t i = 0; i < input_num; ++i) {
        auto item = std::find(none_index.begin(), none_index.end(), i);
        if (item == none_index.end()) {
          auto input_node = common::AnfAlgo::GetInputNode(node, i);
          new_inputs.emplace_back(input_node);
        }
      }
      (void)node->set_inputs(new_inputs);
    }

    // Save the nop_op that needs to be memcpy
    static mindspore::HashSet<std::string> nop_nodes = {prim::kPrimReshape->name(), prim::kPrimExpandDims->name(),
                                                        prim::kPrimSqueeze->name(), prim::kPrimFlatten->name(),
                                                        prim::kPrimFlattenGrad->name()};
    // If the 2nd input of reshape is not a value node, then there are two inputs to select the host reshape operator
    bool is_host_reshape_op = false;
    if (op_name == prim::kPrimReshape->name()) {
      auto kernel_mod = AnfAlgo::GetKernelMod(node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      is_host_reshape_op = kernel_mod->GetKernelModType() == kernel::KernelModType::HostKernelMod;
    }
    bool nop_op_is_not_dynamic_shape = !graph->is_dynamic_shape() && nop_nodes.find(op_name) != nop_nodes.end();
    bool is_transpose_nop = (op_name == prim::kPrimTranspose->name() || op_name == prim::kPrimTransposeD->name()) &&
                            common::AnfAlgo::HasNodeAttr(kAttrNopOp, node);
    if (is_transpose_nop || (nop_op_is_not_dynamic_shape && !is_host_reshape_op)) {
      nop_op_to_memcpy_.insert(node);
    }
  }

  device::ascend::InsertAtomicCleanOps(nodes, &node_atomics_persistent_cache_);
  std::vector<CNodePtr> atomic_nodes;
  for (const auto &node : nodes) {
    auto iter = node_atomics_persistent_cache_.find(node);
    if (iter != node_atomics_persistent_cache_.end()) {
      const auto &atomics = iter->second;
      (void)std::copy(atomics.begin(), atomics.end(), std::back_inserter(atomic_nodes));
    }
  }

  SetAtomicCleanToNodes(graph, node_atomics_persistent_cache_);
  CreateKernel(atomic_nodes);
  LaunchDeviceLibrary();
}

bool AscendKernelExecutor::PySyncRuning() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if ((ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) &&
      ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) &&
      !res_manager_->SyncStream(kDefaultStreamIndex)) {
    return false;
  }
  return true;
}

bool AscendKernelExecutor::MemoryCopyAsync(const CNodePtr &node, const vector<AddressPtr> &inputs,
                                           const vector<AddressPtr> &outputs) const {
  MS_LOG(DEBUG) << "Launch MemoryCopyAsync instead for kernel " << node->fullname_with_scope();
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(WARNING) << "Kernel " << node->fullname_with_scope() << " input output size should be 1 but"
                    << " input size is:" << inputs.size() << " output size is:" << outputs.size();
  }

  const auto stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(stream);
  aclError status = aclrtMemcpyAsync(outputs[0]->addr, outputs[0]->size, inputs[0]->addr, inputs[0]->size,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status << " destMax:" << outputs[0]->size
                  << " count:" << inputs[0]->size;
    return false;
  }
  return true;
}

bool AscendKernelExecutor::GetKernelRealInputs(const CNodePtr &kernel, const vector<AddressPtr> &inputs,
                                               std::vector<AddressPtr> *real_inputs) const {
  if (AnfAlgo::GetKernelType(kernel) == KernelType::ACL_KERNEL) {
    *real_inputs = inputs;
    return true;
  }

  auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  if (input_num != inputs.size()) {
    MS_LOG(ERROR) << "Input num is " << input_num << " but input address num is " << inputs.size();
    return false;
  }

  for (size_t i = 0; i < input_num; ++i) {
    auto real_index = AnfAlgo::GetInputGraphIdxByKernelIdx(kernel, i);
    if (real_index >= input_num) {
      MS_LOG(ERROR) << "Total input num is " << input_num << " but get real_index " << real_index;
      return false;
    }
    real_inputs->push_back(inputs[real_index]);
  }
  return true;
}

bool AscendKernelExecutor::LaunchKernel(const CNodePtr &kernel, const vector<AddressPtr> &inputs,
                                        const vector<AddressPtr> &workspace, const vector<AddressPtr> &outputs,
                                        size_t stream_id) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto graph_id = AnfAlgo::GetGraphId(kernel.get());
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  KernelType kernel_type = AnfAlgo::GetKernelType(kernel);
  MS_EXCEPTION_IF_NULL(kernel);
  (void)res_manager_->BindDeviceToCurrentThread(false);

  std::vector<AddressPtr> real_inputs;
  bool ret = GetKernelRealInputs(kernel, inputs, &real_inputs);
  if (!ret) {
    MS_LOG(ERROR) << "Get real input fail for kernel " << kernel->fullname_with_scope();
    return false;
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  // Stream id may not be assigned in some scenarios, such as PyNative. Use the default stream in those cases.
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
  }
  MS_EXCEPTION_IF_NULL(stream);

  bool is_dynamic_shape = common::AnfAlgo::IsDynamicShape(kernel);
  if (!is_dynamic_shape || !(common::AnfAlgo::GetBooleanAttr(kernel, kAttrMSFunction))) {
    auto iter = node_atomics_persistent_cache_.find(kernel);
    if (iter != node_atomics_persistent_cache_.end()) {
      std::lock_guard<std::mutex> locker(launch_mutex_);
      // launch atomic clean
      if (!LaunchAtomicClean(kernel, workspace, outputs, stream)) {
        MS_LOG(ERROR) << "Launch AtomicClean failed, pre kernel full name: " << kernel->fullname_with_scope();
        return false;
      }
    }
  }

  // launch kernel
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    if (!MemoryCopyAsync(kernel, real_inputs, outputs)) {
      MS_LOG(ERROR) << "Memory copy failed for kernel " << kernel->fullname_with_scope();
      return false;
    }
  } else {
    MS_LOG(DEBUG) << "Begin launch kernel: " << kernel->fullname_with_scope();
    ret = kernel_mod->Launch(real_inputs, workspace, outputs, stream);
    MS_LOG(DEBUG) << "End launch kernel: " << kernel->fullname_with_scope();
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      return false;
    }
  }
#ifndef ENABLE_SECURITY
  auto ascend_instance = profiler::ascend::AscendProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(ascend_instance);
  if (ProfilingManager::GetInstance().IsProfilingInitialized()) {
    ascend_instance->GetNodeTaskIdStreamId(kernel, graph_id, UintToInt(device_id), kernel_type);
  }
#endif
  return PySyncRuning();
}

bool AscendKernelExecutor::LaunchAtomicClean(const CNodePtr &node, const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs, void *stream) const {
  auto iter = node_atomics_persistent_cache_.find(node);
  if (iter == node_atomics_persistent_cache_.end()) {
    return true;
  }
  if (AnfAlgo::IsDynamicShapeSkipExecute(node)) {
    return true;
  }
  MS_LOG(DEBUG) << "Launch atomic clean for kernel " << node->fullname_with_scope();
  auto atomic_node = iter->second.at(0);
  vector<AddressPtr> atomic_inputs;
  // The output addr need to clean
  MS_EXCEPTION_IF_NULL(atomic_node);
  if (atomic_node->inputs().size() != kAtomicCleanInputSize) {
    MS_LOG(EXCEPTION) << "Atomic Addr clean Node Input nodes not equal 2.";
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, node)) {
    auto clean_output_indexes = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(node, kAttrAtomicOutputIndexs);
    for (auto output_index : clean_output_indexes) {
      if (output_index >= outputs.size()) {
        MS_LOG(EXCEPTION) << "Invalid output_index:" << output_index << " except less than " << outputs.size();
      }
      atomic_inputs.push_back(outputs[output_index]);
    }
  }

  // The workspace addr need to clean
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, node)) {
    auto clean_workspace_indexes = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(node, kAttrAtomicWorkspaceIndexs);
    for (auto workspace_index : clean_workspace_indexes) {
      if (workspace_index >= workspace.size()) {
        MS_LOG(EXCEPTION) << "Invalid workspace_index:" << workspace_index << " except less than " << workspace.size();
      }
      atomic_inputs.push_back(workspace[workspace_index]);
    }
  }
  // Launch Atomic Node
  auto kernel_mod = AnfAlgo::GetKernelMod(atomic_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->Launch(atomic_inputs, {}, {}, stream);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
