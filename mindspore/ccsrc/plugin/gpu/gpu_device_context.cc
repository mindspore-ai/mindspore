/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License"){}
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

#include "plugin/gpu/gpu_device_context.h"
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include <unordered_set>
#include "kernel/gpu/gpu_common.h"
#include "include/utils/callback.h"
#include "plugin/gpu/kernel_executor/gpu_kernel_build.h"
#include "plugin/gpu/kernel_executor/gpu_kernel_task.h"
#include "plugin/gpu/res_manager/gpu_device_manager.h"
#include "plugin/gpu/res_manager/event_manager/gpu_event.h"
#include "plugin/gpu/res_manager/device_context_conf/op_tuning_conf.h"
#include "plugin/gpu/res_manager/device_context_conf/op_precision_conf.h"
#include "plugin/gpu/res_manager/mem_manager/gpu_pin_mem_pool.h"
#include "plugin/gpu/res_manager/mem_manager/gpu_memory_allocator.h"
#include "plugin/gpu/graph_optimizer/pass/base/optimizer.h"
#include "plugin/gpu/graph_optimizer/pass/base/kernel_info_setter.h"
#include "plugin/gpu/graph_optimizer/pass/base/reg_gpu_const_input_to_attr.h"
#include "plugin/gpu/graph_optimizer/somas/gpu_somas.h"
#include "plugin/gpu/graph_optimizer/stream_assign/gpu_stream_assign.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/runtime/hardware_abstract/data_queue/data_queue_mgr.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/cuda_common.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "plugin/gpu/profiler/gpu_profiling.h"
#include "plugin/gpu/profiler/gpu_profiling_utils.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "include/backend/common/pass_manager/common_backend_optimization.h"
#include "include/backend/common/pass_manager/dynamic_shape_helper.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/comm_manager.h"
#include "backend/common/pass/optimize_updatestate.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/backend/common/expander/fallback/expander_fallback.h"
#include "backend/common/pass/value_graph_binder.h"
#include "kernel/cpu/cpu_kernel.h"
#include "tools/profiler/profiler.h"
#include "primitive/ascend_op_name.h"
#include "include/backend/common/device_address_utils.h"
#include "runtime/pipeline/task/kernel_task.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/utils/parallel_context.h"
#include "tools/profiler/profiling.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "mindspore/ops/kernel/gpu/cuda/arrays/contiguous_gpu_kernel.h"
#include "mindspore/core/include/ir/tensor_new.h"

namespace mindspore {
namespace device {
namespace gpu {
namespace {
const char kModelNameGPU[] = "GPU";
const char kEventOptimizeGraph[] = "OptimizeGraph";
const char kStageOptimizeWithoutDeviceInfo[] = "OptimizeWithoutDeviceInfo";
const char kStageSetKernelInfo[] = "SetKernelInfo";
const char kStageOptimizeWithDeviceInfo[] = "OptimizeWithDeviceInfo";
constexpr char kKernelObjectTypeNotSupportedStr[] = "KernelObjectTypeNotSupported";
std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string cur_so_path = dl_info.dli_fname;
  return dirname(cur_so_path.data());
#else
  return "";
#endif
}

runtime::KernelTaskPtr GetTaskByTaskType(const runtime::KernelTaskType &task_type,
                                         const std::shared_ptr<runtime::KernelTaskContext> &task_context) {
  switch (task_type) {
    case runtime::KernelTaskType::kCONTIGUOUS_TASK:
      return std::make_shared<GpuContiguousKernelTask>(task_context);
      break;
    case runtime::KernelTaskType::kCOPY_TASK:
      return std::make_shared<GpuCopyWithSliceKernelTask>(task_context);
      break;
    default:
      MS_LOG(EXCEPTION) << "KernelTaskType is invalid, task_type:" << task_type;
  }
}
}  // namespace
using KernelGraph = mindspore::session::KernelGraph;

void GPUDeviceContext::Initialize() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) {
    if (!device_res_manager_->BindDeviceToCurrentThread(false)) {
      MS_LOG(EXCEPTION) << "BindDeviceToCurrentThread failed.";
    }
    GPUMemoryAllocator::GetInstance().CheckMaxDeviceMemory();
    return;
  }

  device_res_manager_->Initialize();
  MS_EXCEPTION_IF_NULL(GetKernelExecutor());
  GetKernelExecutor()->Initialize();
  // Dump json config file if dump is enabled.
  uint32_t rank_id = 0;
  if (distributed::collective::CollectiveManager::instance()->need_init()) {
    rank_id = device_context_key().device_id_;
  }

  MS_LOG(INFO) << "Set rank id " << rank_id << " for dumping.";
  constexpr char kParse[] = "DumpJsonParserParse";
  static auto parse_callback = callback::CommonCallback::GetInstance().GetCallback<void>(kParse);
  if (parse_callback) {
    parse_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get DumpJsonParserParse, data dump function may not work.";
  }
  constexpr char kCopyDumpJsonToDir[] = "CopyDumpJsonToDir";
  static auto copy_dump_json_to_dir_callback =
    callback::CommonCallback::GetInstance().GetCallback<void, uint32_t>(kCopyDumpJsonToDir);
  if (copy_dump_json_to_dir_callback) {
    copy_dump_json_to_dir_callback(rank_id);
  } else {
    MS_LOG(WARNING) << "Failed to get CopyDumpJsonToDir, data dump function may not work.";
  }
  constexpr char kCopyMSCfgJsonToDir[] = "CopyMSCfgJsonToDir";
  static auto copy_mscfg_json_to_dir_callback =
    callback::CommonCallback::GetInstance().GetCallback<void, uint32_t>(kCopyMSCfgJsonToDir);
  if (copy_mscfg_json_to_dir_callback) {
    copy_mscfg_json_to_dir_callback(rank_id);
  } else {
    MS_LOG(WARNING) << "Failed to get CopyMSCfgJsonToDir, data dump function may not work.";
  }
  initialized_ = true;
}

void GPUDeviceContext::Destroy() {
  MS_EXCEPTION_IF_NULL(GetKernelExecutor());
  GetKernelExecutor()->Destroy();
  device_res_manager_->Destroy();
  initialized_ = false;
}

void GPUKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // somas
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    auto somas = std::make_shared<GPUSomas>();
    bool ret = somas->Assign(kernel_graph);
    if (ret) {
      MS_LOG(INFO) << "Somas allocate success for graph " << kernel_graph->graph_id()
                   << " somas size: " << kernel_graph->somas_whole_block_size();
    } else if (somas->IsSupportSomas(*kernel_graph)) {
      MS_LOG(WARNING) << "Somas allocate failed for graph " << kernel_graph->graph_id();
    }
  }
  MS_LOG(INFO) << "Status record: end preprocess before run graph. graph id: " << kernel_graph->graph_id();
}

void GPUKernelExecutor::OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  // Operator fusion optimization.
  FuseOperators(graph);
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageOptimizeWithoutDeviceInfo, start_time,
                                  profiler::GetClockSyscnt(), 1);
}

void GPUKernelExecutor::OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  uint64_t start_time = profiler::GetClockSyscnt();
  // Graph optimization relevant to device data format
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>("insert_type_transform_op"));
  // ReplaceAddNFusion depends on the input expansion of AddN, so must be after the operator select.
  pm->AddPass(std::make_shared<opt::ReplaceAddNFusion>());
  // PrintReduceFusion depends on the input expansion of Print, so must be after the operator select.
  pm->AddPass(std::make_shared<opt::PrintReduceFusion>("print_reduce"));

  // The fusion operator generates a new primitive and can't be supported in dynamic shape scene.
  if (!graph->is_dynamic_shape()) {
    pm->AddPass(std::make_shared<opt::BatchNormReluFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormSiluFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormReluGradFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormSiluGradFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormAddReluFusion>());
    pm->AddPass(std::make_shared<opt::PostBatchNormAddReluFusion>());
    pm->AddPass(std::make_shared<opt::BatchNormAddReluGradFusion>());
    pm->AddPass(std::make_shared<opt::InsertFormatTransformOp>());
    pm->AddPass(std::make_shared<opt::RemoveFormatTransformPair>());
    pm->AddPass(std::make_shared<opt::RemoveRedundantFormatTransform>());
    // Remove node only used by UpdateState, in order to ensure the correct execution sequence in
    // CudnnInplaceAggregate.
    pm->AddPass(std::make_shared<opt::OptimizeUpdateState>());
    pm->AddPass(std::make_shared<opt::CudnnInplaceAggregate>());
  }

  pm->AddPass(std::make_shared<opt::AdjustDependForParallelOptimizerRecomputeAllGather>());
  pm->AddPass(std::make_shared<opt::GetitemTuple>());
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  pm->AddPass(std::make_shared<opt::InsertTensorMoveForCommunication>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageOptimizeWithDeviceInfo, start_time,
                                  profiler::GetClockSyscnt(), 1);
}

void GPUKernelExecutor::FuseOperators(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  // In the dynamic shape scene, the infershape stage needs to call the primitive infer function.
  // When the fusion operator generates a new primitive, but there
  // is no corresponding primitive infer function, an error will occur.
  // Therefore, this kind of scene does not support dynamic shape.
  if (graph->is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic shape skip some fusion pass";
    pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
  } else {
    pm->AddPass(std::make_shared<opt::ClipByNormFission>());
    pm->AddPass(std::make_shared<opt::MatMulBiasAddFusion>());
    pm->AddPass(std::make_shared<opt::AdamWeightDecayFusion>());
    pm->AddPass(std::make_shared<opt::AdamFusion>());
    pm->AddPass(std::make_shared<opt::AllToAllFusion>());
    pm->AddPass(std::make_shared<opt::ApplyMomentumWeightDecayScaleFusion>());
    pm->AddPass(std::make_shared<opt::ApplyMomentumScaleFusion>());
    pm->AddPass(std::make_shared<opt::ApplyMomentumWeightDecayFusion>());
    if (!graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
      pm->AddPass(std::make_shared<opt::CastAllFusion>("cast_all"));
    }
    pm->AddPass(std::make_shared<opt::CombineOptimizerFusion>(kCombineOptimizerOpName));
    pm->AddPass(std::make_shared<opt::ReplaceMomentumCastFusion>());
    pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
    pm->AddPass(std::make_shared<opt::NeighborExchangeV2Fusion>());
    pm->AddPass(std::make_shared<opt::NeighborExchangeV2GradFusion>());
    pm->AddPass(std::make_shared<opt::BiasDropoutAddFusion>());

    // Do communication op fusion before InsertTensorMoveForCommunication pass.
    // So these passes are before kernel select process, no need to generate kernel build info in them.
    if (parallel::ParallelContext::GetInstance()->enable_all_reduce_fusion()) {
      MS_LOG(INFO) << "Parallel comm_fusion of AllReduce is enabled.";
      pm->AddPass(std::make_shared<opt::AllReduceFusion>());
    }
    if (parallel::ParallelContext::GetInstance()->enable_all_gather_fusion()) {
      MS_LOG(INFO) << "Parallel comm_fusion of AllGather is enabled.";
      pm->AddPass(std::make_shared<opt::AllGatherFusion>());
      pm->AddPass(std::make_shared<opt::ConcatOutputsForAllGather>());
    }
  }
  pm->AddPass(std::make_shared<opt::DynamicSequenceOpsAdaptation>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}

namespace {
bool IsAllNopNode(const session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto execution_order = graph->execution_order();
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (!common::AnfAlgo::IsNopNode(cnode)) {
      return false;
    }
  }
  return true;
}

bool NeedHideNode(const std::vector<AnfNodePtr> &outputs, const AnfNodePtr &node, bool need_keep_output_nop_node) {
  MS_EXCEPTION_IF_NULL(node);
  // if node is not a nop node, keep it in execution order
  if (!common::AnfAlgo::IsNopNode(node)) {
    return false;
  }
  // if node is nop node and the graph is dynamic graph, check if the nop node is graph's output.
  if (need_keep_output_nop_node) {
    auto iter = find(outputs.begin(), outputs.end(), node);
    if (iter != outputs.end()) {
      return false;
    }
  }
  return true;
}

void HideNopNode(session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (IsAllNopNode(graph) == true) {
    return;
  }
  auto execution_order = graph->execution_order();
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output());
  // If the graph has flag kFlagEnableZeroCopyInGraph, it means in subgraph sink mode, the inputs and outputs memory of
  // graph should not be allocated, and the node should not be skipped.
  bool need_keep_output_nop_node = (graph->is_dynamic_shape() || graph->has_flag(kFlagEnableZeroCopyInGraph));
  MS_LOG(INFO) << "nop node info (Before Remove) size: " << execution_order.size();
  std::vector<CNodePtr> new_nodes;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (NeedHideNode(outputs, cnode, need_keep_output_nop_node)) {
      common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(true), cnode);
      common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpExecution, MakeValue(true), cnode);
    } else {
      new_nodes.push_back(cnode);
    }
  }
  graph->set_execution_order(new_nodes);
  MS_LOG(INFO) << "nop node info (After Remove) size: " << graph->execution_order().size();
}

void RemoveNopNode(session::KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (IsAllNopNode(graph) == true) {
    return;
  }
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<CNodePtr> new_nodes;
    auto outputs = graph->outputs();
    bool is_dynamic_graph = graph->is_dynamic_shape();
    for (auto &cnode : graph->execution_order()) {
      MS_EXCEPTION_IF_NULL(cnode);
      // ignore nop node itself
      if (NeedHideNode(outputs, cnode, is_dynamic_graph)) {
        common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpAddr, MakeValue(true), cnode);
        common::AnfAlgo::SetNodeAttr(kAttrSkipNopOpExecution, MakeValue(true), cnode);
        continue;
      }
      // Replace the input which is nop node
      std::vector<AnfNodePtr> new_inputs;
      new_inputs.push_back(cnode->input(0));
      bool need_update = false;
      for (size_t i = 1; i < cnode->size(); ++i) {
        auto input = cnode->input(i);
        MS_EXCEPTION_IF_NULL(input);
        auto cinput = input->cast<CNodePtr>();
        if (cinput == nullptr || !common::AnfAlgo::IsNopNode(cinput)) {
          new_inputs.push_back(input);
          continue;
        }
        constexpr auto kInputSize = 2;
        if (cinput->size() == kInputSize) {
          new_inputs.push_back(cinput->input(1));
          need_update = true;
          changed = true;
        } else {
          new_inputs.push_back(input);
        }
      }
      if (need_update) {
        cnode->set_inputs(new_inputs);
      }
      // push into new execution list
      new_nodes.push_back(cnode);
    }
    graph->set_execution_order(new_nodes);
  }
}

void RunOpOptimize(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void RunOpHardwareOptimize(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void RunOpHideNopNode(const KernelGraphPtr &kernel_graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    HideNopNode(kernel_graph.get());
  }
}

void RunOpRemoveNopNode(const KernelGraphPtr &kernel_graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    RemoveNopNode(kernel_graph.get());
  }
}

bool IsKernelObjectTypeNotSupportedError(const std::string &error_str) {
  return error_str.find(kKernelObjectTypeNotSupportedStr) != std::string::npos;
}

bool CheckSupportBackoff(const KernelGraphPtr &graph, const CNodePtr &node,
                         const std::pair<std::string, ExceptionType> &failure_info) {
  MS_EXCEPTION_IF_NULL(node);
  // The single op does not support the backoff ability.
  if (!AnfAlgo::IsNodeSupportKernelSelectBackoff(node, graph)) {
    return false;
  }
  const auto &kernel_name = common::AnfAlgo::GetCNodeName(node);
  const auto &kernel_attrs = kernel::NativeCpuKernelMod::GetCpuSupportedList(kernel_name);
  // CPU also doesn't support the kernel.
  if (kernel_attrs.empty() || IsKernelObjectTypeNotSupportedError(failure_info.first)) {
    return false;
  }
  return true;
}

void SetBackoffInfo(const CNodePtr &node, const std::pair<std::string, ExceptionType> &failure_info) {
  MS_LOG(INFO) << "GPU doesn't support the kernel " << common::AnfAlgo::GetCNodeName(node)
               << " and will try to backoff on CPU.";
  // Mark kernel selection backoff info.
  AnfAlgo::SetKernelSelectBackoffInfo(node, failure_info);
}

// Mark the kernel backoff with failure info when setting operator info fails.
void HandleKernelSelectFailure(const KernelGraphPtr &graph, const CNodePtr &node,
                               const std::pair<std::string, ExceptionType> &failure_info) {
  if (!CheckSupportBackoff(graph, node, failure_info)) {
    MS_EXCEPTION(failure_info.second) << "#umsg#Kernel select failed:#umsg#" << failure_info.first;
  }
  SetBackoffInfo(node, failure_info);
}

bool TryExpandFallback(const KernelGraphPtr &graph, const CNodePtr &node,
                       const std::pair<std::string, ExceptionType> &failure_info) {
  auto f = [ori_node = node, &failure_info, &graph](const CNodePtr &basic_op) mutable {
    MS_EXCEPTION_IF_NULL(basic_op);
    auto res = SetKernelInfoWithMsg(basic_op);
    if (res.first.empty()) {
      // select gpu kernel success.
      return true;
    }
    // select gpu kernel failed, first try to use CPU kernel for original op.
    if (ori_node != nullptr) {
      MS_LOG(DEBUG) << "The basic op " << basic_op->fullname_with_scope()
                    << " select kernel failed. Try to backoff on CPU for original op "
                    << ori_node->fullname_with_scope();
      if (CheckSupportBackoff(graph, ori_node, failure_info)) {
        // original node use cpu kernel, stop expanding.
        MS_LOG(DEBUG) << "Original op " << ori_node->fullname_with_scope() << " use CPU kernel.";
        return false;
      } else {
        MS_LOG(DEBUG) << "Failed to backoff on CPU for original op " << ori_node->fullname_with_scope()
                      << ", try to backoff on CPU for basic op " << basic_op->fullname_with_scope();
      }
      // only try once for original node.
      ori_node = nullptr;
    } else {
      MS_LOG(DEBUG) << "The basic op " << basic_op->fullname_with_scope()
                    << " select kernel failed, try to backoff on CPU";
    }
    // Original op cannot backoff on CPU, try to use CPU kernel for current op.
    if (CheckSupportBackoff(graph, basic_op, res)) {
      AnfAlgo::SetKernelSelectBackoffInfo(basic_op, res);
      MS_LOG(DEBUG) << "The basic op " << basic_op->fullname_with_scope() << " use CPU kernel.";
      return true;
    }
    return false;
  };
  return expander::TryExpandCNode(node, f);
}

// Before creating the kernel, check whether the node has completed the operator selection. If not, the operator
// selection needs to be performed to set kernel info.
void SetKernelInfoBeforeCreateKernel(const std::vector<CNodePtr> &nodes) {
  for (const auto &node : nodes) {
    auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
    // Kernel selection process.
    if (build_info == nullptr) {
      const auto &failure_info = SetKernelInfoWithMsg(node);
      if (!failure_info.first.empty()) {
        const auto &kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
        HandleKernelSelectFailure(kernel_graph, node, failure_info);
      }
    } else if (!build_info->valid()) {
      // Judge whether match strictly between kernel build info and supported kernel attrs.
      const auto &kernel_attr = kernel::GetKernelAttrFromBuildInfo(build_info);
      const auto &supported_kernel_attrs =
        kernel::NativeGpuKernelModFactory::GetInstance().GetGpuSupportedList(common::AnfAlgo::GetCNodeName(node));
      const auto &match_result = kernel::MatchKernelAttrStrict(kernel_attr, supported_kernel_attrs);
      if (!match_result.first) {
        auto attr_info = kernel::FetchPrintInfoByKernelAttr(kernel_attr);
        std::string error_info =
          "Unsupported op [" + common::AnfAlgo::GetCNodeName(node) + "] on GPU, node attr: " + attr_info;
        const auto &kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
        HandleKernelSelectFailure(kernel_graph, node, {error_info, NotSupportError});
      }
      build_info->set_valid(true);
    }
  }
}

// Check whether mutex exists for a stream.
std::pair<bool, std::mutex *> CheckStreamMutexExist(
  const void *stream, const mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> &mtxs_for_streams,
  std::shared_mutex *shd_mtx) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(shd_mtx);
  std::shared_lock<std::shared_mutex> shd_lock(*shd_mtx);
  auto iter = mtxs_for_streams.find(stream);
  if (iter != mtxs_for_streams.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return std::make_pair(true, iter->second.get());
  }
  return std::make_pair(false, nullptr);
}

// Create a mutex for stream.
std::mutex *CreateStreamMutex(const void *stream, std::shared_mutex *shd_mtx,
                              mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> *mtxs_for_streams) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(shd_mtx);
  MS_EXCEPTION_IF_NULL(mtxs_for_streams);

  std::unique_lock<std::shared_mutex> unq_lock(*shd_mtx);
  auto ret_pair = mtxs_for_streams->emplace(stream, std::make_shared<std::mutex>());

  MS_EXCEPTION_IF_NULL(ret_pair.first->second);
  return ret_pair.first->second.get();
}

// The launch kernel is thread-unsafe, and the behavior of delivering the kernel launch to the same stream requires
// lock protection, need to create a separate lock for each stream.
// for GPU, The cublas handle is not thread safety specifically, it is not recommended that multiple threads access the
// same cublas handle at the same time, so need the launch mutex when multiple threads launch the cublas kernels.
std::lock_guard<std::mutex> LockLaunchKernel(const void *stream) {
  MS_EXCEPTION_IF_NULL(stream);
  // Read-write lock for accessing mtxs_for_streams map.
  // When the lock of each stream is created, mtxs_for_streams can be accessed concurrently to improve performance.
  static std::shared_mutex shd_mtx;
  static mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> mtxs_for_streams;

  std::mutex *stream_mtx;
  // Check whether mutex exists for a stream.
  std::pair<bool, std::mutex *> ret_pair = CheckStreamMutexExist(stream, mtxs_for_streams, &shd_mtx);
  if (ret_pair.first) {
    stream_mtx = ret_pair.second;
  } else {
    // Create a mutex for stream.
    stream_mtx = CreateStreamMutex(stream, &shd_mtx, &mtxs_for_streams);
  }

  MS_EXCEPTION_IF_NULL(stream_mtx);
  // Lock kernel launch for the stream.
  return std::lock_guard<std::mutex>(*stream_mtx);
}
}  // namespace

void GPUKernelExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  res_manager_ = dynamic_cast<GPUResManager *>(device_context_->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(res_manager_);
  initialized_ = true;
}

void GPUKernelExecutor::Destroy() {
  if (!initialized_) {
    return;
  }
  res_manager_ = nullptr;
  initialized_ = false;
}

void GPUKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_lazy_inline = ms_context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (enable_lazy_inline) {
    MS_LOG(EXCEPTION) << "GPU does not support the lazy_inline feature, "
                      << "please do not mark @lazy_inline in cell's __init__ func.";
  }
  if (kernel_graph->is_from_single_op()) {
    RunOpOptimize(kernel_graph);

    FormatTransformChecker::GetInstance().CheckSupportFormatTransform(kernel_graph);
    SetOperatorInfo(kernel_graph);

    RunOpHardwareOptimize(kernel_graph);

    RunOpHideNopNode(kernel_graph);
    RunOpRemoveNopNode(kernel_graph);
    UpdateKernelRefInfo(kernel_graph);
    AssignDefaultGpuStream(kernel_graph);
  } else {
    // Optimization pass which is irrelevant to device type or format.
    OptimizeGraphWithoutDeviceInfo(kernel_graph);

    FormatTransformChecker::GetInstance().CheckSupportFormatTransform(kernel_graph);
    SetOperatorInfo(kernel_graph);

    // SetOperatorInfo may generate new node, so need set kernel object type again.
    kernel_graph->SetKernelObjectTypesForUnrealNodes();
#ifdef ENABLE_DUMP_IR
    if (ms_context->CanDump(kIntroductory)) {
      DumpIR("hwopt_comm_after_kernel_select_" + graph->ToString() + ".ir", graph, true);
    }
#endif

    // Optimization pass which is relevant to device type or format.
    OptimizeGraphWithDeviceInfo(kernel_graph);

    // Run final optimization.
    opt::CommonFinalOptimization(kernel_graph);

    // Graph kernel fusion optimization
    if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
      constexpr char kGraphKernelOptimizeCallBackFunc[] = "GraphKernelOptimize";
      static auto graphkernel_optimize_callback =
        callback::CommonCallback::GetInstance().GetCallback<void, const KernelGraphPtr &>(
          kGraphKernelOptimizeCallBackFunc);
      if (graphkernel_optimize_callback) {
        graphkernel_optimize_callback(kernel_graph);
      }
      kernel_graph->SetExecOrderByDefault();
    }

    // Assign the stream and insert the send/recv node for all reduce kernel, so must be the last in the optimizer.
    device::gpu::AssignGpuStream(kernel_graph);
  }
}

void GPUKernelExecutor::UpdateKernelRefInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel);

    auto kernel_attr_list = kernel::NativeGpuKernelModFactory::GetInstance().GetGpuSupportedList(op_name);
    if (kernel_attr_list.empty()) {
      MS_LOG(DEBUG) << "kernel_attr_list is empty";
      return;
    }

    auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    // For the same kernel, there are currently no multiple Ref info.
    kernel_info->set_ref_map(kernel_attr_list[0].GetAllOutInRef(), kernel_attr_list[0].GetOutInRefMap());
  }
}

std::vector<size_t> GPUKernelExecutor::GetLaunchIgnoredInputAddressIdx(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto kernel_mod = kernel_info->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->GetLaunchIgnoredInputAddressIdx();
}

bool GPUKernelExecutor::IsLaunchIgnoredInputAddressIdx(const AnfNodePtr &node, size_t input_idx) const {
  auto ignored_input_list = GetLaunchIgnoredInputAddressIdx(node);
  if (std::find(ignored_input_list.begin(), ignored_input_list.end(), input_idx) != ignored_input_list.end()) {
    return true;
  }
  return false;
}

void GPUKernelExecutor::SetOperatorInfo(const KernelGraphPtr &graph) const {
  uint64_t start_time = profiler::GetClockSyscnt();
  auto mng = graph->manager();
  if (mng == nullptr) {
    mng = Manage(graph, true);
    graph->set_manager(mng);
  }
  bool do_expand = false;
  auto &node_list = graph->execution_order();
  for (auto &node : node_list) {
    const auto &failure_info = SetKernelInfoWithMsg(node);
    if (failure_info.first.empty()) {
      continue;
    }
    auto expand_ret = TryExpandFallback(graph, node, failure_info);
    if (expand_ret) {
      MS_LOG(INFO) << failure_info.first << " but expand success.";
      do_expand = true;
    } else {
      HandleKernelSelectFailure(graph, node, failure_info);
    }
  }
  if (do_expand) {
    opt::BindValueToGraph().Run(graph);
    graph->SetExecOrderByDefault();
  }
  (void)profiler::CollectHostInfo(kModelNameGPU, kEventOptimizeGraph, kStageSetKernelInfo, start_time,
                                  profiler::GetClockSyscnt(), 1);
}

kernel::KernelModPtr GPUKernelExecutor::CreateKernelMod(const std::string &op_name) const {
  return kernel::Factory<kernel::NativeGpuKernelMod>::Instance().Create(op_name);
}

void GPUKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  SetKernelInfoBeforeCreateKernel(nodes);
  CreateGPUKernel(nodes);
}

bool GPUKernelExecutor::LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                     void *stream) const {
  MS_EXCEPTION_IF_NULL(kernel);
  if (!res_manager_->BindDeviceToCurrentThread(false)) {
    return false;
  }
  bool ret = true;

  const auto &profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  if (!profiler_inst->GetEnableFlag() || !profiler_inst->GetOpTimeFlag()) {
    ret = DoLaunchKernel(kernel, inputs, workspace, outputs, kernel_mod, stream);
  } else {
    ret = LaunchKernelWithProfiling(kernel, inputs, workspace, outputs, kernel_mod, stream);
  }
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
    return false;
  }

  return ret;
}

bool GPUKernelExecutor::LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                                  void *stream) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(stream);

  auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(kernel->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  if (profiler::gpu::ProfilingUtils::IsFirstStep(kernel_graph->graph_id())) {
    profiler::gpu::ProfilingTraceInfo profiling_trace =
      profiler::gpu::ProfilingUtils::GetProfilingTraceFromEnv(NOT_NULL(kernel_graph.get()));
    profiler_inst->SetStepTraceOpName(profiling_trace);
  }

  profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), GPUDeviceManager::GetInstance().default_stream());
  bool ret = DoLaunchKernel(kernel, inputs, workspace, outputs, kernel_mod, stream);
  profiler_inst->OpDataProducerEnd();
  profiler_inst->RecordFrameWorkInfo(kernel);

  auto op_launch_start_end_time = profiler_inst->GetSingleOpLaunchTime();
  MS_LOG(DEBUG) << "Launch kernel:" << kernel->fullname_with_scope() << " cost:"
                << (op_launch_start_end_time.second - op_launch_start_end_time.first) / kBasicTimeTransferUnit;

  if (profiler_inst->GetSyncEnableFlag()) {
    CHECK_RET_WITH_RETURN_ERROR(res_manager_->SyncAllStreams(true), "Profiler SyncStream failed.");
  }
  return ret;
}

bool GPUKernelExecutor::DoLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                       void *stream) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  MS_EXCEPTION_IF_NULL(stream);

  uint64_t start_time = 0;
  PROFILER_START(start_time);
  auto ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
  // Sync running.
  bool sync_stream = runtime::RuntimeConf::GetInstance()->launch_blocking();
  if (sync_stream && !res_manager_->SyncAllStreams(true)) {
    return false;
  }
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
               kernel->fullname_with_scope(), false);

  return ret;
}

bool GPUKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                          const tensor::TensorPtrList &input_tensors,
                                          const tensor::TensorPtrList &output_tensors, const size_t &stream_id) const {
  auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_EXCEPTION_IF_NULL(stream);

  auto task_context =
    std::make_shared<runtime::KernelTaskContext>(device_context_, input_tensors, output_tensors, stream);

  auto task = GetTaskByTaskType(task_type, task_context);
  MS_EXCEPTION_IF_NULL(task);

  uint64_t start_time = 0;
  PROFILER_START(start_time);
  auto ret = task->RunWithRet();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Exec task failed, task_type:" << task_type;
  }

  // Sync running.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (runtime::RuntimeConf::GetInstance()->launch_blocking() && !res_manager_->SyncAllStreams(true)) {
    return false;
  }

  runtime::DeviceAddressUtils::ProcessCrossStreamAddress("Contiguous", device_context_, stream_id, input_tensors,
                                                         output_tensors);
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch, "Contiguous",
               false);

  return true;
}

namespace {
constexpr size_t kMaxDim = 9;
void MallocMemoryForDeviceAddress(device::DeviceAddress *device_address, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_address);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Graph", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "Graph", device::tracker::MemType::kContinuousMemory,
                                                 device_address->GetSize(), device_address);
  if (device_address->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_address)) {
      MS_LOG(EXCEPTION) << "Allocate device memory failed!";
    }
  }
}

void MallocMemoryAndCopyValue(const kernel::KernelTensorPtr &kernel_tensor, const device::DeviceContext *device_context,
                              std::vector<int64_t> vec) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const auto &device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Graph", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "Graph", device::tracker::MemType::kWorkSpace,
                                                 device_address->GetSize(), device_address.get());
  if (device_address->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate device memory failed!";
    }
  }

  std::reverse(vec.begin(), vec.end());
  vec.resize(kMaxDim, 0);
  ShapeVector shape{SizeToLong(device_address->GetSize() / sizeof(int64_t))};
  auto tensor = tensor::from_spec(kNumberTypeInt64, shape, device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(tensor->device_address());
  auto tensor_device_address = tensor->device_address();
  tensor_device_address->set_ptr(vec.data());
  tensor_device_address->SetSize(device_address->GetSize());
  tensor->set_format(Format::DEFAULT_FORMAT);
  DeviceContextKey host_key = {device_address->GetDeviceType(), device_address->device_id()};
  DeviceContext *host_context = DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  if (!host_context->device_res_manager_->SyncAllStreams(true) ||
      !SyncCopy(kernel_tensor.get(), tensor.get(), device_address->stream_id())) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed, vec:" << vec;
  }
}
}  // namespace

bool GPUKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                          const std::vector<KernelTensor *> &input_kernel_tensors,
                                          const std::vector<KernelTensor *> &output_kernel_tensors,
                                          const size_t &stream_id) const {
  if (task_type != runtime::KernelTaskType::kCONTIGUOUS_TASK) {
    MS_LOG(EXCEPTION) << "KernelTask not supported, task_type:" << task_type;
  }
  MS_LOG(DEBUG) << "Start gpu contiguous task.";

  const auto &input_kernel_tensor = input_kernel_tensors[0];
  const auto &input_address = input_kernel_tensor->device_address();
  const auto &input_storage_info = input_kernel_tensor->tensor_storage_info();

  const auto &output_kernel_tensor = output_kernel_tensors[0];
  const auto &output_address = output_kernel_tensor->device_address();

  auto stream = device_context_->device_res_manager_->GetStream(stream_id);
  auto device_name = device::GetDeviceNameByType(device_context_->device_context_key().device_type_);
  MS_EXCEPTION_IF_NULL(stream);

  MS_LOG(DEBUG) << "Input_storage_info:" << (input_storage_info == nullptr ? "" : input_storage_info->ToString())
                << ", input_address size:" << input_address->GetSize()
                << ", output_address size:" << output_address->GetSize();
  MallocMemoryForDeviceAddress(input_address.get(), device_context_);
  MallocMemoryForDeviceAddress(output_address.get(), device_context_);

  // Ensure address life cycle
  device::DeviceAddressPtr shape_dev_addr = nullptr;
  device::DeviceAddressPtr strides_dev_addr = nullptr;

  kernel::KernelTensorPtr shape_addr = nullptr;
  kernel::KernelTensorPtr strides_addr = nullptr;

  if (!input_storage_info->is_contiguous) {
    // No need shape_addr and strides_addr, when tensor is contiguous
    auto shape_kernel_tensor =
      AnfAlgo::CreateKernelTensor(nullptr, kMaxDim * sizeof(int64_t), Format::DEFAULT_FORMAT, kNumberTypeInt64,
                                  ShapeVector(), device_name, device_context_->device_context_key().device_id_);

    auto strides_kernel_tensor =
      AnfAlgo::CreateKernelTensor(nullptr, kMaxDim * sizeof(int64_t), Format::DEFAULT_FORMAT, kNumberTypeInt64,
                                  ShapeVector(), device_name, device_context_->device_context_key().device_id_);

    shape_dev_addr = shape_kernel_tensor->device_address();
    strides_dev_addr = strides_kernel_tensor->device_address();

    MallocMemoryAndCopyValue(shape_kernel_tensor, device_context_, input_storage_info->shape);
    MallocMemoryAndCopyValue(strides_kernel_tensor, device_context_, input_storage_info->strides);
  }

  kernel::ContiguousGpuKernel contiguous_kernel;
  auto ret = contiguous_kernel.LaunchContiguous(input_kernel_tensor->dtype_id(), input_address, input_storage_info,
                                                output_kernel_tensor->dtype_id(), output_address, shape_dev_addr,
                                                strides_dev_addr, stream);
  if (!ret) {
    MS_LOG(EXCEPTION) << "LaunchContiguous failed";
  }
  MS_LOG(DEBUG) << "End gpu contiguous task.";

  return true;
}

uint32_t GPUDeviceContext::GetDeviceCount() { return IntToUint(CudaDriver::device_count()); }

std::string GPUDeviceContext::GetDeviceName(uint32_t device_id) {
  return GPUdeviceInfo::GetInstance(device_id)->name();
}

std::tuple<int, int> GPUDeviceContext::GetDeviceCapability(uint32_t device_id) {
  int major_sm = GPUdeviceInfo::GetInstance(device_id)->major_sm();
  int minor_sm = GPUdeviceInfo::GetInstance(device_id)->minor_sm();
  return std::make_tuple(major_sm, minor_sm);
}

cudaDeviceProp GPUDeviceContext::GetDeviceProperties(uint32_t device_id) {
  return GPUdeviceInfo::GetInstance(device_id)->properties();
}

std::string GPUDeviceContext::GetArchList() { return STRING_COMPILE_OPT(CUDA_ARCH_LIST); }

MS_REGISTER_DEVICE(kGPUDevice, GPUDeviceContext);
#ifdef WITH_BACKEND
MSCONTEXT_REGISTER_INIT_FUNC(kGPUDevice, [](MsContext *ctx) -> void {
  MS_EXCEPTION_IF_NULL(ctx);
  if (ctx->backend_policy() != "ms") {
    ctx->set_backend_policy("ms");
  }
});
#endif

// Register functions to _c_expression so python hal module could call GPU device interfaces.
void PybindGPUStatelessFunc(py::module *m) {
  MS_EXCEPTION_IF_NULL(m);
  (void)py::class_<cudaDeviceProp>(*m, "cudaDeviceProp", py::module_local())
    .def_readonly("name", &cudaDeviceProp::name)
    .def_readonly("major", &cudaDeviceProp::major)
    .def_readonly("minor", &cudaDeviceProp::minor)
    .def_readonly("is_multi_gpu_board", &cudaDeviceProp::isMultiGpuBoard)
    .def_readonly("is_integrated", &cudaDeviceProp::integrated)
    .def_readonly("multi_processor_count", &cudaDeviceProp::multiProcessorCount)
    .def_readonly("total_memory", &cudaDeviceProp::totalGlobalMem)
    .def_readonly("warp_size", &cudaDeviceProp::warpSize)
    .def("__repr__", [](const cudaDeviceProp &p) {
      std::ostringstream s;
      s << "cudaDeviceProp(name='" << p.name << "', major=" << p.major << ", minor=" << p.minor
        << ", is_multi_gpu_board=" << p.isMultiGpuBoard << ", is_integrated=" << p.integrated
        << ", multi_processor_count=" << p.multiProcessorCount << ", total_memory=" << p.totalGlobalMem / (1024 * 1024)
        << "MB, warp_size=" << p.warpSize << ")";
      return s.str();
    });
  (void)m->def("gpu_get_device_count", &GPUDeviceContext::GetDeviceCount, "Get GPU device count.");
  (void)m->def("gpu_get_device_name", &GPUDeviceContext::GetDeviceName, "Get GPU device name of specified device id.");
  (void)m->def("gpu_get_device_capability", &GPUDeviceContext::GetDeviceCapability,
               "Get GPU major and minor capability of specified device id.");
  (void)m->def("gpu_get_device_properties", &GPUDeviceContext::GetDeviceProperties,
               "Get GPU device properties of specified device id.");
  (void)m->def("gpu_get_arch_list", &GPUDeviceContext::GetArchList, "Get GPU arch list of this MindSpore package.");

  RegGPUOpPrecisionConf(m);
  RegGPUOpTuningConf(m);
}
REGISTER_DEV_STATELESS_FUNC_CB(kGPUDevice, PybindGPUStatelessFunc);

}  // namespace gpu
}  // namespace device
}  // namespace mindspore
