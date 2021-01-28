/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "backend/session/gpu_session.h"

#include <string>
#include "backend/optimizer/common/helper.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/common/pass_manager.h"
#include "backend/optimizer/common/common_backend_optimization.h"
#include "backend/optimizer/gpu/adam_weight_decay_fusion.h"
#include "backend/optimizer/gpu/adam_fusion.h"
#include "backend/optimizer/gpu/apply_momentum_weight_scale_fusion.h"
#include "backend/optimizer/gpu/apply_momentum_scale_fusion.h"
#include "backend/optimizer/gpu/apply_momentum_weight_fusion.h"
#include "backend/optimizer/gpu/batch_norm_relu_fusion.h"
#include "backend/optimizer/gpu/batch_norm_relu_grad_fusion.h"
#include "backend/optimizer/gpu/batch_norm_add_relu_fusion.h"
#include "backend/optimizer/gpu/batch_norm_add_relu_grad_fusion.h"
#include "backend/optimizer/gpu/combine_momentum_fusion.h"
#include "backend/optimizer/gpu/combine_cast_fusion.h"
#include "backend/optimizer/gpu/cudnn_inplace_fusion.h"
#include "backend/optimizer/gpu/insert_format_transform_op.h"
#include "backend/optimizer/gpu/replace_momentum_cast_fusion.h"
#include "backend/optimizer/gpu/replace_addn_fusion.h"
#include "backend/optimizer/gpu/remove_format_transform_pair.h"
#include "backend/optimizer/gpu/remove_redundant_format_transform.h"
#include "backend/optimizer/gpu/reduce_precision_fusion.h"
#include "backend/optimizer/gpu/relu_v2_pass.h"
#include "backend/optimizer/gpu/add_relu_v2_fusion.h"
#include "backend/optimizer/gpu/add_relu_grad_v2_fusion.h"
#include "backend/optimizer/graph_kernel/add_atomic_clean_gpu.h"
#include "backend/optimizer/graph_kernel/arithmetic_simplify.h"
#include "backend/optimizer/graph_kernel/basic_ops_fusion.h"
#include "backend/optimizer/graph_kernel/clean_all_in_once.h"
#include "backend/optimizer/graph_kernel/depend_formater.h"
#include "backend/optimizer/graph_kernel/eliminate_redundant_output.h"
#include "backend/optimizer/graph_kernel/tensor_promotion.h"
#include "backend/optimizer/graph_kernel/graph_kernel_splitter.h"
#include "backend/optimizer/graph_kernel/graph_kernel_expander.h"
#include "backend/optimizer/graph_kernel/raise_reduction_precision.h"
#include "backend/optimizer/graph_kernel/graph_kernel_cse.h"
#include "backend/optimizer/graph_kernel/shape_ops_splitter.h"
#include "backend/optimizer/graph_kernel/value_graph_binder.h"
#include "backend/optimizer/graph_kernel/parallel_fusion.h"
#include "backend/optimizer/pass/communication_op_fusion.h"
#include "backend/optimizer/pass/getitem_tuple.h"
#include "common/trans.h"
#include "debug/data_dump/e2e_dump_util.h"
#include "debug/tensor_load.h"
#include "debug/dump_proto.h"
#include "runtime/device/gpu/gpu_kernel_build.h"
#include "runtime/device/gpu/gpu_kernel_runtime.h"
#include "runtime/device/gpu/gpu_stream_assign.h"
#include "runtime/device/gpu/kernel_info_setter.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/gpu/cuda_driver.h"
#include "runtime/device/gpu/distribution/collective_init.h"
#include "utils/ms_utils.h"
#include "utils/config_manager.h"
#include "utils/ms_context.h"
#if ENABLE_CPU && ENABLE_GPU
#include "ps/util.h"
#include "ps/ps_cache/ps_cache_manager.h"
#endif

namespace mindspore {
namespace session {
namespace gpu {
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
using CollectiveInitializer = device::gpu::CollectiveInitializer;
using GetLocalRankId = device::gpu::GetLocalRankId;

void GPUSession::Init(uint32_t device_id) {
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  bool collective_inited = CollectiveInitializer::instance().collective_inited();
  if (collective_inited && collective_handle_ != nullptr) {
    auto get_local_rank_funcptr =
      reinterpret_cast<GetLocalRankId>(dlsym(const_cast<void *>(collective_handle_), "local_rank_id"));
    MS_EXCEPTION_IF_NULL(get_local_rank_funcptr);
    device_id = IntToUint((*get_local_rank_funcptr)());
  }
  bool ret = device::gpu::CudaDriver::set_current_device(UintToInt(device_id));
  if (!ret) {
    MS_LOG(EXCEPTION) << "GPUSession failed to set current device id.";
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);

  MS_LOG(INFO) << "Set device id " << device_id << " for gpu session.";
  InitExecutor(kGPUDevice, device_id);
}

void GPUSession::SelectKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  device::gpu::FormatTransformChecker::GetInstance().CheckSupportFormatTransform(kernel_graph);
  for (const auto &kernel_node : kernel_graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    device::gpu::SetKernelInfo(kernel_node);
  }
}

void GPUSession::StartKernelRT() const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Init()) {
    MS_LOG(EXCEPTION) << "GPU start kernel runtime failed";
  }
}

void GPUSession::Optimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamWeightDecayFusion>());
  pm->AddPass(std::make_shared<opt::AdamFusion>());
  pm->AddPass(std::make_shared<opt::ApplyMomentumWeightDecayScaleFusion>());
  pm->AddPass(std::make_shared<opt::ApplyMomentumScaleFusion>());
  pm->AddPass(std::make_shared<opt::ApplyMomentumWeightDecayFusion>());
  if (!(context_ptr->get_param<bool>(MS_CTX_ENABLE_GRAPH_KERNEL))) {
    pm->AddPass(std::make_shared<opt::CastAllFusion>("cast_all"));
  }
  pm->AddPass(std::make_shared<opt::CombineMomentumFusion>("combine_momentum"));
  pm->AddPass(std::make_shared<opt::ReplaceMomentumCastFusion>());
  pm->AddPass(std::make_shared<opt::ReplaceAddNFusion>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNormReluFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormReluGradFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormAddReluFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormAddReluGradFusion>());
  pm->AddPass(std::make_shared<opt::InsertFormatTransformOp>());
  pm->AddPass(std::make_shared<opt::RemoveFormatTransformPair>());
  pm->AddPass(std::make_shared<opt::RemoveRedundantFormatTransform>());
  pm->AddPass(std::make_shared<opt::CudnnInplaceAggregate>());
  pm->AddPass(std::make_shared<opt::ReluV2Pass>());
  pm->AddPass(std::make_shared<opt::AddReluV2Fusion>());
  pm->AddPass(std::make_shared<opt::AddReluGradV2Fusion>());
  pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  pm->AddPass(std::make_shared<opt::GetitemTuple>());
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::RunOpHardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!(context_ptr->get_param<bool>(MS_CTX_ENABLE_GRAPH_KERNEL))) {
    return;
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>("graph_kernel_pm");
  std::vector<PrimitivePtr> duplicated_ops = {prim::kPrimReshape, prim::kPrimExpandDims, prim::kPrimCast};
  pm->AddPass(std::make_shared<opt::DependFormater>());  // Make more fusion opportunity.
  pm->AddPass(std::make_shared<opt::GraphKernelExpander>());
  pm->AddPass(std::make_shared<opt::ShapeOpsSplitter>(duplicated_ops));
  pm->AddPass(std::make_shared<opt::BasicOpsFusion>());
  pm->AddPass(std::make_shared<opt::EliminateRedundantOutput>());
  pm->AddPass(std::make_shared<opt::RaiseReductionPrecision>());
  pm->AddPass(std::make_shared<opt::GraphKernelCSE>(duplicated_ops));
  pm->AddPass(std::make_shared<opt::ArithmeticSimplify>());
  pm->AddPass(std::make_shared<opt::GraphKernelCSE>(duplicated_ops));
  pm->AddPass(std::make_shared<opt::TensorPromotion>());
  pm->AddPass(std::make_shared<opt::GraphKernelSplitter>());
  pm->AddPass(std::make_shared<opt::GraphKernelCSE>());
  // The CSE may output a graph with repeated outputs.
  pm->AddPass(std::make_shared<opt::EliminateRedundantOutput>());
  // After Simplify and Splitter, a lot of redundant getitem/maketuple
  // will be exposed, use GetitemTuple Pass to delete them.
  pm->AddPass(std::make_shared<opt::GetitemTuple>());
  pm->AddPass(std::make_shared<opt::AtomicCleanInsertter>());
  pm->AddPass(std::make_shared<opt::DependFormater>());  // Prevent fake loop in parallel fusion.
  pm->AddPass(std::make_shared<opt::ParallelOpFusion>(kGPUDevice, opt::ParallelConfig(7)));
  pm->AddPass(std::make_shared<opt::BindValueToGraph>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::AssignStream(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  device::gpu::AssignGpuStream(kernel_graph);
}

void GPUSession::BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  device::gpu::GpuBuild(kernel_graph);
}

void GPUSession::AllocateMemory(KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignMemory(kernel_graph);
}

void GPUSession::RunOpAllocateMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                     KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(input_tensors, kernel_graph);
}

void GPUSession::RunOpClearMemory(KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpClearMemory(kernel_graph);
}

void GPUSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                               const std::vector<tensor::TensorPtr> &inputs_const) const {
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &input_nodes = kernel_graph->input_nodes();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (inputs.size() != input_nodes.size()) {
    MS_LOG(EXCEPTION) << "Tensor input:" << inputs.size() << " is not equal graph inputs:" << input_nodes.size();
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<Parameter>() && AnfAlgo::OutputAddrExist(input_node, 0)) {
#if ENABLE_CPU && ENABLE_GPU
      const std::string &param_name = input_node->fullname_with_scope();
      if (ps::ps_cache_instance.IsHashTable(param_name)) {
        continue;
      }
#endif
      auto pk_node = input_node->cast<ParameterPtr>();
      auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
      auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
      bool need_sync = false;
      if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
        if (tensor_address == nullptr || tensor_address != device_address) {
          need_sync = true;
        }
      } else if (tensor->NeedSyncHostToDevice() || tensor_address == nullptr) {
        need_sync = true;
      } else if (tensor_address != device_address) {
        if (tensor_address->DeviceType() == device_address->DeviceType()) {
          AnfAlgo::SetOutputAddr(tensor_address, 0, pk_node.get());
        } else {
          need_sync = true;
        }
      }
      if (need_sync) {
        if (AnfAlgo::IsParameterWeight(input_node->cast<ParameterPtr>()) ||
            ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
          tensor->set_device_address(device_address);
        }
        MS_EXCEPTION_IF_NULL(device_address);
        if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                              LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                              tensor->data_c())) {
          MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
        }
      }
    }
    tensor->set_sync_status(kNoNeedSync);
  }
}

void GPUSession::Execute(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Run(kernel_graph.get(), false)) {
    MS_LOG(EXCEPTION) << "GPU execute graph failed!";
  }
}

GraphId GPUSession::CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  // Construct graph, if successfully, graph_sum_ + 1
  auto graph = ConstructKernelGraph(lst, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  return CompileGraphImpl(graph);
}

GraphId GPUSession::CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) {
  std::vector<KernelGraphPtr> all_graphs;
  auto root_graph = ConstructKernelGraph(func_graph, &all_graphs);
  MS_EXCEPTION_IF_NULL(root_graph);
  if (all_graphs.size() != 1) {
    MS_LOG(EXCEPTION) << "Gpu backend does not support multi-graph schedule. graph num" << all_graphs.size();
  }

  opt::BackendCommonOptimization(root_graph);
  return CompileGraphImpl(root_graph);
}

GraphId GPUSession::CompileGraphImpl(KernelGraphPtr graph) {
  // Prepare ms context info for dump .pb graph
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  // Dump .pb graph before graph optimization
  if (save_graphs) {
    DumpIRProto(graph, "before_opt_" + std::to_string(graph->graph_id()));
  }
  // Graph optimization irrelevant to device data format
  Optimize(graph);
  // Select kernel build info
  SelectKernel(graph);
  // Graph optimization relevant to device data format
  HardwareOptimize(graph);
  // Graph kernel fusion optimization
  GraphKernelOptimize(graph);
  // Start gpu kernel runtime
  StartKernelRT();
#if ENABLE_CPU && ENABLE_GPU
  InitPsWorker(graph);
#endif
  // Assign CUDA streams
  AssignStream(graph);
  // Dump .pb graph before remove nop nodes
  if (save_graphs) {
    DumpIRProto(graph, "before_removeNop_" + std::to_string(graph->graph_id()));
  }
  // Update Graph Dynamic Shape Attr.
  UpdateGraphDynamicShapeAttr(NOT_NULL(graph));
  graph->UpdateGraphDynamicAttr();
  // Hide NopOp from execution graph
  opt::HideNopNode(graph.get());
  // Build kernel if node is cnode
  BuildKernel(graph);
  // Set graph execution order before memory alloc, ensure that memory alloc is according to the reorder graph
  auto execution_order = graph->execution_order();
  Reorder(&execution_order);
  graph->set_execution_order(execution_order);
  // Get summary nodes.
  SetSummaryNodes(graph.get());
  // Dump .pb graph after graph optimization
  if (save_graphs) {
    DumpIRProto(graph, "after_opt_" + std::to_string(graph->graph_id()));
  }
  // Set graph manager.
  MS_EXCEPTION_IF_NULL(context_);
  FuncGraphManagerPtr manager = MakeManager({graph});
  context_->AddManager(manager);
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  // Alloc memory, including static memory and dynamic memory
  AllocateMemory(graph.get());

#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
    debugger_->LoadGraphs(graph);
  }
#endif
  MS_LOG(INFO) << "CompileGraph graph_id: " << graph->graph_id();
  return graph->graph_id();
}

void GPUSession::RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                              VectorRef *outputs) {
  auto &kernel_graph = graphs_[graph_id];
  MS_LOG(INFO) << "RunGraph graph_id: " << graph_id;
  // In pynative mode, device addresses of tensors in value nodes change.
  SyncValueNodeDeviceAddr(kernel_graph);
  // Load input data from user input
  LoadInputData(kernel_graph, inputs);
  if (debugger_) {
    debugger_->PreExecute(kernel_graph, graph_sum_);
  }
#if ENABLE_CPU && ENABLE_GPU
  // Initialize parameter server
  InitPSParamAndOptim(kernel_graph, inputs);
#endif
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // It's InitDataset graph if kernel_num == 1, skip the loop.
  int kernel_num = kernel_graph->execution_order().size();
  int64_t loopsize = (kernel_num > 1) ? ConfigManager::GetInstance().gpu_loopsink_size() : 1;
  for (int64_t i = 0; i < loopsize; i++) {
#if ENABLE_CPU && ENABLE_GPU
    std::string channel_name;
    if (ps::PsDataPrefetch::GetInstance().cache_enable() && IsGetNextGraph(graph_id, &channel_name)) {
      ps::ps_cache_instance.IncreaseGraphStep(channel_name);
    }
#endif
    Execute(kernel_graph);
  }
  // In pynative mode, device addresses of tensors in value nodes need be clean.
  CleanValueNodeDeviceAddr(kernel_graph);
  // Summary
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_GPU_SUMMARY)) {
    Summary(kernel_graph.get());
  }
  PostIterationDbg(kernel_graph);
}

void GPUSession::BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                             const std::vector<tensor::TensorPtr> &input_tensors,
                             const std::vector<int64_t> &tensors_mask) {
  // Check if the graph cache exists.
  if (run_op_graphs_.find(graph_info) != run_op_graphs_.end()) {
    return;
  }
  // Prepare the graph
  auto kernel_graph = ConstructSingleOpGraph(op_run_info, input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  SelectKernel(kernel_graph);
  RunOpHardwareOptimize(kernel_graph);
  StartKernelRT();
  RunOpHideNopNode(kernel_graph);
  BuildKernel(kernel_graph);
  run_op_graphs_[graph_info] = kernel_graph;
}

void GPUSession::RunOpImpl(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                           std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                           const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(op_run_info);
  BuildOpImpl(*op_run_info, graph_info, *input_tensors, tensors_mask);
  EraseValueNodeTensor(tensors_mask, input_tensors);
  // run op
  auto kernel_graph = run_op_graphs_[graph_info];
  MS_EXCEPTION_IF_NULL(kernel_graph);
  RunOpRemoveNopNode(kernel_graph);
  RunOpAllocateMemory(*input_tensors, kernel_graph.get());
  // Execute the computation
  LoadInputData(kernel_graph, *input_tensors);
  Execute(kernel_graph);
  // Fetch outputs
  UpdateOutputs(kernel_graph, outputs, *input_tensors);
  // update output abstract of dynamic op to op_run_info
  if (op_run_info->is_dynamic_shape) {
    UpdateOutputAbstract(kernel_graph, op_run_info);
  }
  RunOpClearMemory(kernel_graph.get());
}

void GPUSession::Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  if (debugger_->DebuggerBackendEnabled()) {
    MS_EXCEPTION_IF_NULL(kernel_graph);
    E2eDumpUtil::DumpData(kernel_graph.get(), device_id_, debugger_.get());
  } else {
    DumpJsonParser::GetInstance().UpdateDumpIter();
  }
}

bool GPUSession::DumpDataEnabledIteration() const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  return runtime_instance->DumpDataEnabledIteration();
}

void GPUSession::PostIterationDbg(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  bool dump_enabled = DumpDataEnabledIteration();
  // debug used for dump
  if (debugger_ && dump_enabled) {
    Dump(kernel_graph);
  } else {
    DumpJsonParser::GetInstance().UpdateDumpIter();
  }
  if (debugger_) {
    debugger_->PostExecute();
  }
}

void GPUSession::SyncValueNodeDeviceAddr(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    return;
  }
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->SyncValueNodeDeviceAddr(kernel_graph.get());
}

void GPUSession::CleanValueNodeDeviceAddr(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    return;
  }
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->CleanValueNodeDeviceAddr(kernel_graph.get());
}

void GPUSession::SyncStream() {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->SyncStream();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
}
}  // namespace gpu
}  // namespace session
}  // namespace mindspore
