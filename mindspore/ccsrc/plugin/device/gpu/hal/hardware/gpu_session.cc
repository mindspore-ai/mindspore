/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/hal/hardware/gpu_session.h"

#include <string>
#include <utility>
#include "backend/common/optimizer/helper.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pass_manager.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "plugin/device/gpu/optimizer/adam_weight_decay_fusion.h"
#include "plugin/device/gpu/optimizer/adam_fusion.h"
#include "plugin/device/gpu/optimizer/alltoall_fusion.h"
#include "plugin/device/gpu/optimizer/apply_momentum_weight_scale_fusion.h"
#include "plugin/device/gpu/optimizer/apply_momentum_scale_fusion.h"
#include "plugin/device/gpu/optimizer/apply_momentum_weight_fusion.h"
#include "plugin/device/gpu/optimizer/batch_norm_relu_fusion.h"
#include "plugin/device/gpu/optimizer/batch_norm_relu_grad_fusion.h"
#include "plugin/device/gpu/optimizer/batch_norm_add_relu_fusion.h"
#include "plugin/device/gpu/optimizer/post_batch_norm_add_relu_fusion.h"
#include "plugin/device/gpu/optimizer/batch_norm_add_relu_grad_fusion.h"
#include "plugin/device/gpu/optimizer/combine_momentum_fusion.h"
#include "plugin/device/gpu/optimizer/combine_cast_fusion.h"
#include "plugin/device/gpu/optimizer/cudnn_inplace_fusion.h"
#include "plugin/device/gpu/optimizer/insert_format_transform_op.h"
#include "plugin/device/gpu/optimizer/replace_momentum_cast_fusion.h"
#include "plugin/device/gpu/optimizer/replace_addn_fusion.h"
#include "plugin/device/gpu/optimizer/print_reduce_fusion.h"
#include "plugin/device/gpu/optimizer/bce_with_logits_loss_fusion.h"
#include "plugin/device/gpu/optimizer/remove_format_transform_pair.h"
#include "plugin/device/gpu/optimizer/remove_redundant_format_transform.h"
#include "plugin/device/gpu/optimizer/reduce_precision_fusion.h"
#include "plugin/device/gpu/optimizer/insert_cast_gpu.h"
#include "plugin/device/gpu/optimizer/relu_v2_pass.h"
#include "plugin/device/gpu/optimizer/add_relu_v2_fusion.h"
#include "plugin/device/gpu/optimizer/add_relu_grad_v2_fusion.h"
#include "plugin/device/gpu/optimizer/matmul_biasadd_fusion.h"
#include "plugin/device/gpu/optimizer/neighbor_exchange_v2_fusion.h"
#include "plugin/device/gpu/optimizer/clip_by_norm_fission.h"
#ifdef ENABLE_GPU_INFER
#include "plugin/device/gpu/optimizer/trt_pass/graph_converter.h"
#endif
#include "backend/common/graph_kernel/adapter/graph_kernel_optimization.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "plugin/device/gpu/optimizer/concat_outputs_for_all_gather.h"
#include "backend/common/pass/getitem_tuple.h"
#include "backend/common/pass/optimize_updatestate.h"
#include "backend/common/pass/adjust_depend_for_parallel_optimizer_recompute_all_gather.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#ifdef ENABLE_DEBUGGER
#include "debug/data_dump/e2e_dump.h"
#include "debug/data_dump/dump_json_parser.h"
#include "debug/debugger/proto_exporter.h"
#include "debug/data_dump/dump_utils.h"
#include "debug/tensor_load.h"
#else
#include "debug/debugger/proto_exporter_stub.h"
#endif
#include "plugin/device/gpu/hal/device/gpu_kernel_build.h"
#include "plugin/device/gpu/hal/device/gpu_kernel_runtime.h"
#include "plugin/device/gpu/hal/device/gpu_stream_assign.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "distributed/init.h"
#include "plugin/device/gpu/hal/device/gpu_device_address.h"
#include "utils/ms_utils.h"
#include "include/common/utils/config_manager.h"
#include "utils/ms_context.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "include/common/utils/utils.h"
#include "abstract/utils.h"
#include "kernel/graph_kernel_info.h"
#ifdef WITH_BACKEND
#include "ps/util.h"
#include "ps/ps_context.h"
#endif

#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif

namespace mindspore {
namespace session {
namespace gpu {
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

void GPUSession::Init(uint32_t device_id) {
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
    device_id = distributed::collective::CollectiveManager::instance()->local_rank_id();
  }
  bool ret = device::gpu::CudaDriver::SetDevice(UintToInt(device_id));
  if (!ret) {
    MS_LOG(EXCEPTION) << "GPUSession failed to set current device id:" << device_id;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, device_id);
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
#ifndef _WIN32
    rank_id_ = GetRankId();
#else
    MS_LOG(EXCEPTION) << "windows not support nccl.";
#endif
  }
#ifndef ENABLE_SECURITY
  auto &json_parser = DumpJsonParser::GetInstance();
  // Dump json config file if dump is enabled for GPU old runtime.
  json_parser.CopyDumpJsonToDir(rank_id_);
  json_parser.CopyMSCfgJsonToDir(rank_id_);
#endif
  MS_LOG(INFO) << "Set device id " << device_id << " for gpu session.";
  InitExecutor(kGPUDevice, device_id);
}

void GPUSession::SelectKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  device::gpu::FormatTransformChecker::GetInstance().CheckSupportFormatTransform(kernel_graph);
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kGPUDevice);
  for (const auto &kernel_node : kernel_graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_info_setter->SetKernelInfo(kernel_node, KernelType::UNKNOWN_KERNEL_TYPE);
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
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
#ifdef ENABLE_GPU_INFER
  pm->AddPass(std::make_shared<opt::GraphConverter>());
#endif
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
  pm->AddPass(std::make_shared<opt::CombineMomentumFusion>("combine_momentum"));
  pm->AddPass(std::make_shared<opt::ReplaceMomentumCastFusion>());
  pm->AddPass(std::make_shared<opt::ReplaceAddNFusion>());
  pm->AddPass(std::make_shared<opt::PrintReduceFusion>("print_reduce"));
  pm->AddPass(std::make_shared<opt::BCEWithLogitsLossFusion>());
  pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
  pm->AddPass(std::make_shared<opt::NeighborExchangeV2Fusion>());
  pm->AddPass(std::make_shared<opt::NeighborExchangeV2GradFusion>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNormReluFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormReluGradFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormAddReluFusion>());
  pm->AddPass(std::make_shared<opt::PostBatchNormAddReluFusion>());
  pm->AddPass(std::make_shared<opt::BatchNormAddReluGradFusion>());
  pm->AddPass(std::make_shared<opt::InsertFormatTransformOp>());
  pm->AddPass(std::make_shared<opt::RemoveFormatTransformPair>());
  pm->AddPass(std::make_shared<opt::RemoveRedundantFormatTransform>());
  // Remove node only used by UpdateState, in order to ensure the correct execution sequence in CudnnInplaceAggregate.
  pm->AddPass(std::make_shared<opt::OptimizeUpdateState>());
  pm->AddPass(std::make_shared<opt::CudnnInplaceAggregate>());
  pm->AddPass(std::make_shared<opt::ReluV2Pass>());
  pm->AddPass(std::make_shared<opt::AddReluV2Fusion>());
  pm->AddPass(std::make_shared<opt::AddReluGradV2Fusion>());
  pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  pm->AddPass(std::make_shared<opt::AdjustDependForParallelOptimizerRecomputeAllGather>());
  pm->AddPass(std::make_shared<opt::AllGatherFusion>());
  pm->AddPass(std::make_shared<opt::ConcatOutputsForAllGather>());
  pm->AddPass(std::make_shared<opt::GetitemTuple>());
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::RunOpOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BCEWithLogitsLossFusion>());
  pm->AddPass(std::make_shared<opt::InsertCastGPU>("insert_cast_gpu"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::RunOpHardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ReducePrecisionFusion>("reduce_precision"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  if (!graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    return;
  }
  graphkernel::GraphKernelOptimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::AssignStream(const std::shared_ptr<KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  device::gpu::AssignGpuStream(kernel_graph);
}

void GPUSession::BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  auto kernels = kernel_graph->execution_order();
  device::gpu::CreateGPUKernel(kernels);
}

void GPUSession::AllocateMemory(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignMemory(*kernel_graph);
}

void GPUSession::RunOpAllocateMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                     const KernelGraph *kernel_graph, bool is_gradient_out) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(input_tensors, *kernel_graph, is_gradient_out);
}

void GPUSession::RunOpGenKernelEvent(const KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->GenKernelEvents(*graph);
}

void GPUSession::RunOpClearMemory(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpClearMemory(*kernel_graph);
}

namespace {
constexpr auto kAssignInputSize = 3;
constexpr auto kAssignUpdateIndex = 1;
bool UpdatedByAssign(const KernelGraphPtr &kernel_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto manager = kernel_graph->manager();
  if (manager == nullptr) {
    return false;
  }
  auto &node_users = manager->node_users();
  auto iter = node_users.find(node);
  if (iter == node_users.end()) {
    return false;
  }
  auto &users = iter->second;
  return std::any_of(users.begin(), users.end(), [](const std::pair<AnfNodePtr, int64_t> &user) {
    MS_EXCEPTION_IF_NULL(user.first);
    auto output_cnode = user.first->cast<CNodePtr>();
    return output_cnode != nullptr && IsPrimitiveCNode(output_cnode, prim::kPrimAssign) &&
           user.second == kAssignUpdateIndex && output_cnode->inputs().size() > kAssignInputSize;
  });
}

size_t UpdateGraphInputAbstract(const AnfNodePtr input_node, const tensor::TensorPtr tensor) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(tensor);
  size_t size = LongToSize(tensor->data().nbytes());
  if (!input_node->isa<Parameter>()) {
    return size;
  }
  auto input_param = input_node->cast<ParameterPtr>();
  if (input_param != nullptr && input_param->has_dynamic_shape()) {
    auto tensor_shape = tensor->shape();
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(input_node, 0)},
                                                {tensor_shape}, input_node.get());
    size = abstract::ShapeSize(tensor_shape) * abstract::TypeIdSize(tensor->data_type());
  }
  return size;
}

bool CheckIfNeedSync(const tensor::TensorPtr &tensor, const DeviceAddressPtr &device_address,
                     const ParameterPtr &pk_node) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(pk_node);
  auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  bool need_sync = false;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    if (tensor_address == nullptr || tensor_address != device_address) {
      need_sync = true;
    }
  } else if (tensor->NeedSyncHostToDevice() || tensor_address == nullptr) {
    need_sync = true;
  } else if (tensor_address != device_address) {
    if (tensor_address->GetDeviceType() == device_address->GetDeviceType()) {
      AnfAlgo::SetOutputAddr(tensor_address, 0, pk_node.get());
    } else {
      need_sync = true;
    }
  }
  return need_sync;
}
}  // namespace

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
      auto pk_node = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(pk_node);
      if (!pk_node->IsUsedByRealKernelInGraph(kernel_graph->graph_id())) {
        MS_LOG(INFO) << "Kernel graph inputs have anfnode which has no user.";
        tensor->set_sync_status(kNoNeedSync);
        continue;
      }
      auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
      MS_EXCEPTION_IF_NULL(device_address);
      bool need_sync = CheckIfNeedSync(tensor, device_address, pk_node);
      if (need_sync) {
        if (common::AnfAlgo::IsParameterWeight(pk_node) || UpdatedByAssign(kernel_graph, input_node) ||
            ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
          tensor->set_device_address(device_address);
        }
        auto size = UpdateGraphInputAbstract(input_node, tensor);
        if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0), size, tensor->data_type(),
                                              tensor->data_c())) {
          MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
        }
        if (kernel_graph->IsUpdatedParameter(pk_node)) {
          tensor->SetIsUpdateByDevice();
        }
      }
    }
    tensor->set_sync_status(kNoNeedSync);
  }
}

GraphId GPUSession::CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  // Construct graph, if successfully, graph_sum_ + 1
  auto graph = ConstructKernelGraph(lst, outputs, DeviceType::kGPU);
  MS_EXCEPTION_IF_NULL(graph);
  return CompileGraphImpl(graph);
}

GraphId GPUSession::CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) {
  std::vector<KernelGraphPtr> all_graphs;
  auto root_graph = ConstructKernelGraph(func_graph, &all_graphs, DeviceType::kGPU);
  MS_EXCEPTION_IF_NULL(root_graph);
  if (all_graphs.size() != 1) {
    MS_LOG(EXCEPTION) << "Gpu backend does not support multi-graph schedule, graph num is " << all_graphs.size();
  }
  // Insert maketuple graph output in case of multi-outputs.
  // The ConvertTupleOutputToMaketuple pass will insert TupleGetItem.
  AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(root_graph));
  SessionBasic::UnifyMindIR(root_graph);
  opt::BackendCommonOptimization(root_graph);
  return CompileGraphImpl(root_graph);
}

GraphId GPUSession::CompileGraphImpl(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // Prepare ms context info for dump .pb graph for GPU old runtime.
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
#ifndef ENABLE_SECURITY
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
#endif
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    // Dump .pb graph before graph optimization
    DumpIRProto(graph, "before_opt_" + std::to_string(graph->graph_id()));
  }
#endif
  // Graph optimization irrelevant to device data format
  Optimize(graph);
  // Select kernel build info
  SelectKernel(graph);
  // Graph optimization relevant to device data format
  HardwareOptimize(graph);
  // Run final optimization
  FinalOptimize(graph);
  // Graph kernel fusion optimization
  GraphKernelOptimize(graph);
  // Start gpu kernel runtime
  StartKernelRT();
  // Assign CUDA streams
  AssignStream(graph);
#ifdef ENABLE_DUMP_IR
  // Dump .pb graph before remove nop nodes
  if (context_ptr->CanDump(kIntroductory)) {
    DumpIRProto(graph, "before_removeNop_" + std::to_string(graph->graph_id()));
  }
#endif
  opt::AddDynamicShapeAttrPass(graph);
  const bool pynative_mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
  // Hide NopOp from execution graph in graph mode
  if (!pynative_mode) {
    opt::HideNopNode(graph.get());
  }
  // Build kernel if node is cnode
  BuildKernel(graph);
#ifndef ENABLE_SECURITY
  // Get summary nodes.
  SetSummaryNodes(graph.get());
#endif
  // Dump .pb graph after graph optimization
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    DumpIRProto(graph, "after_opt_" + std::to_string(graph->graph_id()));
  }
#endif
#ifndef ENABLE_SECURITY
  // GPU old runtime.
  if (json_parser.e2e_dump_enabled()) {
    graph->set_root_graph_id(graph->graph_id());
    std::string final_graph = "trace_code_graph_" + std::to_string(graph->graph_id());
    std::string root_dir = json_parser.path() + "/rank_" + std::to_string(rank_id_);
    std::string target_dir = root_dir + "/graphs";
    std::string ir_file_path = target_dir + "/" + "ms_output_" + final_graph + ".ir";
    DumpIRProtoWithSrcInfo(graph, final_graph, target_dir, kDebugWholeStack);
    DumpIR("trace_code_graph", graph, true, kWholeStack, ir_file_path);
    DumpGraphExeOrder("ms_execution_order_graph_" + std::to_string(graph->graph_id()) + ".csv", root_dir,
                      graph->execution_order());
  }
#endif
  // Set graph manager.
  MS_EXCEPTION_IF_NULL(context_);
  FuncGraphManagerPtr manager = MakeManager({graph});
  context_->AddManager(manager);
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }

  // Alloc memory in graph mode, including static memory and dynamic memory
  if (!pynative_mode) {
    AllocateMemory(graph.get());
  }

  DumpGraphs({graph});

#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
    debugger_->LoadGraphs(graph);
  }
#endif
  MS_LOG(INFO) << "CompileGraph graph_id: " << graph->graph_id();
  return graph->graph_id();
}

// GPU old runtime.
void GPUSession::PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                 const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
#ifdef ENABLE_DEBUGGER
  if (debugger_) {
    debugger_->PreExecute(kernel_graph);
  }

  E2eDump::UpdateIterOldRTDump(kernel_graph.get());
#endif
}

// GPU old runtime.
void GPUSession::PostExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                  const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  // Summary
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifndef ENABLE_SECURITY
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_GPU_SUMMARY)) {
    Summary(kernel_graph.get());
  }
#endif
#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->DebuggerBackendEnabled()) {
    debugger_->LoadParametersAndConst(kernel_graph);
  }

  // debug used for dump
  if (debugger_ && debugger_->CheckDebuggerDumpEnabled()) {
    Dump(kernel_graph);
  }

  if (debugger_) {
    debugger_->PostExecute();
  }
#endif
}

void GPUSession::ExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph) {
  int kernel_num = kernel_graph->execution_order().size();
  int64_t loopsize = (kernel_num > 1) ? ConfigManager::GetInstance().gpu_loopsink_size() : 1;
  for (int64_t i = 0; i < loopsize; i++) {
    Execute(kernel_graph);
  }
}

void GPUSession::UpdateOutputTensors(const VectorRef *outputs,
                                     const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                                     std::map<DeviceAddressPtr, DeviceAddressPtr> *new_to_old_device_address) {
  MS_EXCEPTION_IF_NULL(outputs);
  for (const auto &item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      const auto &vector_ref = utils::cast<VectorRef>(item);
      UpdateOutputTensors(&vector_ref, tensor_to_node, new_to_old_device_address);
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      const auto &tensor = utils::cast<tensor::TensorPtr>(item);
      MS_EXCEPTION_IF_NULL(tensor);
      const auto &iter = tensor_to_node.find(tensor);
      if (iter != tensor_to_node.end()) {
        const auto &node = iter->second.first;
        const auto &output_index = iter->second.second;
        MS_EXCEPTION_IF_NULL(node);
        // When the parameter does not have a user in the graph and is used as an output, the device address is null,
        // and there is no need to set the device address for tensor.
        if (!AnfAlgo::OutputAddrExist(node, output_index, true)) {
          continue;
        }
        auto address = AnfAlgo::GetMutableOutputAddr(node, output_index);
        // The outputs may have the same tensor, so need skip when the tensor has been set to device address.
        if ((address == nullptr) || (address->GetPtr() == nullptr)) {
          // If the device address in the node is invalid, you need to find out whether there is a corresponding
          // device address in the new to old device address map to check whether the device address in the node
          // has been replaced with a new one.
          if ((*new_to_old_device_address).find(address) != (*new_to_old_device_address).end()) {
            address = (*new_to_old_device_address)[address];
          } else {
            continue;
          }
        }
        tensor->set_device_address(address);

        // When the device address of graph cnode output is set in tensor, the graph output need be set new device
        // address, to avoid that the device address context of tensor be rewritten in the next step or next loop.
        // But one time memory application scenarios need to be skipped, because the memory is not allocated next step:
        // 1. Non cnode 2. Communication kernel.
        bool ps_mode = false;
#ifdef WITH_BACKEND
        ps_mode = ps::PSContext::instance()->is_ps_mode();
#endif
        if (node->isa<CNode>() && !common::AnfAlgo::IsCommunicationOp(node) && !ps_mode) {
          auto new_address = std::make_shared<device::gpu::GPUDeviceAddress>(nullptr, address->GetSize());
          // If a nop node is output, its previous node should be set.
          if (common::AnfAlgo::IsNopNode(node)) {
            auto pre_node = common::AnfAlgo::GetPrevNodeOutput(node, 0, true);
            if (!pre_node.first->isa<Parameter>()) {
              AnfAlgo::SetOutputAddr(new_address, pre_node.second, pre_node.first.get());
            }
          } else {
            AnfAlgo::SetOutputAddr(new_address, output_index, node.get());
          }
          (*new_to_old_device_address)[new_address] = address;
          if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
            auto runtime_instance =
              device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
            MS_EXCEPTION_IF_NULL(runtime_instance);
            auto gpu_runtime_instance = dynamic_cast<device::gpu::GPUKernelRuntime *>(runtime_instance);
            gpu_runtime_instance->SetAddrInvalid(address);
          }
        }

        if (common::AnfAlgo::IsDynamicShape(node)) {
          const auto &updated_shape = common::AnfAlgo::GetOutputInferShape(node, output_index);
          tensor->set_shape(updated_shape);
        }
      }
      if (tensor->NeedSyncDeviceToHostImmediately()) {
        tensor->data_sync(false);
        tensor->set_device_address(nullptr);
        tensor->set_sync_status(kNeedSyncHostToDevice);
      }
    }
  }
}

void GPUSession::Execute(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Run(*kernel_graph, false)) {
    MS_LOG(EXCEPTION) << "GPU execute graph failed!";
  }
}

KernelGraphPtr GPUSession::BuildOpImpl(const BackendOpRunInfoPtr &op_run_info, const GraphInfo &graph_info,
                                       const std::vector<tensor::TensorPtr> &input_tensors,
                                       const std::vector<int64_t> &tensors_mask) {
  // Check if the graph cache exists.
  auto it = run_op_graphs_.find(graph_info);
  if (it != run_op_graphs_.end() && !IsOneOfCacheBlackList(op_run_info->base_op_run_info.op_name)) {
    return it->second;
  }

  // Prepare the graph
  const auto &kernel_graph = ConstructSingleOpGraph(op_run_info, input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  RunOpOptimize(kernel_graph);
  SelectKernel(kernel_graph);
  RunOpHardwareOptimize(kernel_graph);
  StartKernelRT();
  RunOpHideNopNode(kernel_graph);
  BuildKernel(kernel_graph);
  auto enable_op_graph_cache = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
  if (enable_op_graph_cache) {
    run_op_graphs_[graph_info] = kernel_graph;
  }
  return kernel_graph;
}

void GPUSession::RunOpImplOrigin(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                                 std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                                 const std::vector<int64_t> &tensors_mask) {
  RunOpImpl(graph_info, op_run_info, input_tensors, outputs, tensors_mask);
}

void GPUSession::RunOpImpl(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                           std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                           const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(op_run_info);
  ProcessInputTensorsForHeterogeneous("GPU", *input_tensors);
  const auto &kernel_graph = BuildOpImpl(op_run_info, graph_info, *input_tensors, tensors_mask);
  EraseValueNodeTensor(tensors_mask, input_tensors);
  // wait for allreduce
  for (auto &tensor : *input_tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->NeedWaitDevice()) {
      tensor->WaitDevice();
    }
  }

  // run op
  MS_EXCEPTION_IF_NULL(kernel_graph);
  RunOpRemoveNopNode(kernel_graph);
  RunOpAllocateMemory(*input_tensors, kernel_graph.get(), op_run_info->is_gradient_out);
  RunOpGenKernelEvent(kernel_graph.get());
  // Execute the computation
  LoadInputData(kernel_graph, *input_tensors);
  Execute(kernel_graph);
  // Fetch outputs
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  UpdateOutputs(kernel_graph, outputs, *input_tensors, &tensor_to_node);
  RunOpClearMemory(kernel_graph.get());
  if (IsOneOfCacheBlackList(op_run_info->base_op_run_info.op_name)) {
    run_op_graphs_.erase(graph_info);
  }
}

#ifdef ENABLE_DEBUGGER

void GPUSession::Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  // Dump graph and graph history file if e2e_dump is enabled and update cur_dump_iter for GPU old runtime.
  if (debugger_->DebuggerBackendEnabled()) {
    MS_EXCEPTION_IF_NULL(kernel_graph);
    E2eDump::DumpRunIter(kernel_graph, rank_id_);
    E2eDump::DumpData(kernel_graph.get(), rank_id_, debugger_.get());
  } else {
    DumpJsonParser::GetInstance().UpdateDumpIter();
  }
}

bool GPUSession::DumpDataEnabledIteration() const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  return runtime_instance->DumpDataEnabledIteration();
}
#endif

void GPUSession::SyncStream() const {
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
