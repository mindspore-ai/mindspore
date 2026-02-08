/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "plugin/cpu/cpu_device_context.h"

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>

#include "include/utils/callback.h"
#include "plugin/cpu/graph_optimizer/pass/base/reg_cpu_const_input_to_attr.h"
#include "plugin/cpu/graph_optimizer/pass/train/insert_cast_to_pyexecute.h"
#include "plugin/cpu/graph_optimizer/pass/train/softmax_grad_fusion.h"
#include "plugin/cpu/graph_optimizer/pass/train/matmul_biasadd_relu_fusion.h"
#include "plugin/cpu/graph_optimizer/pass/inference/print_value_type.h"
#include "plugin/cpu/graph_optimizer/pass/inference/insert_cast_cpu.h"
#include "plugin/cpu/graph_optimizer/pass/inference/insert_format_transform_op.h"
#include "plugin/cpu/graph_optimizer/somas/cpu_somas.h"
#ifdef ENABLE_AKG
#include "kernel/cpu/akg/akg_cpu_kernel_build.h"
#endif
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/cpu/cpu_kernel.h"
#include "plugin/cpu/kernel_executor/kernel_select/kernel_select_cpu.h"
#include "plugin/cpu/kernel_executor/kernel_select/cpu_kernel_task.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "utils/trace_base.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/common/pass_manager/pass_manager.h"
#include "include/backend/common/pass_manager/common_backend_optimization.h"
#include "backend/common/pass/insert_type_transform_op.h"
#include "backend/common/pass/flatten_value_sequence_in_pyexecute.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "backend/common/pass/add_training_attr.h"
#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "backend/common/pass/dynamic_sequence_ops_adaptation.h"
#include "include/backend/common/expander/fallback/expander_fallback.h"
#include "backend/common/pass/value_graph_binder.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "plugin/cpu/profiler/cpu_profiling.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "plugin/cpu/res_manager/collective/ms_collective_comm_lib.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#endif
#include "tools/profiler/profiler.h"
#include "include/utils/parallel_context.h"
#include "primitive/framework_ops.h"
#include "ops_utils/op_constants.h"
#include "include/runtime/hardware_abstract/kernel_base/oplib/oplib.h"
#include "tools/profiler/profiling.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "kernel/cpu/contiguous_cpu_kernel.h"
#include "kernel/cpu/op_plugin/custom_op_plugin_kernel.h"
#include "ops_utils/op_utils.h"
#include "mindspore/ops/kernel/host/host_kernel_mod.h"
#include "mindspore/ops/kernel/host/host_kernel_build.h"

namespace mindspore {
namespace device {
namespace cpu {
namespace {
const char kModelNameCPU[] = "CPU";
const char kEventOptimizeGraph[] = "OptimizeGraph";
const char kStageSetKernelInfo[] = "SetKernelInfo";

std::pair<bool, size_t> MatchMultiDynamicKernelAttr(const kernel::KernelAttr &kernel_attr,
                                                    const std::vector<int64_t> &dyn_input_sizes,
                                                    const std::vector<kernel::KernelAttr> &kernel_attr_list) {
  auto output_num = kernel_attr.GetOutputSize();
  for (size_t index = 0; index < kernel_attr_list.size(); ++index) {
    // support multi dynamic inputs.
    const auto &cur_kernel_attr = kernel_attr_list[index];
    auto cur_input_num = cur_kernel_attr.GetInputSize();
    if (dyn_input_sizes.size() != cur_input_num) {
      MS_LOG(EXCEPTION) << "Kernel attr's input num: " << cur_input_num
                        << ", is not equal to dynamic input size: " << dyn_input_sizes.size();
    }
    bool mis_match = false;
    size_t input_index = 0;
    for (size_t i = 0; i < cur_input_num; ++i) {
      int64_t dyn_input_size = dyn_input_sizes[i];
      if (dyn_input_size < 0) {
        dyn_input_size = 1;
      }
      auto dtype = cur_kernel_attr.GetInputAttr(i).dtype;
      for (size_t j = 0; j < LongToSize(dyn_input_size); ++j) {
        if (kernel_attr.GetInputAttr(input_index).dtype != dtype) {
          mis_match = true;
          break;
        }
        ++input_index;
      }
      if (mis_match) {
        break;
      }
    }
    if (mis_match) {
      continue;
    }

    // only support one dynamic output. TODO: support multi dynamic output.
    for (size_t i = 0; i < output_num; ++i) {
      auto dtype = cur_kernel_attr.GetOutputAttr(i).dtype;
      if (kernel_attr.GetInputAttr(i).dtype != dtype) {
        mis_match = true;
        break;
      }
    }
    if (!mis_match) {
      return std::make_pair(true, index);
    }
  }
  return std::make_pair(false, 0);
}

runtime::KernelTaskPtr GetTaskByTaskType(const runtime::KernelTaskType &task_type,
                                         const std::shared_ptr<runtime::KernelTaskContext> &task_context) {
  switch (task_type) {
    case runtime::KernelTaskType::kCONTIGUOUS_TASK:
      return std::make_shared<CpuContiguousKernelTask>(task_context);
    case runtime::KernelTaskType::kCOPY_TASK:
      return std::make_shared<CpuCopyWithSliceKernelTask>(task_context);
    default:
      MS_LOG(EXCEPTION) << "KernelTaskType is invalid, task_type:" << task_type;
  }
}

void MallocMemoryForDeviceAddress(device::DeviceAddress *device_address, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_address);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Graph", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "Graph", device::tracker::MemType::kPyNativeOutput,
                                                 device_address->GetSize(), device_address);
  if (device_address->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_address)) {
      MS_LOG(EXCEPTION) << "Allocate device memory failed!";
    }
  }
}

}  // namespace

void SyncOutInRef(const kernel::KernelAttr &from_kernel_attr, kernel::KernelAttr *to_kernel_attr) {
  const auto &out_in_ref = from_kernel_attr.GetOutInRefMap();
  bool all_out_in_ref = from_kernel_attr.GetAllOutInRef();
  for (const auto &ref : out_in_ref) {
    (void)to_kernel_attr->AddOutInRef(ref.first, ref.second);
  }
  (void)to_kernel_attr->AddAllOutInRef(all_out_in_ref);
}

void SetCpuRefMapToKernelInfo(const CNodePtr &apply_kernel, const std::vector<kernel::KernelAttr> &apply_kernel_attrs) {
  MS_EXCEPTION_IF_NULL(apply_kernel);
  auto kernel_attrs = apply_kernel_attrs;
  if (kernel_attrs.empty()) {
    return;
  }

  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(apply_kernel);
  MS_EXCEPTION_IF_NULL(build_info);
  auto kernel_attr = GetKernelAttrFromBuildInfo(build_info);
  std::vector<int64_t> dyn_input_sizes = {};
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, apply_kernel)) {
    dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(apply_kernel, kAttrDynInputSizes);
  }
  std::pair<bool, int64_t> match_result;

  if (kernel_attrs[0].GetSkipCheck()) {
    // If kernel skips attr check, we need to synchronize the ref map in case it's discarded.
    SyncOutInRef(kernel_attrs[0], &kernel_attr);
    kernel_attrs[0] = kernel_attr;
    match_result = {true, 0};
  } else if (dyn_input_sizes.empty() || kernel_attrs[0].GetAllSame()) {
    match_result = MatchKernelAttr(kernel_attr, kernel_attrs);
  } else {
    match_result = MatchMultiDynamicKernelAttr(kernel_attr, dyn_input_sizes, kernel_attrs);
  }

  auto [is_match, index] = match_result;
  if (!is_match) {
    MS_LOG_WITH_NODE(EXCEPTION, apply_kernel)
      << apply_kernel->fullname_with_scope() << " does not support this kernel data type: " << build_info->ToString()
      << ", node debug name: " << apply_kernel->DebugString(AnfNode::DebugStringLevel::kLevel2);
  }

  auto kernel_info = dynamic_cast<device::KernelInfo *>(apply_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &matched_kernel_attr = kernel_attrs[index];
  if (!matched_kernel_attr.GetOutInRefMap().empty() || matched_kernel_attr.GetAllOutInRef()) {
    kernel_info->set_ref_map(matched_kernel_attr.GetAllOutInRef(), matched_kernel_attr.GetOutInRefMap());
  }
}

using mindspore::kernel::KernelBuildInfo;

void CPUDeviceContext::Initialize() {
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(init_lock_);
#else
  std::lock_guard<std::mutex> lock(init_mutex_);
#endif
  if (initialized_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Initialize();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice) {
    // Dump json config file if dump is enabled.
    uint32_t rank_id = 0;
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
  }
#ifdef __linux__
  if (ms_context->IsDefaultDeviceTarget() && ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice) {
    MS_LOG(INFO)
      << "No device_target set, set CPU as default. You can call mindspore.set_context(device_target=\"XXX\")";
  }
#endif  // __linux__
  initialized_ = true;
}

void CPUDeviceContext::Destroy() {
  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Destroy();
  initialized_ = false;
}

void CPUKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_lazy_inline = ms_context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (enable_lazy_inline) {
    MS_LOG(EXCEPTION) << "CPU does not support the lazy_inline feature, "
                      << "please do not mark @lazy_inline in cell's __init__ func.";
  }
  if (kernel_graph->is_from_single_op()) {
    SetOperatorInfo(kernel_graph);
    SingleOpGraphOptimize(kernel_graph);
    UpdateKernelRefInfo(kernel_graph);
  } else {
    // The passes in this function must be before ops select: SetOperatorInfo()
    OptimizeMindIR(kernel_graph);
    // Update Graph Dynamic Shape Attr.
    opt::AddDynamicShapeAttrPass(kernel_graph);

    SetOperatorInfo(kernel_graph);
    // SetOperatorInfo may generate new node, so need set kernel object type again.
    kernel_graph->SetKernelObjectTypesForUnrealNodes();
#ifdef ENABLE_DUMP_IR
    if (ms_context->CanDump(kIntroductory)) {
      DumpIR("hwopt_comm_after_kernel_select_" + graph->ToString() + ".ir", graph, true);
    }
#endif

    OptimizeGraphImpl(kernel_graph);

    // Run final optimization.
    opt::CommonFinalOptimization(kernel_graph);

    // Run graph kernel fusion optimization
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
  }
}

void CPUKernelExecutor::UpdateKernelRefInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<CNodePtr> &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel);
    if (IsPrimitiveCNode(kernel, prim::kPrimCustom) &&
        mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyCPU) == nullptr) {
      MS_LOG(DEBUG) << "Not find operator information for Custom operator [" << op_name << "]";
      return;
    }

    auto kernel_attr_list = kernel::NativeCpuKernelMod::GetCpuSupportedList(op_name);
    if (kernel_attr_list.empty()) {
      MS_LOG(DEBUG) << "kernel_attr_list is empty";
      return;
    }

    auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    kernel_info->set_ref_map(kernel_attr_list[0].GetAllOutInRef(), kernel_attr_list[0].GetOutInRefMap());
  }
}

void CPUKernelExecutor::OptimizeMindIR(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::SoftmaxGradFusionCpu>("softmax_grad_fusion_cpu"));
  // Match MatMul+BiasAdd+ReLU first, if no match, then match MatMul+BiasAdd
  pm->AddPass(std::make_shared<opt::MatMulBiasAddReluFusionCPU>("matmul_biasadd_relu_fusion_cpu"));
  pm->AddPass(std::make_shared<opt::DynamicSequenceOpsAdaptation>());

  // Do communication op fusion before InsertTensorMoveForCommunication pass.
  // So these passes are before kernel select process, no need to generate kernel build info in them.
  if (parallel::ParallelContext::GetInstance()->enable_all_reduce_fusion()) {
    MS_LOG(INFO) << "Parallel comm_fusion of AllReduce is enabled.";
    pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  }

  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}

void CPUKernelExecutor::OptimizeGraphImpl(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>("insert_type_transform_op"));
  pm->AddPass(std::make_shared<opt::FlattenValueSequenceInPyExecute>("flatten_value_sequence_in_pyexecute"));
  pm->AddPass(std::make_shared<opt::InsertFormatTransformOpCPU>("insert_format_transform_op_cpu"));
  pm->AddPass(std::make_shared<opt::InsertCastCPU>("insert_cast"));
  pm->AddPass(std::make_shared<opt::EraseVisitAttr>());
  pm->AddPass(std::make_shared<opt::InsertTensorMoveForCommunication>());
  pm->AddPass(std::make_shared<opt::AddTrainingAttr>());
  pm->AddPass(std::make_shared<opt::PrintValueType>("print_value_type"));
  pm->AddPass(std::make_shared<opt::InsertCastToPyExecute>("insert_cast_for_pyexecute"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}

void CPUKernelExecutor::SingleOpGraphOptimize(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertCastCPU>("insert_cast"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
  graph->SetExecOrderByDefault();
}

namespace {
void SetControlOpInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<std::string> inputs_format;
  std::vector<TypeId> inputs_type;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    (void)inputs_format.emplace_back(kOpFormat_DEFAULT);
    inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index));
  }
  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_type;
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    (void)outputs_format.emplace_back(kOpFormat_DEFAULT);
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(kernel_node, output_index));
  }

  auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
  builder->SetInputsFormat(inputs_format);
  builder->SetInputsDeviceType(inputs_type);
  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_type);

  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());
}

// Before creating the kernel, check whether the node has completed the operator selection. If not, the operator
// selection needs to be performed to set kernel info.
void SetKernelInfoBeforeCreateKernel(const std::vector<CNodePtr> &nodes) {
  // Check whether the node has completed operator selection.
  for (const auto &node : nodes) {
    if (AnfAlgo::GetSelectKernelBuildInfo(node) != nullptr) {
      continue;
    }

    // Kernel selection process for non control op.
    if (!common::AnfAlgo::IsBpropCutOpExecInBackend(node)) {
      auto [msg, etype] = SetKernelInfoWithMsg(node);
      if (!msg.empty()) {
        MS_EXCEPTION(etype) << "#umsg#Kernel select failed:#umsg#" << msg;
      }
    } else {
      // Kernel selection process for control op.
      SetControlOpInfo(node);
    }
  }
}
}  // namespace

void CPUKernelExecutor::SetOperatorInfo(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  bool do_expand = false;
  auto mng = graph->manager();
  if (mng == nullptr) {
    mng = Manage(graph, true);
    MS_EXCEPTION_IF_NULL(mng);
    graph->set_manager(mng);
  }
  auto &node_list = graph->execution_order();
  for (auto &node : node_list) {
    if (!common::AnfAlgo::IsBpropCutOpExecInBackend(node)) {
      auto [msg, etype] = SetKernelInfoWithMsg(node);
      if (msg.empty()) {
        continue;
      }
      auto f = [](const CNodePtr &n) {
        auto res = SetKernelInfoWithMsg(n);
        return res.first.empty();
      };
      auto expand_ret = expander::TryExpandCNode(node, f);
      if (!expand_ret) {
        MS_EXCEPTION(etype) << "#umsg#Kernel select failed:#umsg#" << msg
                            << "\nnode: " << node->DebugString(AnfNode::DebugStringLevel::kLevel2);
      }
      MS_LOG(INFO) << msg << " but expand success.";
      do_expand = true;
    } else {
      SetControlOpInfo(node);
    }
  }
  if (do_expand) {
    (void)opt::BindValueToGraph().Run(graph);
    graph->SetExecOrderByDefault();
  }
  (void)profiler::CollectHostInfo(kModelNameCPU, kEventOptimizeGraph, kStageSetKernelInfo, start_time,
                                  profiler::GetClockSyscnt(), 1);
}

kernel::KernelModPtr CPUKernelExecutor::CreateKernelMod(const std::string &op_name) const {
  if (ops::IsOpPluginKernel(op_name)) {
    return kernel::Factory<kernel::op_plugin::CustomOpPluginCpuKernelMod>::Instance().Create(op_name);
  }
  return kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Create(op_name);
}

void CPUKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  SetKernelInfoBeforeCreateKernel(nodes);

  kernel::KernelMeta *bin_map = kernel::KernelMeta::GetInstance();
  std::vector<AnfNodePtr> akg_nodes;
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (common::AnfAlgo::IsBpropCutOpExecInBackend(node)) {
      continue;
    }
    if (session::AnfRuntimeAlgorithm::GetKernelType(node) == KernelType::AKG_KERNEL) {
      if (!bin_map->initialized()) {
        bin_map->Initialize();
      }
      akg_nodes.push_back(node);
      continue;
    }
    std::string kernel_name = common::AnfAlgo::GetCNodeName(node);

    const auto kernel_type = AnfAlgo::GetKernelType(node);
    if (kernel_type == KernelType::HOST_KERNEL) {
      kernel::KernelModPtr kernel_mod_ptr = kernel::HostOpBuild(node);
      if (kernel::CheckResizeCondition(node)) {
        std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(node);
        std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(node);
        if (kernel_mod_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == kernel::KRET_RESIZE_FAILED) {
          MS_LOG(EXCEPTION) << "#dmsg#Kernel resize failed:#dmsg#internal kernel op[" << node->fullname_with_scope()
                            << "] Resize failed.";
        }
      }
      AnfAlgo::SetKernelMod(kernel_mod_ptr, node.get());
      continue;
    }

    std::shared_ptr<kernel::NativeCpuKernelMod> cpu_kernel =
      std::dynamic_pointer_cast<kernel::NativeCpuKernelMod>(CreateKernelMod(kernel_name));

    if (cpu_kernel == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#Build cpu operator[" << node->fullname_with_scope()
                                 << "] failed";
    }

    auto kernel_attrs = cpu_kernel->GetOpSupport();
    SetCpuRefMapToKernelInfo(node, kernel_attrs);
    auto thread_pool = kernel::GetActorMgrInnerThreadPool();
    cpu_kernel->SetThreadPool(thread_pool);
    std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(node);
    std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(node);
    auto ret = cpu_kernel->Init(common::AnfAlgo::GetCNodePrimitive(node), input_kernel_tensors, output_kernel_tensors);
    if (!ret) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << trace::DumpSourceLines(node);
    }
    if (kernel::CheckResizeCondition(node)) {
      if (cpu_kernel->Resize(input_kernel_tensors, output_kernel_tensors) == kernel::KRET_RESIZE_FAILED) {
        MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Kernel build failed:#dmsg#CPU kernel op [" << node->fullname_with_scope()
                                   << "] resize failed.";
      }
    }

    AnfAlgo::SetKernelMod(cpu_kernel, node.get());
  }
#ifdef ENABLE_AKG
  kernel::AkgCpuKernelBuilder akg_cpu_kernel_builder;
  (void)akg_cpu_kernel_builder.SingleOpParallelBuild(akg_nodes);
#endif
}

std::vector<size_t> CPUKernelExecutor::GetLaunchIgnoredInputAddressIdx(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto kernel_mod = kernel_info->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->GetLaunchIgnoredInputAddressIdx();
}

bool CPUKernelExecutor::IsLaunchIgnoredInputAddressIdx(const AnfNodePtr &node, size_t input_idx) const {
  auto ignored_input_list = GetLaunchIgnoredInputAddressIdx(node);
  if (std::find(ignored_input_list.begin(), ignored_input_list.end(), input_idx) != ignored_input_list.end()) {
    return true;
  }
  return false;
}

void CPUKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (!kernel_graph->is_from_single_op()) {
    // Remove reorder after PS feature finish adapting push/pull in auto_monad.
    auto execution_order = kernel_graph->execution_order();
    common::AnfAlgo::ReorderPosteriorExecList(NOT_NULL(&execution_order));
    kernel_graph->set_execution_order(execution_order);
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // somas
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    auto somas = std::make_shared<CPUSomas>();
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

bool CPUKernelExecutor::LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                     void * /* stream */) const {
  MS_EXCEPTION_IF_NULL(kernel);

  const auto &profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  if (profiler_inst->GetEnableFlag() && profiler_inst->GetOpTimeFlag()) {
    auto ret = LaunchKernelWithProfiling(kernel, inputs, workspace, outputs, kernel_mod);
    return ret;
  }
  auto ret = DoLaunchKernel(kernel, inputs, workspace, outputs, kernel_mod);
  return ret;
}

bool CPUKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                          const tensor::TensorPtrList &input_tensors,
                                          const tensor::TensorPtrList &output_tensors, const size_t &stream_id) const {
  auto task_context =
    std::make_shared<runtime::KernelTaskContext>(device_context_, input_tensors, output_tensors, nullptr);
  auto task = GetTaskByTaskType(task_type, task_context);
  MS_EXCEPTION_IF_NULL(task);

  auto ret = task->RunWithRet();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Exec task failed, task_type:" << task_type;
  }
  return ret;
}

bool CPUKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                          const std::vector<KernelTensor *> &input_kernel_tensors,
                                          const std::vector<KernelTensor *> &output_kernel_tensors,
                                          const size_t &stream_id) const {
  if (task_type != runtime::KernelTaskType::kCONTIGUOUS_TASK) {
    MS_LOG(EXCEPTION) << "KernelTaskType not supported, task_type:" << task_type;
  }
  MS_LOG(DEBUG) << "Start Contiguous task";

  const auto &input = input_kernel_tensors[0];
  MS_EXCEPTION_IF_NULL(input);
  const auto &input_address = input->device_address();
  const auto &input_storage_info = input->tensor_storage_info();

  const auto &output = output_kernel_tensors[0];
  MS_EXCEPTION_IF_NULL(output);
  const auto &output_address = output->device_address();

  MS_LOG(DEBUG) << "Input:" << input->ToString() << " output:" << output->ToString();

  MallocMemoryForDeviceAddress(input_address.get(), device_context_);
  MallocMemoryForDeviceAddress(output_address.get(), device_context_);

  kernel::ContiguousCpuKernel contiguous_kernel;
  auto ret = contiguous_kernel.LaunchContiguous(input->dtype_id(), input_address, input_storage_info,
                                                output->dtype_id(), output_address);
  if (!ret) {
    MS_LOG(EXCEPTION) << "CpuContiguous failed";
  }

  MS_LOG(DEBUG) << "End Contiguous task";
  return true;
}

bool CPUKernelExecutor::LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs,
                                                  KernelMod *kernel_mod) const {
  MS_EXCEPTION_IF_NULL(kernel);

  auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  uint32_t pid = IntToUint(getpid());
  // cpu support multi-thread with mindrt for profiling.
  profiler_inst->OpDataProducerBeginParallel(kernel->fullname_with_scope(), pid);
  bool ret = DoLaunchKernel(kernel, inputs, workspace, outputs, kernel_mod);
  profiler_inst->OpDataProducerEndParallel(kernel->fullname_with_scope());
  profiler_inst->RecordFrameWorkInfo(kernel);
  return ret;
}

bool CPUKernelExecutor::DoLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  auto ret = kernel_mod->Launch(inputs, workspace, outputs, nullptr);
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
               kernel->fullname_with_scope(), false);
  return ret;
}

void CPUKernelExecutor::RebuildKernelSelectBackoffOp(const std::vector<CNodePtr> &nodes) const {
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!AnfAlgo::IsKernelSelectBackoffOp(node)) {
      continue;
    }
    auto [failure_info, failure_type] = AnfAlgo::GetKernelSelectBackoffInfo(node);
    if (IsVmapNotSupported(node)) {
      MS_EXCEPTION(failure_type) << "#umsg#Kernel select failed:#umsg#" << failure_info;
    }

    // Judge whether match strictly between kernel build info and supported kernel attrs.
    const auto &kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
    MS_EXCEPTION_IF_NULL(kernel_build_info);
    const auto &kernel_attr = kernel::GetKernelAttrFromBuildInfo(kernel_build_info);
    const auto &supported_kernel_attrs =
      kernel::NativeCpuKernelMod::GetCpuSupportedList(common::AnfAlgo::GetCNodeName(node));
    const auto &match_result = kernel::MatchKernelAttrStrict(kernel_attr, supported_kernel_attrs);
    auto attr_info = kernel::FetchPrintInfoByKernelAttr(kernel_attr);
    if (!match_result.first) {
      MS_LOG(INFO) << "Backoff and rebuild kernel on CPU failed for node: " << node->fullname_with_scope()
                   << ", node attr: " << attr_info;
      MS_EXCEPTION(failure_type) << "#umsg#Kernel select failed:#umsg#" << failure_info;
    } else {
      // Set the CPU flag.
      common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(kCPUDevice), node);
      kernel_build_info->set_kernel_type(CPU_KERNEL);
      kernel_build_info->set_processor(kernel::Processor::CPU);
      MS_LOG(INFO) << "Backoff and rebuild kernel on CPU successfully for node: " << node->fullname_with_scope()
                   << ", node attr: " << attr_info;
    }

    CreateKernel({node});
  }
}

MS_REGISTER_DEVICE(kCPUDevice, CPUDeviceContext);
#ifdef WITH_BACKEND
MSCONTEXT_REGISTER_INIT_FUNC(kCPUDevice, [](MsContext *ctx) -> void {
  MS_EXCEPTION_IF_NULL(ctx);
  if (ctx->backend_policy() != "ms") {
    (void)ctx->set_backend_policy("ms");
  }
});
#endif

// Register functions to _c_expression so python hal module could call CPU device interfaces.
void PybindCPUStatelessFunc(py::module *m) { MS_EXCEPTION_IF_NULL(m); }
REGISTER_DEV_STATELESS_FUNC_CB(kCPUDevice, PybindCPUStatelessFunc);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
