/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/ge_backend.h"

#include <algorithm>
#include <set>
#include <utility>
#include <queue>
#include <regex>
#include <map>
#include <string>
#include <memory>
#include <vector>

#include "backend/ge_backend/pass/ge_backend_optimization.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "ir/manager.h"
#include "ir/map_tensor.h"
#include "ir/tensor_new.h"
#include "ir/graph_utils.h"
#include "backend/ge_backend/utils/device_address_utils.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "include/utils/callback.h"
#include "utils/config_manager.h"
#include "utils/convert_utils.h"
#include "device_address/device_address.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/profiler.h"
#include "tools/profiler/mstx/mstx_impl.h"
#include "utils/log_adapter.h"
#include "utils/file_utils.h"
#include "utils/info.h"
#include "utils/trace_info.h"
#ifndef ENABLE_SECURITY
#include "backend/ge_backend/dump/hook_debugger.h"
#include "backend/ge_backend/dump/deprecated_env.h"
#endif
#include "include/utils/callbacks.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "backend/ge_backend/graph_ir/utils.h"
#include "abstract/abstract_function.h"
#include "primitive/array_ops.h"
#include "primitive/sparse_tensor_ops.h"
#include "include/runtime/pipeline/pipeline.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/utils/scoped_long_running.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "backend/ge_backend/runtime/segment_runner.h"
#include "primitive/nn_ops.h"
#include "primitive/framework_ops.h"
#include "backend/ge_backend/runtime/graph_scheduler.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "backend/ge_backend/runtime/control_node_parser.h"
#include "include/utils/parallel_context.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "include/utils/signal_util.h"
#endif
#include "pybind_api/gil_scoped_long_running.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "plugin/ascend/res_manager/device_context_conf/op_tuning_conf.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"
#include "plugin/ascend/res_manager/device_context_conf/op_debug_conf.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorreport_utils.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorprint_utils.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorsummary_utils.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
namespace {
constexpr size_t kNormalTensorNum = 1;
constexpr size_t kMapTensorNum = 3;
constexpr size_t kMapTensorKeyIndex = 0;
constexpr size_t kMapTensorValueIndex = 1;
constexpr size_t kMapTensorStatusIndex = 2;
constexpr size_t kGraphInfoSavePrefixLen = 5;
using KernelWithIndex = session::KernelWithIndex;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using PrimTypePair = std::pair<PrimitivePtr, AbstractFunctionPtr>;
using MapPrimTypeFuncGraph = std::map<PrimTypePair, FuncGraphPtr>;
using TypedPrimitiveAbstractClosurePtr = std::shared_ptr<abstract::TypedPrimitiveAbstractClosure>;
using MbufDataItem = std::variant<std::string, mindspore::tensor::TensorPtr>;
const char kModelNameRuntime[] = "Runtime";
const char kEventCompileGraph[] = "CompileGraph";
const char kStageCompileGraphs[] = "CompileGraphs";
constexpr auto kTensorDumpOpName = "TensorDump";
constexpr auto kTensorDumpChannelName = "ms_tensor_dump";
std::mutex g_tsd_mutex;

void CheckContiguousTensor(const tensor::TensorPtr &tensor) {
  if (!DeviceAddressUtils::IsContiguousTensor(tensor)) {
    MS_LOG(EXCEPTION) << "The ge backend don't support view inputs, please check.";
  }
}

void CheckNodeValid(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // Check the joined any abstract.
  const auto &node_abs = node->abstract();
  if (node_abs != nullptr && node_abs->isa<abstract::AbstractJoinedAny>()) {
    auto abs_joined_any = node_abs->cast<abstract::AbstractJoinedAnyPtr>();
    if (abs_joined_any != nullptr) {
      TraceGuard guard(MakeTraceInfo<TraceTypeJoin>(node->debug_info()));
      abs_joined_any->ThrowException();
    }
  }
}

bool IsEmptySequence(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                     const size_t *const output_position) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(output_position);
  // When the output node is a valuenode, the position may out of range.
  if (*output_position >= output_tensors.size()) {
    return false;
  }

  if (output_node->abstract() == nullptr || (!output_node->abstract()->isa<abstract::AbstractSequence>())) {
    return false;
  }
  const auto &tuple_abs = output_node->abstract()->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abs);
  if ((!tuple_abs->dynamic_len()) && tuple_abs->dynamic_len_element_abs() == nullptr) {
    return false;
  }
  const auto &tensor = output_tensors[*output_position];
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->base_shape_ptr() == nullptr || (!tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    return false;
  }
  const auto &sequence_shape = tensor->base_shape_ptr()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  return sequence_shape->size() == 0;
}

bool IsTupleOutputOfAnyType(const abstract::AbstractBasePtr &abstract, const tensor::TensorPtr &tensor) {
  if (abstract == nullptr || !abstract->isa<abstract::AbstractAny>() || tensor == nullptr) {
    return false;
  }
  auto device_tensor = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  return device_tensor != nullptr && tensor->base_shape_ptr() != nullptr &&
         tensor->base_shape_ptr()->isa<abstract::SequenceShape>();
}

mindspore::ge_backend::runtime::KernelMapPosition FetchOriginOutputOrder(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  mindspore::ge_backend::runtime::KernelMapPosition outputs_order;
  const auto &root_output = common::AnfAlgo::VisitKernelWithReturnType(node, 0, false, {prim::kPrimTupleGetItem}).first;
  size_t position = 0;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(root_output);
  for (const auto &output : outputs) {
    if (outputs_order.count(output) == 0) {
      outputs_order[output] = {position++};
    } else {
      (void)outputs_order[output].emplace_back(position++);
    }
  }
  return outputs_order;
}

bool IsLazyinlineAndPipeline(const FuncGraphPtr &func_graph) {
  // cell reuse + pipeline parallel
  // only O2
  if (func_graph == nullptr) {
    return false;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  bool has_cell_reuse = std::any_of(nodes.begin(), nodes.end(), [](const AnfNodePtr &node) {
    if (node == nullptr || !node->isa<CNode>()) {
      return false;
    }
    auto cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();

    // for func graph
    AnfNodePtr fn = inputs[0];
    FuncGraphPtr child_graph = common::AnfAlgo::GetValueNodeFuncGraph(fn);
    bool func_graph_has_cell_reuse = child_graph != nullptr && child_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE);
    if (func_graph_has_cell_reuse) {
      return func_graph_has_cell_reuse;
    }

    // for kernel graph
    bool kernel_graph_has_cell_reuse = false;
    if (IsPrimitiveCNode(cnode, prim::kPrimCall)) {
      auto call_graph = cnode->input(kIndex1);
      auto sub_kernel_graph = session::AnfRuntimeAlgorithm::GetValueNodeKernelGraph(call_graph);
      kernel_graph_has_cell_reuse = sub_kernel_graph != nullptr && sub_kernel_graph->need_inline();
    }
    return kernel_graph_has_cell_reuse;
  });

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stages = parallel_context->pipeline_stage_split_num();
  auto grad_accu_step = parallel_context->grad_accumulation_step();
  MS_LOG(INFO) << "graph: " << func_graph->ToString() << "stages: " << stages << ", grad_accu_step: " << grad_accu_step;
  if (stages <= 1 && grad_accu_step <= 1) {
    return false;
  }
  if (has_cell_reuse) {
    context->SetCellReuseLevel(CellReuseLevel::kNoInline);
  }
  return has_cell_reuse;
}

bool HasIncorporateCall(const std::vector<AnfNodePtr> &all_nodes) {
  for (const auto &node : all_nodes) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode, prim::kPrimPartial)) {
      auto partial_function = cnode->input(kPartialGraphIndex);
      if (!IsValueNode<FuncGraph>(partial_function)) {
        MS_LOG(INFO) << "Partial has indirect call: " << cnode->DebugString();
        return true;
      }
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
      const auto &switch_inputs = cnode->inputs();
      if (std::any_of(switch_inputs.begin() + kSwitchTrueBranchIndex, switch_inputs.end(), [](const AnfNodePtr &input) {
            return !IsPrimitiveCNode(input, prim::kPrimPartial) && !IsValueNode<FuncGraph>(input);
          })) {
        MS_LOG(INFO) << "Switch has indirect call: " << cnode->DebugString();
        return true;
      }
      continue;
    }
    if (IsPrimitiveCNode(cnode, prim::kPrimSwitchLayer)) {
      auto make_tuple = cnode->input(kSwitchLayerBranchesIndex);
      if (!IsPrimitiveCNode(make_tuple, prim::kPrimMakeTuple)) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
          << "SwitchLayer input2 should be make_tuple, but got: " << make_tuple->DebugString();
      }
      const auto &make_tuple_inputs = make_tuple->cast<CNodePtr>()->inputs();
      if (std::any_of(make_tuple_inputs.begin() + 1, make_tuple_inputs.end(), [](const AnfNodePtr &input) {
            return !IsPrimitiveCNode(input, prim::kPrimPartial) && !IsValueNode<FuncGraph>(input);
          })) {
        MS_LOG(INFO) << "SwitchLayer has indirect call: " << cnode->DebugString();
        return true;
      }
      continue;
    }
    if (common::AnfAlgo::HasIncorporateCallNode(cnode)) {
      return true;
    }
  }
  return false;
}

bool ExistSwitchRef(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &all_nodes) {
  // %1 = switch(cond, func1, func2)
  // %2 = %1()  if the abstract of the node is AbstractRefTensor or Tuple/List(AbstractRefTensor, ...), return true.
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSwitch)) {
      continue;
    }
    auto iter = node_users.find(node);
    if (iter != node_users.end()) {
      auto &users = iter->second;
      for (auto &user : users) {
        auto &user_node = user.first;
        if (common::AnfAlgo::HasAbstractRef(user_node) || common::AnfAlgo::SequenceHasAbstractRef(user_node)) {
          if (device_target == kAscendDevice) {
            MS_LOG(WARNING) << "On the Ascend platform, if you read-only access to the parameter, "
                            << "you can take the value of the parameter, so that the system can do more optimization. "
                            << "For example, change 'return param' to 'return param.value()'\n"
                            << "Please check your code:" << trace::GetDebugInfoStr(user_node->debug_info());
          }
          return true;
        }
      }
    }
  }
  return false;
}

bool IsControlFlowGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  auto graphs = func_graph->func_graphs_used_total();
  (void)graphs.insert(func_graph);
  bool exist_control_flow = !func_graph->func_graphs_used_total().empty();
  bool exist_func = exist_control_flow && HasIncorporateCall(all_nodes);
  if (exist_func) {
    return true;
  }
  bool exist_while =
    std::any_of(graphs.cbegin(), graphs.cend(), [](const FuncGraphPtr &fg) { return fg->recursive(); });
  MS_LOG(INFO) << func_graph->ToString() << " exist_while: " << exist_while;
  if (exist_while || ExistSwitchRef(func_graph, all_nodes)) {
    return true;
  }
  return false;
}

int64_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The node tuple_get_item must have 2 inputs!";
  }
  auto output_index_value_node = tuple_get_item->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(output_index_value_node);
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto idx = value->isa<Int64Imm>() ? GetValue<int64_t>(value) : GetValue<int>(value);
  return idx;
}

KernelWithIndex VisitRealNodeWithNestLevel(const AnfNodePtr &anf_node, size_t index, size_t *nest_level) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<CNode>()) {
    return {anf_node, index};
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (common::AnfAlgo::GetCNodeName(cnode) == mindspore::kTupleGetItemOpName) {
    (*nest_level)++;
    auto real_node_with_index = VisitRealNodeWithNestLevel(common::AnfAlgo::GetTupleGetItemRealInput(cnode),
                                                           common::AnfAlgo::GetTupleGetItemOutIndex(cnode), nest_level);
    auto real_node = real_node_with_index.first;
    auto real_index = real_node_with_index.second;
    MS_EXCEPTION_IF_NULL(real_node);
    if (real_node->isa<CNode>() && common::AnfAlgo::GetCNodeName(real_node) == mindspore::kMakeTupleOpName) {
      (*nest_level)--;
      auto make_tuple = real_node->cast<CNodePtr>();
      return VisitRealNodeWithNestLevel(make_tuple->input(real_index + 1), index, nest_level);
    }
    return real_node_with_index;
  }
  return common::AnfAlgo::VisitKernelWithReturnType(anf_node, index, false,
                                                    {prim::kPrimMakeTuple, prim::kPrimTupleGetItem});
}

bool NeedConvertToRealTupleGetItem(const CNodePtr &cnode) {
  if (cnode->size() != kTupleGetItemInputSize) {
    return false;
  }
  if (!cnode->input(kInputNodeOutputIndexInTupleGetItem)->isa<ValueNode>() || GetTupleGetItemOutIndex(cnode) < 0) {
    return true;
  }
  size_t nest_level = 0;
  const size_t nest_limit = 1;
  auto real_node = VisitRealNodeWithNestLevel(cnode, 0, &nest_level);
  if (!common::AnfAlgo::IsCallNode(real_node.first) && AnfUtils::IsRealCNodeKernel(real_node.first) &&
      nest_level > nest_limit) {
    return true;
  }
  return false;
}

void UnifyIR(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  static const std::map<std::string, std::string> kOpListToTupleNames = {
    {mindspore::kMakeListNewOpName, mindspore::kMakeTupleOpName},
    {mindspore::kListGetItemOpName, mindspore::kTupleGetItemOpName},
    {mindspore::kListSetItemOpName, mindspore::kTupleSetItemOpName}};
  // List name --> tuple name.
  auto &&op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto iter = kOpListToTupleNames.find(op_name);
  if (iter != kOpListToTupleNames.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(iter->second)));
    // Reset full scope name.
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op from " << iter->first << " to " << iter->second << " for op "
                 << cnode->fullname_with_scope() << ", debug name:" << cnode->DebugString();
    op_name = iter->second;
  }

  // TupleGetItem --> RealTupleGetItem.
  if (op_name == mindspore::kTupleGetItemOpName && NeedConvertToRealTupleGetItem(cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(mindspore::kRealTupleGetItemOpName)));
    // Reset full scope name.
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op from TupleGetItem to RealTupleGetItem for op " << cnode->fullname_with_scope()
                 << ", debug name:" << cnode->DebugString();
  }

  // MakeTuple --> RealMakeTuple
  if (op_name == mindspore::kMakeTupleOpName && common::AnfAlgo::IsDynamicSequence(cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(mindspore::kRealMakeTupleOpName)));
    // Reset full scope name.
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op from MakeTuple to RealMakeTuple for op " << cnode->fullname_with_scope()
                 << ", debug name:" << cnode->DebugString();
  }
}

void CheckOutputIdx(size_t output_position, size_t output_tensors_size) {
  if (output_position >= output_tensors_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The output position is out of range: "
                               << output_position;
  }
}

void SyncTensorData(const tensor::TensorPtr &host_tensor, kernel::KernelTensor *const kernel_tensor,
                    const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const auto &device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  MS_EXCEPTION_IF_NULL(node);
  // memory has been allocate early in AllocGEInputOutputMemory
  MS_EXCEPTION_IF_NULL(device_tensor->GetPtr());
  // sync host tensor to device
  auto get_tensor_by_index = [&host_tensor](size_t index) {
    if (!host_tensor->isa<tensor::MapTensor>()) {
      return host_tensor;
    }
    const auto &map_tensor = host_tensor->cast<tensor::MapTensorPtr>();
    MS_EXCEPTION_IF_NULL(map_tensor);
    switch (index) {
      case kMapTensorKeyIndex:
        return map_tensor->key_tensor();
      case kMapTensorValueIndex:
        return map_tensor->value_tensor();
      case kMapTensorStatusIndex:
        return map_tensor->status_tensor();
      default:
        MS_LOG(EXCEPTION) << "Invalid index:" << index << " for map tensor:" << host_tensor->ToString();
    }
  };

  auto get_tensor_num = (host_tensor->isa<tensor::MapTensor>() ? kMapTensorNum : kNormalTensorNum);
  for (size_t i = 0; i < get_tensor_num; ++i) {
    const auto &real_host_tensor = get_tensor_by_index(i);
    MS_EXCEPTION_IF_NULL(real_host_tensor);
    // Copy data from host tensor to device.
    auto host_tensor_size = LongToSize(real_host_tensor->DataNBytes());
    auto host_tensor_type = real_host_tensor->data_type();
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams() ||
        !SyncCopy(kernel_tensor, real_host_tensor.get(), kDefaultStreamIndex)) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " + node->fullname_with_scope() +
                             ", host tensor size: " + std::to_string(host_tensor_size) +
                             ", host tensor type: " + std::to_string(static_cast<int>(host_tensor_type)) +
                             ", device tensor size: " + std::to_string(device_tensor->GetSize());
    }
  }
}
}  // namespace
mindspore::HashSet<const tensor::Tensor *> GEBackend::weights_need_reprepare_ = {};
BackendGraphId GEBackend::backend_graph_id_ = 0;

GEBackend::GEBackend() {
  Init();
  graph_compiler_ = std::make_shared<mindspore::ge_backend::runtime::GraphCompiler>(graph_executor_);
  mindspore::ge_backend::runtime::GraphScheduler::GetInstance().Initialize();
#ifndef ENABLE_SECURITY
  dump::CheckDeprecatedDumpEnv();
#endif
}

void GEBackend::Init() {
  if (is_initialized_) {
    return;
  }
  GilReleaseWithCheck gil_release;
  std::lock_guard<std::mutex> lock(init_mutex_);
  graph_executor_ = std::make_shared<GeGraphExecutor>();

  MS_LOG(INFO) << "Start initializing GE backend.";

  // set overflow mode
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &soc_version = ms_context->ascend_soc_version();
  if (soc_version == "ascend910b" || soc_version == "ascend910_93") {
    bool is_sat = (common::GetEnv("MS_ASCEND_CHECK_OVERFLOW_MODE") == "SATURATION_MODE");
    auto mode = (is_sat) ? aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION
                         : aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN;
    device::ascend::AscendHalManager::GetInstance().SetDeviceSatMode(mode);
  }

  // ascend_res_manager init
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->Initialize();

  // set MS_CTX_ENABLE_GE_HETEROGENOUS true according to heterogeneous mode
  ms_context->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, false);
  if (!UseSimulationApi()) {
    graph_executor_->Initialize();
  }

  device::ascend::AscendHalManager::GetInstance().InitializeAcl();

  auto op_tuning_conf = device::ascend::OpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  if (op_tuning_conf->EnableAoeOnline()) {
    std::string aoe_job_type = op_tuning_conf->aoe_job_type();
    backend::ge_backend::InitializeAoeUtil(aoe_job_type);
  }
  if (op_tuning_conf->EnableAoeOffline()) {
    backend::ge_backend::EnableAoeOffline();
  }
  // open tsd
  if (!OpenTsd(ms_context)) {
    MS_LOG(EXCEPTION) << "Open tsd failed";
  }

  is_initialized_ = true;
  MS_LOG(INFO) << "End initializing GE backend.";
}

void GEBackend::Clear() {
  if (!is_initialized_) {
    return;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  auto op_tuning_conf = device::ascend::OpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  if (op_tuning_conf->EnableAoeOnline()) {
    backend::ge_backend::DestroyAoeUtil();
  }

  graph_executor_->Finalize();

  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->Destroy();

  CloseTsd(true);
  is_initialized_ = false;
}

bool GEBackend::OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr) {
  std::unique_lock<std::mutex> lock(g_tsd_mutex);
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  if (ms_context_ptr->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return true;
  }

  if (UseSimulationApi()) {
    return true;
  }

  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) != 0) {
    MS_LOG(DEBUG) << "ACLTDT Dataset client is already opened.";
    ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);
    return true;
  }

  auto role = common::GetEnv("MS_ROLE");
  if (strcmp(role.c_str(), "MS_SCHED") == 0 || strcmp(role.c_str(), "MS_PSERVER") == 0) {
    return true;
  }

  uint32_t device_id = ms_context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  ms_context_ptr->increase_param<uint32_t>(MS_CTX_TSD_REF);

  if (!ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    device::ascend::MbufDataHandlerManager::GetInstance().AddHandler(std::make_unique<device::ascend::MbufDataHandler>(
      std::bind(&device::ascend::TensorPrintUtils::PrintReceiveData, &device::ascend::TensorPrintUtils::GetInstance(),
                std::placeholders::_1, std::placeholders::_2),
      device_id, kChannelNameNpuLog, kPrintOpName));
  }
  constexpr char kMbufTensorDumpCallback[] = "MbufTensorDumpCallback";
  static auto tensordump_callback =
    callback::CommonCallback::GetInstance().GetCallback<void, const std::string &, const std::vector<MbufDataItem> &>(
      kMbufTensorDumpCallback);
  if (tensordump_callback) {
    device::ascend::MbufDataHandlerManager::GetInstance().AddHandler(std::make_unique<device::ascend::MbufDataHandler>(
      tensordump_callback, device_id, kTensorDumpChannelName, kTensorDumpOpName));
  } else {
    MS_LOG(WARNING) << "Failed to get MbufTensorDumpCallback, tensor dump function may not work.";
  }

  if (device::ascend::TensorReportUtils::IsEnable()) {
    device::ascend::MbufDataHandlerManager::GetInstance().AddHandler(std::make_unique<device::ascend::MbufDataHandler>(
      std::bind(&device::ascend::TensorReportUtils::ReportReceiveData,
                &device::ascend::TensorReportUtils::GetInstance(), std::placeholders::_1, std::placeholders::_2),
      device_id, device::ascend::tensorreport_mapping.first, device::ascend::tensorreport_mapping.second));
  }
  for (const std::pair<string, string> &summary_mapping : device::ascend::summary_mappings) {
    device::ascend::MbufDataHandlerManager::GetInstance().AddHandler(std::make_unique<device::ascend::MbufDataHandler>(
      std::bind(device::ascend::SummaryReceiveData, std::placeholders::_1, std::placeholders::_2,
                summary_mapping.first),
      device_id, summary_mapping.first, summary_mapping.second));
  }

  return true;
}

bool GEBackend::CloseTsd(bool force) {
  std::unique_lock<std::mutex> lock(g_tsd_mutex);
  auto ms_context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_LOG(INFO) << "Start to close tsd, ref = " << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF);
  if (ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    return true;
  }
  ms_context_ptr->decrease_param<uint32_t>(MS_CTX_TSD_REF);
  if (force || ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) == 0) {
    ms_context_ptr->set_param<uint32_t>(MS_CTX_TSD_REF, 0);
    pybind11::gil_scoped_release gil_release;
    device::ascend::MbufDataHandlerManager::GetInstance().DestoryPrintHandler();
    device::ascend::MbufDataHandlerManager::GetInstance().DestoryHandler();
    ms_context_ptr->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
    MS_LOG(INFO) << "Call  close tsd successful.";
  } else {
    MS_LOG(DEBUG) << "Acltdt Dataset client is used, no need to close, tsd reference = "
                  << ms_context_ptr->get_param<uint32_t>(MS_CTX_TSD_REF) << ".";
  }
  return true;
}

BackendGraphId GEBackend::Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  if (!RegisterGlobalSignalHandler(DefaultIntHandler)) {
    MS_EXCEPTION(RuntimeError) << "Failed to register the callback signal handling.";
  }
#endif
  WaitTaskFinish();
  MS_EXCEPTION_IF_NULL(func_graph);
  // Clear the temp members of last graph.
  ClearGraphBuildMember();
  MS_LOG(INFO) << "Status record: start compile function graph: " << func_graph->ToString();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(compile_backend_graph);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  host_context->device_res_manager_->BindDeviceToCurrentThread(false);

  auto root_graph = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph);

  PROF_START(WaitAllCommInit);
  (void)distributed::collective::CollectiveManager::instance()->WaitAllCommInitDone();
  PROF_END(WaitAllCommInit);

  UnifyMindIR(root_graph);
  if (common::AnfAlgo::IsDynamicShapeFuncGraph(root_graph)) {
    opt::GEDynamicUnifyMindIR(root_graph);
  }

  // Register a summary callback function, which is called in the final stages of summary.
  constexpr char kRegisterSummaryCallBackFunc[] = "RegisterSummaryCallBackFunc";
  static auto register_summary_callback_func_callback =
    callback::CommonCallback::GetInstance().GetCallback<void>(kRegisterSummaryCallBackFunc);
  if (register_summary_callback_func_callback) {
    register_summary_callback_func_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get RegisterSummaryCallBackFunc, summary function may not work.";
  }
  // check if supported in ge_backend, and the compile_type
  auto compile_type = CheckGraph(func_graph);
  if (compile_type == CompileType::WholeGraph) {
    PROF_START(CompileWholeGraph);
    auto graph_id = CompileWholeGraph(func_graph, backend_jit_config);
    PROF_END(CompileWholeGraph);
    PROF_END(compile_backend_graph);
    (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCompileGraphs, start_time,
                                    profiler::GetClockSyscnt(), 1);
    graph_compile_type_[graph_id] = compile_type;
    // Clear the temp members.
    ClearGraphBuildMember();
    return graph_id;
  }
  if (compile_type == CompileType::SubGraph) {
    PROF_START(CompileSubGraph);
    auto graph_id = CompileSubGraph(func_graph, backend_jit_config);
    PROF_END(CompileSubGraph);
    PROF_END(compile_backend_graph);
    (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCompileGraphs, start_time,
                                    profiler::GetClockSyscnt(), 1);
    graph_compile_type_[graph_id] = compile_type;

    // Clear the temp members.
    ClearGraphBuildMember();
    return graph_id;
  }
  MS_LOG(EXCEPTION)
    << "The GE backend does not support subgraph sink and heterogeneous scenarios, please use the ms backend.";
  return 0;
}

void GEBackend::UnifyMindIR(const FuncGraphPtr &root_graph) const {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(root_graph->manager());
  // When the input is an empty sequence, the number of inputs will be recorded as 0, and the tensor cannot be
  // expressed, so the empty sequence is set to dynamic len.
  for (const auto &parameter : root_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(parameter);
    const auto &abs = parameter->abstract();
    if (abs != nullptr && abs->isa<abstract::AbstractSequence>()) {
      const auto &sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abs);
      if ((!sequence_abs->dynamic_len()) && sequence_abs->empty()) {
        MS_LOG(INFO) << "Set dynamic len flag for empty sequence input:" << parameter->DebugString();
        sequence_abs->set_dynamic_len(true);
      }
    }
  }
  const auto &graphs = root_graph->manager()->func_graphs();
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    auto output = graph->get_return();
    if (!output->isa<CNode>()) {
      continue;
    }
    auto seen = NewSeenGeneration();
    std::queue<AnfNodePtr> to_visit;
    to_visit.emplace(output);
    while (!to_visit.empty()) {
      auto node = to_visit.front();
      to_visit.pop();
      MS_EXCEPTION_IF_NULL(node);
      CheckNodeValid(node);

      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      UnifyIR(cnode);
      for (auto &input : cnode->inputs()) {
        MS_EXCEPTION_IF_NULL(input);
        if (input->seen_ == seen || !input->isa<CNode>()) {
          continue;
        }
        to_visit.emplace(input);
        input->seen_ = seen;
      }
    }
  }
  opt::UnifyMindIRPass(root_graph);
}

CompileType GEBackend::CheckGraph(const FuncGraphPtr &func_graph) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  // heterogeneous
  if (func_graph->exist_multi_target()) {
    MS_LOG(ERROR) << "The ge_backend do not support heterogeneous scenarios";
    return CompileType::NotSupport;
  }

  // lazy_inline + pipeline
  if (IsLazyinlineAndPipeline(func_graph)) {
    return CompileType::SubGraph;
  }

  // control flow | Closure\ENV\While scenario
  if (IsControlFlowGraph(func_graph)) {
    MS_LOG(ERROR) << "The ge_backend do not support control flow";
    return CompileType::NotSupport;
  }

  // whole graph and dynamic shape
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_dynamic_graph = common::AnfAlgo::IsDynamicShapeFuncGraph(func_graph);
  if (is_dynamic_graph) {
    MS_LOG(WARNING) << "The dynamic shape in ge backend is not full support yet, if some error occurred, please use "
                       "ms backend and try again.";
  }

  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  const auto &sub_graphs = mng->func_graphs();
  for (const auto &sub_graph : sub_graphs) {
    if (sub_graph == nullptr) {
      continue;
    }
    auto nodes = TopoSort(sub_graph->get_return());
    for (const auto &node : nodes) {
      if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
        continue;
      }
      if (GetCNodeTarget(node) != kAscendDevice) {
        MS_LOG(ERROR) << "The ge_backend do not support heterogeneous scenarios, the graph has cpu node, node: "
                      << node->fullname_with_scope();
        return CompileType::NotSupport;
      }
      if (GetCNodePrimitive(node) == nullptr) {
        continue;
      }
      if (is_dynamic_graph && common::AnfAlgo::IsDynamic(node)) {
        if (!device::ascend::ConvertCheck(node)) {
          MS_LOG(ERROR) << node->fullname_with_scope() << " can not find adpt.";
          return CompileType::NotSupport;
        }
        if (!device::ascend::DynamicShapeSupportCheck(node)) {
          MS_LOG(ERROR) << node->fullname_with_scope() << " not support dynamic shape.";
          return CompileType::NotSupport;
        }
        if (!device::ascend::SinkGraphCheck(node)) {
          MS_LOG(ERROR) << node->fullname_with_scope() << " have attrs is not ValueNode.";
          return CompileType::NotSupport;
        }
      }

      if (common::AnfAlgo::GetGraphSplitGroup(node) == kKernelGroup) {
        MS_LOG(ERROR) << "Ge backend do not support kernel group.";
        return CompileType::NotSupport;
      }
    }
  }

  return CompileType::WholeGraph;
}

FuncGraphPtr GEBackend::WrapPrimitives(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  FuncGraphManagerPtr manager_ptr = graph->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  MapPrimTypeFuncGraph prim_graphs;
  const auto &get_prim_graph = [&prim_graphs](const PrimitivePtr &prim, const abstract::AbstractFunctionPtr &type) {
    PrimTypePair prim_type = std::make_pair(prim, type);
    if (prim_graphs.end() == prim_graphs.find(prim_type)) {
      FuncGraphPtr g = std::make_shared<FuncGraph>();
      std::vector<AnfNodePtr> args;
      ValueNodePtr prim_ct = NewValueNode(prim);
      MS_EXCEPTION_IF_NULL(prim_ct);
      prim_ct->set_abstract(type);
      args.push_back(prim_ct);
      MS_EXCEPTION_IF_NULL(type);
      TypedPrimitiveAbstractClosurePtr tp = dyn_cast<abstract::TypedPrimitiveAbstractClosure>(type->GetUnique());
      if (tp == nullptr) {
        MS_LOG(INTERNAL_EXCEPTION) << "Not TypedPrimitiveAbstractClosure, but got " << type->GetUnique()->ToString();
      }
      MS_EXCEPTION_IF_NULL(g);
      for (const auto &t : tp->args_abs_list()) {
        ParameterPtr p = g->add_parameter();
        p->set_abstract(t);
        args.push_back(p);
      }
      AnfNodePtr out = g->NewCNode(args);
      out->set_abstract(tp->output());
      g->set_output(out);
      prim_graphs[prim_type] = g;
    }

    return prim_graphs[prim_type];
  };

  FuncGraphTransaction tr = manager_ptr->Transact();
  auto &fgs = manager_ptr->func_graphs();
  TraverseGraphMap(manager_ptr, &tr, fgs, get_prim_graph);
  tr.Commit();

  return graph;
}

void GEBackend::TraverseGraphMap(
  const FuncGraphManagerPtr &manager_ptr, FuncGraphTransaction *tr, const FuncGraphSet &fgs,
  const std::function<std::shared_ptr<FuncGraph>(const PrimitivePtr, const abstract::AbstractFunctionPtr)>
    &get_prim_graph) {
  MS_EXCEPTION_IF_NULL(manager_ptr);
  MS_EXCEPTION_IF_NULL(tr);
  for (const auto &fg : fgs) {
    MS_EXCEPTION_IF_NULL(fg);
    for (const auto &ct_any : fg->value_nodes()) {
      AnfNodePtr const_primitive_node = ct_any.first;
      if (const_primitive_node != nullptr && IsValueNode<Primitive>(const_primitive_node)) {
        auto users = manager_ptr->node_users()[const_primitive_node];
        for (auto &use : users) {
          CNodePtr node = use.first->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(node);
          if (node->func_graph() != fg) {
            continue;
          }
          int64_t key = use.second;
          if (key != 0) {
            MS_EXCEPTION_IF_NULL(node->input(0));
            bool key_is_const = node->input(0)->isa<ValueNode>();
            PrimitivePtr value = GetValueNode<PrimitivePtr>(node->input(0));
            if (value != nullptr) {
              bool is_prim_array_map = !(prim::kPrimArrayMap->name().compare(value->name()));
              bool is_prim_array_reduce = !(prim::kPrimArrayReduce->name().compare(value->name()));
              if (key == 1 && key_is_const && (is_prim_array_map || is_prim_array_reduce)) {
                continue;
              }
            }
            FuncGraphPtr g = get_prim_graph(GetValueNode<PrimitivePtr>(const_primitive_node),
                                            dyn_cast<AbstractFunction>(const_primitive_node->abstract()));
            tr->SetEdge(node, key, NewValueNode(g));
          }
        }
      }
    }
  }
}

BackendGraphId GEBackend::CompileWholeGraph(const FuncGraphPtr &func_graph,
                                            const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);

  MS_LOG(INFO) << "Status record: start compile graph." << func_graph->ToString();
  // Generate kernel graph.
  std::vector<KernelGraphPtr> all_graphs;

  auto kg_mgr = std::make_shared<session::KernelGraphMgr>();
  KernelGraphPtr root_graph =
    kg_mgr->ConstructKernelGraph(func_graph, &all_graphs, device::DeviceType::kAscend, backend_jit_config);
  MS_EXCEPTION_IF_NULL(root_graph);
  if (AnfAlgo::IsGraphOutputValueNodeOrParameterForCompile(root_graph->output())) {
    auto cur_backend_graph_id = backend_graph_id_++;
    graph_map_[cur_backend_graph_id] = root_graph;
    graph_run_iter_[root_graph] = 0;
    root_graph_map_[cur_backend_graph_id] = func_graph;
    MS_LOG(INFO) << "Status record: end compile graph. backend_graph_id: " << cur_backend_graph_id
                 << ", kernel graph id: " << root_graph->graph_id();
    return cur_backend_graph_id;
  }

  for (const auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    MS_LOG(INFO) << "Set root graph for graph: " << graph->graph_id() << " to: " << root_graph->graph_id() << ".";
    graph->set_root_graph_id(root_graph->graph_id());
    graph->set_run_mode(device::RunMode::kGraphMode);
    graph->set_is_loop_count_sink(true);
    graph->set_attrs(func_graph->attrs());
    opt::OptimizationWithoutBackend(graph);
  }

  graph_executor_->OptimizeBeforeCompileGraph(root_graph);

  auto manager = MakeManager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    graph->set_flag(kFlagEnableZeroCopyInGraph, true);
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
    graph->SetInputNodes();
  }
  root_graph->SetInputNodes();

  if (!graph_executor_->CompileGraph(std::dynamic_pointer_cast<FuncGraph>(root_graph), {})) {
    MS_LOG(EXCEPTION) << "Compile graph failed: " << root_graph->graph_id();
  }
  root_graph->CacheGraphOutputToFrontNodeWithIndex({root_graph->output()}, {func_graph->output()});

  graph_executor_->InitGraphInfo(root_graph);

  auto cur_backend_graph_id = backend_graph_id_;
  ++backend_graph_id_;
  graph_map_[cur_backend_graph_id] = root_graph;
  graph_run_iter_[root_graph] = 0;
  root_graph_map_[cur_backend_graph_id] = func_graph;
  MS_LOG(INFO) << "Status record: end compile graph. backend_graph_id: " << cur_backend_graph_id
               << ", kernel graph id: " << root_graph->graph_id();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_target = device::GetDeviceTypeByName(context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  device::DeviceContextManager::GetInstance().GetMultiStreamController(device_target)->Refresh();
  return cur_backend_graph_id;
}

void GEBackend::WaitTaskFinish() const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  runtime::Pipeline::Get().WaitAll();
}

void GEBackend::WaitMultiStream() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  if (device::ascend::AscendStreamMng::GetInstance().single_op_multi_stream_enable()) {
    device::DeviceContextManager::GetInstance()
      .GetMultiStreamController(device::GetDeviceTypeByName(device_target))
      ->WaitMultiStream(device::ascend::AscendStreamMng::GetInstance().default_stream_id());
  }
}

RunningStatus GEBackend::Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
  MSTX_RANGE_GUARD("GEBackendRun", nullptr, profiler::MSTX_DOMAIN_MODEL_PREPARATION);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kBackendGraphRunInner,
                                     "graph_" + std::to_string(graph_id), true);

  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_PRECOMPILE_ONLY) || common::GetEnv("MS_DEV_PRECOMPILE_ONLY") == "1") {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return kRunningSuccess;
  }

  // Open abstract_lock for dynamic_shape
  AnfUtils::OpenAbstractLock();
  WaitTaskFinish();
  // wait for other streams finish
  WaitMultiStream();
  // Release python gil.
  mindspore::ScopedLongRunning long_running;

  if (graph_compile_type_[graph_id] == CompileType::WholeGraph) {
    RunWholeGraph(graph_id, inputs, outputs);
    return kRunningSuccess;
  }
  if (graph_compile_type_[graph_id] == CompileType::SubGraph) {
    RunSubGraph(graph_id, inputs, outputs);
    return kRunningSuccess;
  }
  MS_LOG(EXCEPTION) << "The graph is not supported in ge backend.";
  return kRunningFailure;
}

void GEBackend::ConstructOutputByTupleTensor(tensor::TensorPtr output_tensor,
                                             const abstract::SequenceShapePtr &tensor_shape, VectorRef *outputs,
                                             std::vector<tensor::TensorPtr> *tuple_tensors,
                                             const TypePtr &output_type) const {
  MS_LOG(EXCEPTION) << "Not support real sequence output in ge backend, output shape:" << tensor_shape->ToString();
}

BaseRef GEBackend::ConstructOutputByAbstract(const abstract::AbstractBasePtr &abstract,
                                             const std::vector<tensor::TensorPtr> &output_tensors,
                                             size_t *output_position, std::vector<tensor::TensorPtr> *tuple_tensors,
                                             const std::vector<TypePtr> &output_types) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(output_position);
  MS_EXCEPTION_IF_NULL(tuple_tensors);

  size_t outputs_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
  if (*output_position + outputs_num > output_tensors.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The output position is out of range: "
                               << *output_position << " need:" << outputs_num << " total:" << output_tensors.size();
  }

  if (!abstract->isa<abstract::AbstractSequence>()) {
    if (IsTupleOutputOfAnyType(abstract, output_tensors[*output_position])) {
      MS_LOG(DEBUG) << "Any output for position:" << *output_position;
      VectorRef outputs;
      ConstructOutputByTupleTensor(
        output_tensors[*output_position],
        output_tensors[*output_position]->base_shape_ptr()->cast<abstract::SequenceShapePtr>(), &outputs, tuple_tensors,
        output_types[*output_position]);
      (*output_position)++;
      std::vector<ValuePtr> values;

      (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(values),
                           [](const auto &output) { return utils::cast<ValuePtr>(output); });
      return std::make_shared<ValueList>(values);
    }

    (*output_position)++;
    return output_tensors[(*output_position) - 1];
  }

  VectorRef outputs;
  const auto &tuple_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  // Dynamic len tuple.
  if (tuple_abstract->dynamic_len()) {
    auto &output_tensor = output_tensors[*output_position];
    MS_EXCEPTION_IF_NULL(output_tensor);
    auto &tensor_shape = output_tensor->base_shape_ptr();
    // Restore the tuple output by the tensor of tuple.
    if ((tensor_shape != nullptr) && tensor_shape->isa<abstract::SequenceShape>()) {
      ConstructOutputByTupleTensor(output_tensor, tensor_shape->cast<abstract::SequenceShapePtr>(), &outputs,
                                   tuple_tensors, output_types[*output_position]);
      (*output_position)++;
      return outputs;
    }
  }

  const auto &sub_abstracts = tuple_abstract->elements();
  for (const auto &sub_abstract : sub_abstracts) {
    MS_EXCEPTION_IF_NULL(sub_abstract);
    outputs.emplace_back(
      ConstructOutputByAbstract(sub_abstract, output_tensors, output_position, tuple_tensors, output_types));
  }
  return outputs;
}

void GEBackend::ClearGraph(BackendGraphId backend_graph_id) {
  (void)graph_map_.erase(backend_graph_id);
  (void)root_graph_map_.erase(backend_graph_id);
  (void)graph_compile_type_.erase(backend_graph_id);
  (void)graph_id_to_graph_compiler_info_.erase(backend_graph_id);
}

void GEBackend::ConstructOutputs(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                                 size_t *output_position, VectorRef *outputs,
                                 std::vector<tensor::TensorPtr> *tuple_tensors,
                                 const std::vector<TypePtr> &output_types) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_position);
  MS_EXCEPTION_IF_NULL(tuple_tensors);
  static const PrimitiveSet expand_prims{
    prim::kPrimMakeTuple,
    prim::kPrimMakeCSRTensor,
    prim::kPrimMakeCOOTensor,
    prim::kPrimMakeRowTensor,
  };
  MS_LOG(DEBUG) << "output node:" << output_node->DebugString();
  // If outputs an empty sequence return an empty sequence value.
  if (IsEmptySequence(output_node, output_tensors, output_position)) {
    if (output_node->abstract()->isa<abstract::AbstractTuple>()) {
      outputs->emplace_back(std::make_shared<ValueTuple>(std::vector<ValuePtr>()));
    } else {
      outputs->emplace_back(std::make_shared<ValueList>(std::vector<ValuePtr>()));
    }
    ++(*output_position);
    return;
  }

  // The MakeTuple/MakeSaprse node need expand and recurse.
  if (IsOneOfPrimitiveCNode(output_node, expand_prims)) {
    auto make_tuple = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    VectorRef make_tuple_output;
    for (size_t i = 1; i < make_tuple->size(); i++) {
      ConstructOutputs(make_tuple->input(i), output_tensors, output_position, &make_tuple_output, tuple_tensors,
                       output_types);
    }
    outputs->emplace_back(std::move(make_tuple_output));
    return;
  }

  // The depend node need get the real node.
  if (common::AnfAlgo::CheckPrimitiveType(output_node, prim::kPrimDepend)) {
    auto depend_node = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_node);
    ConstructOutputs(depend_node->input(kRealInputIndexInDepend), output_tensors, output_position, outputs,
                     tuple_tensors, output_types);
    return;
  }

  auto outputs_num = AnfAlgo::GetOutputElementNum(output_node);
  // The value node uses the value to be output, to avoid the host memory of value free due to value node destruction.
  if (output_node->isa<ValueNode>()) {
    auto value = output_node->cast<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueSequence>()) {
      outputs->emplace_back(value);
      (*output_position) += CountValueNum(value->cast<ValueSequencePtr>());
    } else if (outputs_num != 0) {
      outputs->emplace_back(value);
      (*output_position) += outputs_num;
    }
    // The empty value node return the empty VectorRef.
    return;
  }

  if (common::AnfAlgo::IsCallNode(output_node) ||
      (output_node->abstract() != nullptr && output_node->abstract()->isa<abstract::AbstractSequence>())) {
    auto abstract = output_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    outputs->emplace_back(
      ConstructOutputByAbstract(abstract, output_tensors, output_position, tuple_tensors, output_types));
    return;
  }

  auto &output_abstract = output_node->abstract();
  MS_EXCEPTION_IF_NULL(output_abstract);
  // Wrap output to VectorRef if the output is tuple.
  MS_LOG(DEBUG) << "output abstract:" << output_abstract->ToString();
  if (output_abstract->isa<abstract::AbstractSequence>()) {
    VectorRef output_tuple;
    for (size_t i = 0; i < outputs_num; ++i) {
      MS_LOG(DEBUG) << "output index:" << i;
      CheckOutputIdx(*output_position, output_tensors.size());

      auto &output_tensor = output_tensors[*output_position];
      MS_EXCEPTION_IF_NULL(output_tensor);
      auto &tensor_shape = output_tensor->base_shape_ptr();
      // Restore the tuple output by the tensor of tuple.
      if ((tensor_shape != nullptr) && tensor_shape->isa<abstract::SequenceShape>()) {
        ConstructOutputByTupleTensor(output_tensor, tensor_shape->cast<abstract::SequenceShapePtr>(), &output_tuple,
                                     tuple_tensors, output_types[*output_position]);
      } else {
        output_tuple.emplace_back(output_tensor);
      }
      ++(*output_position);
    }
    outputs->emplace_back(std::move(output_tuple));
  } else {
    for (size_t i = 0; i < outputs_num; ++i) {
      CheckOutputIdx(*output_position, output_tensors.size());
      outputs->emplace_back(output_tensors[*output_position]);
      ++(*output_position);
    }
  }
}

void GEBackend::SetTensorUpdateCallback(const tensor::TensorPtr &update_tensor) {
  if (update_tensor != nullptr && update_tensor->update_value_callback() == nullptr && update_tensor->is_parameter()) {
    static auto callback = [this](const tensor::Tensor *tensor) { weights_need_reprepare_.insert(tensor); };
    update_tensor->set_update_value_callback(callback);
  }
}

void GEBackend::UpdateInputsShapeAndSize(const ParameterPtr &input_node,
                                         const mindspore::kernel::KernelTensorPtr &kernel_tensor,
                                         const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const auto &device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  MS_EXCEPTION_IF_NULL(input_tensor);
  // update shape and size, for dynamic shape
  if (!input_node->has_dynamic_shape() && !IsDynamic(kernel_tensor->GetShapeVector())) {
    return;
  }

  // update shape
  MS_LOG(DEBUG) << "Update dynamic shape for parameter:" << input_node->DebugString();
  const auto &output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  if (input_tensor->base_shape_ptr() == nullptr || (!input_tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    output_kernel_tensor->SetShape(input_tensor->ToAbstract()->GetShape());
    graph_executor_->AllocInputMemory(device_tensor);
    return;
  }
  output_kernel_tensor->SetShape(input_tensor->base_shape_ptr());

  // Update size.
  auto device_format = kernel::GetFormatFromEnumToStr(kernel_tensor->format());
  static const std::set<std::string> kNormalFormat = {
    kOpFormat_DEFAULT, kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN,
  };
  if (kNormalFormat.find(device_format) != kNormalFormat.end()) {
    auto tensor_data_size = input_tensor->DataNBytes();
    MS_LOG(DEBUG) << "Set device address:" << device_tensor << " size from:" << device_tensor->GetSize()
                  << " to:" << tensor_data_size;
    device_tensor->SetSize(tensor_data_size);
  } else {
    MS_LOG(DEBUG) << "Update data node device address size";
    // Size of 5D format device_tensor is larger than tensor_data_size.
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(input_node, 0);
    if (output_type_id == kTypeUnknown) {
      output_type_id = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
    }
    auto device_shape = trans::TransShapeToDevice(
      input_tensor->shape(), kernel::GetFormatFromEnumToStr(kernel_tensor->format()), input_node, 0, output_type_id);
    size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
    auto device_address_size = type_size * SizeOf(device_shape);
    MS_LOG(INFO) << "Size of device_address is updated from " << device_tensor->GetSize() << " to "
                 << device_address_size;
    device_tensor->SetSize(device_address_size);
  }

  graph_executor_->AllocInputMemory(device_tensor);
}

void GEBackend::ConstructInputsRefMode(const KernelGraphPtr &func_graph, const VectorRef &args,
                                       std::vector<tensor::TensorPtr> *inputs_tensor) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto inputs = func_graph->inputs();
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == args.size(), "The args size is not equal to graph inputs size.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::vector<tensor::TensorPtr> flatten_tensors;
    auto params = common::AnfAlgo::GetAllOutput(inputs[i]);
    for (size_t j = 0; j < params.size(); ++j) {
      auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(params[j], 0, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto device_tensor = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_tensor);
      // skip const input
      if (TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
        MS_LOG(INFO) << "The input[" << i << "] is convert to const op, skip.";
        continue;
      }
      // for refmode, weight copy to device just once
      auto parameter = params[j]->cast<ParameterPtr>();
      if (is_weight_init_[parameter] && weights_need_reprepare_.empty()) {
        continue;
      }
      // get host tensor
      if (flatten_tensors.empty()) {
        const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(inputs[i], *func_graph);
        AnfAlgo::FlattenInputArg(args[i], front_node, &flatten_tensors);
        MS_EXCEPTION_IF_CHECK_FAIL(flatten_tensors.size() == params.size(),
                                   "The flatten_tensors size is not equal to params size.");
      }

      bool is_need_sync = true;
      auto host_tensor_address = flatten_tensors[j]->device_address();

      UpdateInputsShapeAndSize(parameter, kernel_tensor, flatten_tensors[j]);
      CheckContiguousTensor(flatten_tensors[j]);
      // in different backend object, but has init, skip
      if (common::AnfAlgo::IsParameterWeight(parameter)) {
        is_weight_init_[parameter] = true;
        // for weight value update in python
        SetTensorUpdateCallback(flatten_tensors[j]);

        kernel_tensor->set_is_ptr_persisted(true);
        if (host_tensor_address == device_tensor) {
          continue;
        }

        if (host_tensor_address == nullptr) {
          // host is nullptr -> set & copy_to_device
          host_tensor_address = device_tensor;
          flatten_tensors[j]->set_device_address(host_tensor_address);
          flatten_tensors[j]->set_format(kernel_tensor->format());
          is_need_sync = true;
        } else if (host_tensor_address->GetDeviceType() != device_tensor->GetDeviceType()) {
          // device_type not same -> sync_to_host & copy_to_device
          auto origin_tensor = flatten_tensors[j];
          flatten_tensors[j] = flatten_tensors[j]->cpu();
          host_tensor_address = device_tensor;
          origin_tensor->set_device_address(device_tensor);
          origin_tensor->set_format(kernel_tensor->format());
          is_need_sync = true;
        } else {
          // other not same condition -> device_copy
          if (!Copy(kernel_tensor.get(), flatten_tensors[j])) {
            MS_LOG(EXCEPTION) << "Sync data error.";
          }
          host_tensor_address = device_tensor;
          flatten_tensors[j]->set_device_address(device_tensor);
          flatten_tensors[j]->set_format(kernel_tensor->format());
          is_need_sync = false;
        }
      } else {
        if (host_tensor_address == device_tensor) {
          continue;
        }

        if (host_tensor_address != nullptr) {
          if (host_tensor_address->GetPtr() == device_tensor->GetPtr()) {
            continue;
          } else if (host_tensor_address->GetPtr() == nullptr) {
            flatten_tensors[j]->set_device_address(nullptr);
            host_tensor_address = nullptr;
            is_need_sync = true;
          } else if (host_tensor_address->GetDeviceType() != device_tensor->GetDeviceType()) {
            // device type not same: tensor sync to host & copy to device_tensor
            flatten_tensors[j] = flatten_tensors[j]->cpu();
            is_need_sync = true;
          } else {
            host_tensor_address =
              std::dynamic_pointer_cast<mindspore::device::DeviceAddress>(flatten_tensors[j]->device_address());
            // other not same: device copy
            if (!Copy(kernel_tensor.get(), flatten_tensors[j])) {
              MS_LOG(EXCEPTION) << "Sync data error.";
            }
            is_need_sync = false;
          }
        } else {
          is_need_sync = true;
        }
      }
      if (is_need_sync) {
        SyncTensorData(flatten_tensors[j], kernel_tensor.get(), params[j]);
      }
    }
  }
  // clear every step
  weights_need_reprepare_.clear();
  // GEBackend is only created once in new backend.
  is_weight_init_.clear();
}

void GEBackend::ConstructInputs(const KernelGraphPtr &func_graph, const VectorRef &args,
                                std::vector<tensor::TensorPtr> *inputs_tensor) {
  ConstructInputsRefMode(func_graph, args, inputs_tensor);
}

bool GEBackend::Copy(KernelTensor *const dst_kernel_tensor, const tensor::TensorPtr &src_tensor) const {
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor);
  MS_EXCEPTION_IF_NULL(src_tensor);
  const auto &src_device_tensor = src_tensor->device_address();
  const auto &dst_device_tensor = dst_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(src_device_tensor);
  MS_EXCEPTION_IF_NULL(dst_device_tensor);
  if (src_device_tensor->GetSize() != dst_device_tensor->GetSize()) {
    MS_LOG(INFO) << "Copy size is not equal, input size:" << src_device_tensor->GetSize()
                 << ", output size:" << dst_device_tensor->GetSize();
    if (src_tensor->format() == dst_kernel_tensor->format()) {
      auto new_address_size = GetTypeByte(TypeIdToType(src_tensor->data_type())) * SizeOf(src_tensor->shape());
      src_device_tensor->SetSize(new_address_size);
    }
  }
  // Exist the size alignment in some device, so get the min device size.
  auto target_device_address =
    (dst_device_tensor->GetDeviceType() == device::DeviceType::kCPU ? src_device_tensor : dst_device_tensor);
  if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
    return false;
  }
  return SyncCopy(dst_kernel_tensor, src_tensor.get(), target_device_address->stream_id());
}

bool GEBackend::Copy(KernelTensor *const dst_kernel_tensor, KernelTensor *const src_kernel_tensor) const {
  MS_EXCEPTION_IF_NULL(dst_kernel_tensor);
  MS_EXCEPTION_IF_NULL(src_kernel_tensor);
  const auto &src_device_tensor = src_kernel_tensor->device_address();
  const auto &dst_device_tensor = dst_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(src_device_tensor);
  MS_EXCEPTION_IF_NULL(dst_device_tensor);
  if (src_device_tensor->GetSize() != dst_device_tensor->GetSize()) {
    MS_LOG(INFO) << "Copy size is not equal, input size:" << src_device_tensor->GetSize()
                 << ", output size:" << dst_device_tensor->GetSize();
    if (src_kernel_tensor->format() == dst_kernel_tensor->format()) {
      auto new_address_size =
        GetTypeByte(TypeIdToType(src_kernel_tensor->dtype_id())) * SizeOf(src_kernel_tensor->GetShapeVector());
      src_device_tensor->SetSize(new_address_size);
    }
  }
  // Exist the size alignment in some device, so get the min device size.
  auto target_device_address =
    (dst_device_tensor->GetDeviceType() == device::DeviceType::kCPU ? src_device_tensor : dst_device_tensor);
  if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
    return false;
  }
  return SyncCopy(dst_kernel_tensor, src_kernel_tensor, target_device_address->stream_id());
}

void GEBackend::ConstructOutputs(const KernelGraphPtr &func_graph, std::vector<tensor::TensorPtr> *outputs,
                                 std::vector<TypePtr> *output_types) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_types);
  auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(func_graph->output());
  // map of output_node ptr and corresponding tensor, for same output condition
  // 1. same device_address; 2. io_index, same pointer_ref_count
  mindspore::HashMap<DevicePointerPtr, KernelTensorPtr> output_node_tensor_map;
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    const auto &output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, idx, false);
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_addr = output_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(output_addr);

    // when output_addr exist, need gen fake output
    if (common::AnfAlgo::IsNoOuputNode(output_node) && output_addr == nullptr) {
      continue;
    }

    auto out_tensor = tensor::from_spec(output_kernel_tensor->dtype_id(), output_kernel_tensor->GetShapeVector(),
                                        device::DeviceType::kNone);

    auto kernel_tensor =
      AnfAlgo::CreateKernelTensor(nullptr, output_addr->GetSize(), output_kernel_tensor->format(),
                                  output_kernel_tensor->dtype_id(), output_kernel_tensor->GetShapeVector(),
                                  kAscendDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID));
    kernel_tensor->SetType(output_kernel_tensor->GetType());
    kernel_tensor->SetShape(output_kernel_tensor->GetShape());
    kernel_tensor->set_stream_id(output_addr->stream_id());
    // SetShape will calculate a default size by host shape, need to set real device size for special format.
    kernel_tensor->set_size(output_addr->GetSize());
    auto tensor_device_address =
      graph_executor_->CreateDeviceAddress(kernel_tensor, output_kernel_tensor->is_ptr_persisted());
    MS_EXCEPTION_IF_NULL(tensor_device_address);

    if (output_kernel_tensor->is_ptr_persisted()) {
      // device_tensor persisted or format not same -> device_copy
      if (!Copy(kernel_tensor.get(), output_kernel_tensor.get())) {
        MS_LOG(EXCEPTION) << "Sync data error.";
      }
    } else if (output_node_tensor_map[output_addr->device_pointer()] != nullptr) {
      // create new device_address because they may have same ptr but different shape
      auto tmp_kernel_tensor = output_node_tensor_map[output_addr->device_pointer()];
      kernel_tensor->set_pointer_ref_count(tmp_kernel_tensor.get());
    } else {
      output_node_tensor_map[output_addr->device_pointer()] = kernel_tensor;
      MS_LOG(DEBUG) << "Swap from kernel tensor:" << output_kernel_tensor->ToString()
                    << " to: " << kernel_tensor->ToString() << ", output node:" << output_node->fullname_with_scope()
                    << " output index:" << idx << " in funcgraph:" << func_graph->ToString();
      output_kernel_tensor->Swap(kernel_tensor.get());
      kernel_tensor->set_managed_by_somas(output_kernel_tensor->managed_by_somas());
    }

    MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString()
                  << ", output node:" << output_node->fullname_with_scope() << " output index:" << idx
                  << ", origin output kernel tensor: " << output_kernel_tensor->ToString();
    out_tensor->set_device_address(tensor_device_address);
    out_tensor->set_format(kernel_tensor->format());
    outputs->emplace_back(out_tensor);
    output_types->emplace_back(kernel_tensor->GetType());
  }
}

void GEBackend::RunWholeGraph(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
  MS_LOG(INFO) << "Status record: start run graph: " << graph_id;
  MS_EXCEPTION_IF_NULL(outputs);

  if (graph_map_.find(graph_id) == graph_map_.end()) {
    MS_LOG(EXCEPTION) << "The graph is not found, graph: " << graph_id;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  MS_EXCEPTION_IF_NULL(graph_executor_);
  auto func_graph = graph_map_[graph_id];
  if (common::AnfAlgo::IsGraphOutputValueNodeOrParameter(func_graph->output(), inputs, outputs)) {
    MS_LOG(INFO) << "Status record: end run graph: " << graph_id;
    return;
  }

// for data_dump
#ifndef ENABLE_SECURITY
  bool dump_flag = DebugOnStepBegin(func_graph);
#endif

  // for profiling
  bool profile_started = ProfilerOnStepBegin(func_graph);

  // alloc input(static), output device memory; dynamic input will alloc later
  graph_executor_->AllocGEInputOutputMemory(func_graph);
  // alloc fixed feature memory when enable gekernel, once | const memory alloc in compilegraph
  graph_executor_->AllocGEFixMemory();
  // alloc refreshable feature memory
  graph_executor_->AllocGERefreshableFeatureMemory(func_graph);
  // const alloc in compile graph

  // input, weight from host(inputs) to device(device_address in graph)
  std::vector<tensor::TensorPtr> inputs_tensor;
  ConstructInputs(func_graph, inputs, &inputs_tensor);

  // run graph
  {
    std::vector<tensor::TensorPtr> outputs_tensor;
    const std::map<string, string> compile_options;
    MS_LOG(INFO) << "Start run graph, input size: " << inputs_tensor.size();
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kGraphLaunch,
                                       func_graph->ToString());
    auto ret = graph_executor_->RunGraph(func_graph, inputs_tensor, &outputs_tensor, compile_options);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch graph failed, graph id: " + std::to_string(func_graph->graph_id());
    }
  }
  auto ret = host_context->device_res_manager_->SyncAllStreams(false);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync Stream failed";
  }

  // output ->std::vector<tensor::TensorPtr> *outputs
  std::vector<tensor::TensorPtr> output_tensors;
  std::vector<TypePtr> output_types;
  ConstructOutputs(func_graph, &output_tensors, &output_types);
  if (!output_tensors.empty()) {
    size_t output_position = 0;
    std::vector<tensor::TensorPtr> tuple_tensors;
    // std::vector<tensor::TensorPtr> ->VectorRef *outputs
    ConstructOutputs(root_graph_map_[graph_id]->output(), output_tensors, &output_position, outputs, &tuple_tensors,
                     output_types);
  }

// for data_dump
#ifndef ENABLE_SECURITY
  DebugOnStepEnd(func_graph, dump_flag);
#endif

  // for profiling
  ProfilerOnStepEnd(profile_started);

  // free resource
  graph_executor_->FreeGERefreshableFeatureMemory(func_graph);
  graph_executor_->FreeInputOutputMemory(func_graph);

  graph_run_iter_[func_graph]++;
  MS_LOG(INFO) << "Status record: end run graph: " << graph_id;
  return;
}

bool GEBackend::DebugOnStepBegin(const KernelGraphPtr &func_graph) {
  MS_LOG(INFO) << "Debug on step begin.";
  if (func_graph->IsDatasetGraph()) {
    return false;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
#ifndef ENABLE_SECURITY
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profiler == nullptr || !profiler->IsInitialized()) {
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto &hookDebugger = dump::HookDebugger::GetInstance();
    if (hookDebugger.IsHookerEnabled()) {
      MS_LOG(INFO) << "On step begin, hookdebugger is enable.";
      auto step_count_num = graph_run_iter_[func_graph];
      hookDebugger.HookOnStepBegin(device_id, func_graph, step_count_num, false);
      return true;
    }
  }
#endif
  return false;
}

void GEBackend::DebugOnStepEnd(const KernelGraphPtr &graph, bool dump_flag) {
  if (!dump_flag) {
    return;
  }
#ifndef ENABLE_SECURITY
  MS_LOG(INFO) << "Debug on step end.";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  auto &hookDebugger = dump::HookDebugger::GetInstance();
  if (hookDebugger.IsHookerEnabled()) {
    MS_LOG(INFO) << "On step end, hookdebugger is enable.";
    host_context->device_res_manager_->SyncAllStreams(false);
    hookDebugger.HookOnStepEnd();
  }
#endif
  host_context->device_res_manager_->SyncAllStreams(false);
}

bool GEBackend::ProfilerOnStepBegin(const KernelGraphPtr &graph) {
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profiler == nullptr || !profiler->IsInitialized() || !profiler->GetEnableFlag()) {
    return false;
  }
  if (graph->IsDatasetGraph()) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  if (device::GetDeviceTypeByName(device_name) != device::DeviceType::kAscend) {
    MS_LOG(EXCEPTION) << "GE backend only support Ascend, but got " << device_name;
  }

  host_context->device_res_manager_->BindDeviceToCurrentThread(false);
  MS_LOG(INFO) << "Dot step start timestamp.";
  profiler->StepStart(graph_run_iter_[graph], host_context->device_res_manager_->GetStream());
  return true;
}

void GEBackend::ProfilerOnStepEnd(bool profile_started) {
  if (!profile_started) {
    return;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  host_context->device_res_manager_->BindDeviceToCurrentThread(false);
  host_context->device_res_manager_->SyncAllStreams(false);
  MS_LOG(INFO) << "Dot step end timestamp.";
  profiler->StepStop();
  host_context->device_res_manager_->SyncAllStreams(false);
}

void GEBackend::ConvertIR(const FuncGraphPtr &func_graph,
                          const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                          IRFormat ir_format) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (ir_format != IRFormat::kAir) {
    MS_LOG(EXCEPTION) << "The ir format not support.";
  }

  MS_EXCEPTION_IF_NULL(graph_executor_);
  std::map<std::string, std::shared_ptr<tensor::Tensor>> real_init_tensors{};
  const auto &infer_need_update_parameter_names = graph_executor_->GetInferParameterNames();

  graph_executor_->BuildDFGraph(func_graph, real_init_tensors, true);
}

std::string GEBackend::ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file,
                                IRFormat ir_format) {
  if (ir_format != IRFormat::kAir) {
    MS_LOG(EXCEPTION) << "The ir format not support.";
  }

  MS_EXCEPTION_IF_NULL(graph_executor_);
  return graph_executor_->ExportDFGraph(file_name, anf_graph, is_save_to_file);
}

BackendGraphId GEBackend::CompileSubGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Status record: start compile graph: " << func_graph->ToString();
  // compile graph
  auto manager = func_graph->manager();
  CompileGraph(func_graph, backend_jit_config);
  auto mscontext = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(mscontext);
  MS_EXCEPTION_IF_NULL(manager);
  const auto &sub_graphs = manager->func_graphs_used_total(func_graph);
  std::vector<FuncGraphPtr> cand_graph(sub_graphs.begin(), sub_graphs.end());
  std::sort(cand_graph.begin(), cand_graph.end(),
            [](const FuncGraphPtr &a, const FuncGraphPtr &b) { return a->ToString() < b->ToString(); });
  for (const auto &sub_graph : cand_graph) {
    MS_EXCEPTION_IF_NULL(sub_graph);
    bool skip_inline_graph =
      (sub_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE) && mscontext->CellReuseLevel() == CellReuseLevel::kLazyInline) ||
      sub_graph->has_flag(kFlagSwitchInline);
    if (sub_graph != func_graph && sub_graph != nullptr && !sub_graph->has_flag(kFlagJitCallGraph) &&
        !skip_inline_graph) {
      MS_LOG(INFO) << "Compile sub graph " << sub_graph->ToString();
      CompileGraph(sub_graph, backend_jit_config);
    }
  }

  // Construct the graph compiler info.
  auto graph_compiler_info = ConstructGraphCompilerInfo(func_graph, backend_jit_config);
  MS_LOG(INFO) << "Status record: construct the graph compiler info.";
  MS_EXCEPTION_IF_NULL(graph_compiler_info);
  if ((!graph_compiler_info->graphs_.empty()) || graph_compiler_info->control_nodes_.size() > 1) {
    MS_LOG(DEBUG) << "Start transform";
    PROF_START(GraphScheduler);
    // Transform graph to actor DAG, and schedule the actor DAG.
    ParseControlNodes(*graph_compiler_info, func_graph);
    const auto &actor_set =
      mindspore::ge_backend::runtime::GraphScheduler::GetInstance().Transform(*graph_compiler_info);
    mindspore::ge_backend::runtime::GraphScheduler::GetInstance().Schedule(actor_set);
    PROF_END(GraphScheduler);
  }

  auto cur_graph_id = backend_graph_id_;
  ++backend_graph_id_;
  (void)graph_id_to_graph_compiler_info_.emplace(cur_graph_id, std::move(graph_compiler_info));

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_target = device::GetDeviceTypeByName(context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  device::DeviceContextManager::GetInstance().GetMultiStreamController(device_target)->Refresh();
  MS_LOG(INFO) << "Status record: end compile graph.";

  return cur_graph_id;
}

void GEBackend::CompileGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  // Split graph to segments.
  const std::vector<PrimitivePtr> cut_list = {prim::kPrimReturn,    prim::kPrimPartial,  prim::kPrimSwitch,
                                              prim::kPrimMakeTuple, prim::kPrimBpropCut, prim::kPrimSwitchLayer};
  auto graph_partition = std::make_shared<mindspore::ge_backend::runtime::GraphPartition>(cut_list, "ge");
  MS_EXCEPTION_IF_NULL(graph_partition);
  const auto &segments = graph_partition->Partition(func_graph);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph,
                                  mindspore::ge_backend::runtime::kStageGraphPartition, start_time,
                                  profiler::GetClockSyscnt(), 1);
  MS_LOG(INFO) << "Compile graph: " << func_graph->ToString() << ", Split segments size: " << segments.size();

  // Foreach the segments to compile graph.
  for (const auto &segment : segments) {
    CompileGraphFromSegment(segment, backend_jit_config);
  }
}

void GEBackend::CompileGraphFromSegment(const GraphSegmentPtr &segment, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(segment);
  // Compile the normal nodes, which doesn't contain the cut node.
  if (segment->nodes_.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The segments size is 0.";
  }
  if (!segment->is_cut_) {
    MS_EXCEPTION_IF_NULL(segment->nodes_[0]);
    MS_LOG(INFO) << "Compile normal segment, the first node: " << segment->nodes_[0]->DebugString();

    // Transform nodes to inputs and outputs.
    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = mindspore::ge_backend::runtime::TransformSegmentToAnfGraph(segment->nodes_);

    GraphId graph_id = graph_compiler_->CompileGraph(segment, std::make_pair(inputs, outputs), backend_jit_config);
    auto new_fg = graph_compiler_->Fetch(graph_id);
    MS_EXCEPTION_IF_NULL(new_fg);

    graph_ids_.insert(graph_id);
    if (func_graph_to_kernel_graph_ids_.find(segment->nodes_[0]->func_graph()) ==
        func_graph_to_kernel_graph_ids_.end()) {
      (void)func_graph_to_kernel_graph_ids_[segment->nodes_[0]->func_graph()].emplace_back(
        std::vector<GraphId>{graph_id});
    } else {
      (void)func_graph_to_kernel_graph_ids_[segment->nodes_[0]->func_graph()].back().emplace_back(graph_id);
    }
  } else {
    // Compile the cut node.
    auto cut_node = segment->nodes_[0];
    MS_EXCEPTION_IF_NULL(cut_node);
    MS_LOG(INFO) << "Compile cut segment, the cut node: " << cut_node->DebugString();
    control_nodes_.push_back(cut_node);
    if (common::AnfAlgo::IsCallNode(cut_node) || common::AnfAlgo::CheckPrimitiveType(cut_node, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(cut_node, prim::kPrimSwitchLayer)) {
      const auto &func_graph = cut_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      (void)func_graph_to_kernel_graph_ids_[func_graph].emplace_back(std::vector<GraphId>());
    }
  }
}

std::shared_ptr<mindspore::ge_backend::runtime::GraphCompilerInfo> GEBackend::ConstructGraphCompilerInfo(
  const FuncGraphPtr &root_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);

  std::vector<KernelGraphPtr> graphs;
  std::string name = "kernel_graph";
  size_t graph_index = 0;
  for (const auto &graph_id : graph_ids_) {
    (void)graphs.emplace_back(graph_compiler_->Fetch(graph_id));
    if (graph_index == 0) {
      (void)name.append("_").append(std::to_string(graph_id));
    } else if (graph_index == graph_ids_.size() - 1) {
      (void)name.append("-").append(std::to_string(graph_id));
    }
    ++graph_index;
  }
  auto parser = std::make_shared<mindspore::ge_backend::runtime::ControlNodeParser>();
  const auto &root_output =
    common::AnfAlgo::VisitKernelWithReturnType(root_graph->output(), 0, false, {prim::kPrimTupleGetItem}).first;
  auto outputs_num = common::AnfAlgo::GetAllOutputWithIndex(root_output).size();
  mindspore::ge_backend::runtime::KernelMapPosition outputs_order = FetchOriginOutputOrder(root_graph->output());

  std::vector<std::vector<int64_t> *> tensors_mask;
  std::vector<std::vector<tensor::TensorPtr> *> input_tensors;
  auto strategy = mindspore::ge_backend::runtime::GraphExecutionStrategy::kPipeline;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (mindspore::runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    strategy = mindspore::ge_backend::runtime::GraphExecutionStrategy::kPipelineWithExecutionOrder;
  }

  return std::make_shared<mindspore::ge_backend::runtime::GraphCompilerInfo>(
    graphs, tensors_mask, input_tensors, control_nodes_, root_graph->parameters(), parser, outputs_order, outputs_num,
    root_graph->GetPositionalArgsCount(), name, false, strategy, root_graph->phase(), root_graph, graph_executor_);
}

void GEBackend::ParseControlNodes(const mindspore::ge_backend::runtime::GraphCompilerInfo &graph_compile_info,
                                  const FuncGraphPtr &root_graph) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_EXCEPTION_IF_NULL(graph_compile_info.control_node_parser_);

  mindspore::ge_backend::runtime::FuncGraphToKernelGraphGroup func_graph_to_kernel_graphs;
  for (const auto &func_graph_to_kernel_graph_ids : func_graph_to_kernel_graph_ids_) {
    const auto &func_graph = func_graph_to_kernel_graph_ids.first;
    for (const auto &sub_kernel_graphs_ids : func_graph_to_kernel_graph_ids.second) {
      std::vector<KernelGraphPtr> kernel_graphs;
      for (const auto &graph_id : sub_kernel_graphs_ids) {
        const auto &kernel_graph = graph_compiler_->Fetch(graph_id);
        MS_EXCEPTION_IF_NULL(kernel_graph);
        (void)kernel_graphs.emplace_back(kernel_graph);
      }
      (void)func_graph_to_kernel_graphs[func_graph].emplace_back(kernel_graphs);
    }
  }

  graph_compile_info.control_node_parser_->Parse(control_nodes_, graph_compile_info.graphs_, root_graph,
                                                 func_graph_to_kernel_graphs);
}

// Clear the temp members at the end of graph building.
void GEBackend::ClearGraphBuildMember() {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->ClearGraphBuildMember();

  graph_ids_.clear();
  func_graph_to_kernel_graph_ids_.clear();
  control_nodes_.clear();
}

void GEBackend::RunSubGraph(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
  // Fetch the graph compiler info.
  const auto &graph_iter = graph_id_to_graph_compiler_info_.find(graph_id);
  if (graph_iter == graph_id_to_graph_compiler_info_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Can't find the graph compiler info.";
  }
  MS_EXCEPTION_IF_NULL(graph_iter->second);
  const auto &graph_compiler_info = *(graph_iter->second);

  MS_LOG(INFO) << "Status record: start run actor: " << graph_compiler_info.name_;
  uint64_t start_time = profiler::GetClockSyscnt();
  std::vector<std::vector<tensor::TensorPtr>> input_tensors;
  // Release python gil.
  mindspore::ScopedLongRunning long_running;
  // Run actor DAG.
  const auto &actor_set =
    mindspore::ge_backend::runtime::GraphScheduler::GetInstance().Fetch(graph_compiler_info.name_);
  MS_EXCEPTION_IF_NULL(actor_set);
  mindspore::ge_backend::runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors, inputs);

  {
    uint64_t start_time2 = 0;
    PROFILER_START(start_time2);
    MS_EXCEPTION_IF_NULL(graph_compiler_);
    graph_compiler_->Summary(graph_compiler_info.graphs_);
    ConstructOutputs(actor_set, outputs, graph_compiler_info.root_graph_);
    actor_set->output_actor_->FreeSummaryNodeMem();
    mindspore::ge_backend::runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
    PROFILER_END(start_time2, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kOutputProcess,
                 actor_set->name_, false);
  }
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  (void)profiler::CollectHostInfo(kModelNameRuntime, mindspore::ge_backend::runtime::kEventRunGraph,
                                  mindspore::ge_backend::runtime::kStageRunGraph, start_time,
                                  profiler::GetClockSyscnt(), 1);
  MS_LOG(INFO) << "Status record: end run actor: " << graph_compiler_info.name_;
}

void GEBackend::ConstructOutputs(mindspore::ge_backend::runtime::ActorSet *actor_set, VectorRef *outputs,
                                 const FuncGraphPtr &root_graph) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(root_graph);

  // Update device address for output node of graph.
  // Summary processing will use the output device address, so must be after the summary processing.
  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  actor_set->output_actor_->UpdateOutputDeviceAddress();

  // Fetch outputs.
  auto &output_tensors = actor_set->output_actor_->outputs();
  auto &output_types = actor_set->output_actor_->output_types();
  if (!output_tensors.empty()) {
    size_t output_position = 0;
    std::vector<tensor::TensorPtr> tuple_tensors;
    ConstructOutputs(root_graph->output(), output_tensors, &output_position, outputs, &tuple_tensors, output_types);

    // The tensor may be repeated, so it needs to be set null last.
    for (auto &tuple_tensor : tuple_tensors) {
      MS_EXCEPTION_IF_NULL(tuple_tensor);
      tuple_tensor->set_device_address(nullptr);
    }
  }
}

MS_REGISTER_BACKEND(kGEBackendName, GEBackend);
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
