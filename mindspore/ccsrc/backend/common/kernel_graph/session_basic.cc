/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#include "include/backend/common/kernel_graph/session_basic.h"

#include <algorithm>
#include <set>
#include <queue>
#include <utility>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

#include "primitive/ascend_op_name.h"
#include "primitive/structure_op_name.h"
#include "primitive/framework_op_name.h"
#include "primitive/sequence_ops.h"
#include "ir/manager.h"
#include "ir/map_tensor.h"
#include "ir/tensor_new.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "utils/config_manager.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/common_backend_optimization.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/pass_manager/op_adaptation_info_factory.h"
#include "pynative/utils/base.h"
#include "utils/ms_utils.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "include/utils/utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "utils/file_utils.h"
#include "utils/trace_base.h"
#include "utils/log_adapter.h"
#include "include/utils/parallel_context.h"
#include "include/runtime/hardware_abstract/kernel_base/oplib/oplib.h"
#include "include/backend/common/kernel_graph/session_factory.h"
#ifdef ENABLE_DUMP_IR
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#endif
#include "include/utils/callback.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace session {
MS_REG_SESSION(kSessionBasic, SessionBasic);

namespace {
// Need to discard input tensor properties in heterogeneous scenarios.
// For example, the format of device_address in input_tensor is 5D format,
// and it's invalid for CPU graph parameter.
bool NeedDiscardTensorProperties(device::DeviceType op_device_target,
                                 const device::DeviceAddressPtr &tensor_device_address) {
  if (tensor_device_address == nullptr) {
    return true;
  }

  if (op_device_target == tensor_device_address->GetDeviceType()) {
    return false;
  }
  return true;
}

ParameterPtr ConstructRunOpParameter(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor,
                                     const BackendOpRunInfoPtr &op_run_info, InputType input_type) {
  MS_EXCEPTION_IF_NULL(graph);
  auto param = graph->NewParameter();
  MS_EXCEPTION_IF_NULL(param);
  if (input_type == InputType::kParameter) {
    param->set_default_param(input_tensor);
  }

  // set the kernel info of parameter
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
  if (NeedDiscardTensorProperties(op_run_info->base_op_run_info.device_target, device_address)) {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    TypeId param_init_data_type = common::AnfAlgo::IsParameterWeight(param) ? kTypeUnknown : input_tensor->data_type();
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{param_init_data_type});
  } else {
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{input_tensor->data_type()});
    kernel_build_info_builder->SetOutputsReshapeType({device_address->padding_type()});
    kernel_build_info_builder->SetOutputsFormat(
      std::vector<std::string>{kernel::GetFormatFromEnumToStr(input_tensor->format())});
  }
  if (input_tensor->isa<tensor::MapTensor>()) {
    auto map_tensor = input_tensor->cast<tensor::MapTensorPtr>();
    auto map_tensor_abs = std::make_shared<abstract::AbstractMapTensor>(map_tensor);
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());
    param->set_abstract(map_tensor_abs);
    return param;
  }
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());
  // construct abstract of parameter
  auto type_of_tensor = input_tensor->Dtype();
  std::shared_ptr<abstract::AbstractTensor> abstract;
  // Base_shape_ptr is set in dynamic shape scenario, if nullptr, not dynamic shape
  if (input_tensor->base_shape_ptr() != nullptr) {
    abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, input_tensor->base_shape_ptr());
  } else {
    abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, input_tensor->shape());
  }
  param->set_abstract(abstract);
  return param;
}
}  // namespace

void SessionBasic::RegisterSummaryCallBackFunc() {
  constexpr char kRegisterSummaryCallBackFunc[] = "RegisterSummaryCallBackFunc";
  static auto register_summary_callback_func_callback =
    callback::CommonCallback::GetInstance().GetCallback<void>(kRegisterSummaryCallBackFunc);
  if (register_summary_callback_func_callback) {
    register_summary_callback_func_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get RegisterSummaryCallBackFunc, summary function may not work.";
  }
}

void SessionBasic::RecurseSetSummaryNodesForAllGraphs(KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Recurse set summary nodes for all graphs in graph: " << graph->graph_id() << " start";
  constexpr char kRecurseSetSummaryNodesForAllGraphs[] = "RecurseSetSummaryNodesForAllGraphs";
  static auto recurse_set_summary_nodes_for_all_graphs_callback =
    callback::CommonCallback::GetInstance().GetCallback<void, KernelGraph *>(kRecurseSetSummaryNodesForAllGraphs);
  if (recurse_set_summary_nodes_for_all_graphs_callback) {
    recurse_set_summary_nodes_for_all_graphs_callback(graph);
  } else {
    MS_LOG(WARNING) << "Failed to get RecurseSetSummaryNodesForAllGraphs, summary function may not work.";
  }
}

void SessionBasic::Summary(KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  constexpr char kSummaryTensor[] = "SummaryTensor";
  static auto summary_tensor_callback =
    callback::CommonCallback::GetInstance().GetCallback<void, KernelGraph *>(kSummaryTensor);
  if (summary_tensor_callback) {
    summary_tensor_callback(graph);
  } else {
    MS_LOG(WARNING) << "Failed to get SummaryTensor, summary function may not work.";
  }
}

void SessionBasic::CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph) const {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> make_tuple_inputs;
  (void)make_tuple_inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(*prim::kPrimMakeTuple)));
  MS_EXCEPTION_IF_NULL(graph);
  if (AnfAlgo::GetOutputElementNum(cnode) > 1) {
    for (size_t output_index = 0; output_index < AnfAlgo::GetOutputElementNum(cnode); output_index++) {
      auto idx = NewValueNode(SizeToLong(output_index));
      MS_EXCEPTION_IF_NULL(idx);
      auto imm = std::make_shared<Int64Imm>(output_index);
      idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
      auto getitem = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(*prim::kPrimTupleGetItem)), cnode, idx});
      std::vector<TypeId> types = {common::AnfAlgo::GetOutputInferDataType(cnode, output_index)};
      auto shapes = {common::AnfAlgo::GetOutputInferShape(cnode, output_index)};
      common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
      (void)make_tuple_inputs.emplace_back(getitem);
    }
  } else {
    (void)make_tuple_inputs.emplace_back(cnode);
  }
  // create output
  auto g_output = graph->NewCNode(make_tuple_inputs);
  graph->set_output(g_output);
}

std::shared_ptr<KernelGraph> SessionBasic::ConstructSingleOpGraph(const BackendOpRunInfoPtr &op_run_info,
                                                                  const std::vector<ValuePtr> &input_values,
                                                                  const std::vector<InputType> &input_type) {
  auto graph = NewPynativeKernelGraph();
  std::vector<AnfNodePtr> inputs;
  // set input[0]
  auto op_prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(op_prim);
  // Decoupling of frontend PrimitivePy and backend Primitive
  auto new_prim = std::make_shared<Primitive>(*op_prim);
  if (op_run_info->base_op_run_info.use_dynamic_shape_process) {
    AnfAlgo::SetDynamicAttrToPrim(new_prim);
  }
  (void)inputs.emplace_back(std::make_shared<ValueNode>(new_prim));
  // set input parameter
  if (input_values.size() != input_type.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_values.size() << " should be equal to tensors mask size "
                      << input_type.size();
  }
  for (size_t i = 0; i < input_values.size(); ++i) {
    if (input_type[i] == InputType::kConstant) {
      auto value_node = graph->NewValueNode(input_values[i]);
      (void)inputs.emplace_back(value_node);
      continue;
    }
    auto parameter =
      ConstructRunOpParameter(graph, input_values[i]->cast<tensor::TensorPtr>(), op_run_info, input_type[i]);
    (void)inputs.emplace_back(parameter);
    auto mutable_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(mutable_inputs);
    (void)mutable_inputs->emplace_back(parameter);
  }
  // set execution order
  auto cnode = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  auto is_mutable = common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, cnode);
  if (is_mutable) {
    graph->set_flag(kAttrMutableKernel, true);
  }
  // set abstract,which include inferred shapes and types
  cnode->set_abstract(op_run_info->base_op_run_info.abstract);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(op_run_info->base_op_run_info.has_dynamic_output),
                               cnode);
  if (op_run_info->base_op_run_info.is_mixed_precision_cast) {
    common::AnfAlgo::SetNodeAttr(kAttrPynativeNextOpName, MakeValue(op_run_info->base_op_run_info.next_op_name), cnode);
    common::AnfAlgo::SetNodeAttr(kAttrPynativeNextIndex, MakeValue(op_run_info->base_op_run_info.next_input_index),
                                 cnode);
  }
  // set execution order
  graph->set_execution_order({cnode});
  CreateOutputNode(cnode, graph);
  graph->SetInputNodes();
  auto manager = MakeManager({graph});
  if (manager != nullptr) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    UnifyMindIR(graph);
  }
  graph->UpdateGraphDynamicAttr();
  return graph;
}

void SessionBasic::DumpGraphs(const std::vector<KernelGraphPtr> &graphs) const {
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->CanDump(kIntroductory);

  constexpr char kParse[] = "DumpJsonParserParse";
  static auto parse_callback = callback::CommonCallback::GetInstance().GetCallback<void>(kParse);
  if (parse_callback) {
    parse_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get DumpJsonParserParse, data dump function may not work.";
  }

  constexpr char kE2eDumpEnabled[] = "E2eDumpEnabled";
  constexpr char kAsyncDumpEnabled[] = "AsyncDumpEnabled";

  static auto e2e_dump_enabled_callback = callback::CommonCallback::GetInstance().GetCallback<bool>(kE2eDumpEnabled);
  bool e2e_dump_enabled_flag = false;
  if (e2e_dump_enabled_callback) {
    e2e_dump_enabled_flag = e2e_dump_enabled_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get e2e_dump_enabled, data dump function may not work.";
  }

  static auto async_dump_enabled_callback =
    callback::CommonCallback::GetInstance().GetCallback<bool>(kAsyncDumpEnabled);
  bool async_dump_enabled_flag = false;
  if (async_dump_enabled_callback) {
    async_dump_enabled_flag = async_dump_enabled_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get async_dump_enabled, data dump function may not work.";
  }

  if (!save_graphs && !e2e_dump_enabled_flag && !async_dump_enabled_flag) {
    return;
  }
  for (auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);

    if (graph->memory_managed_by_ge()) {
      continue;
    }

    if (save_graphs) {
      std::string file_name = "graph_build_" + std::to_string(graph->graph_id()) + ".ir";
      DumpIR(file_name, graph, true, kWholeStack);
      DumpIRProto(graph, "vm_build_" + std::to_string(graph->graph_id()));
      DumpIR("trace_code_graph", graph, true, kWholeStack);
    }
    std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    if (device_target != kAscendDevice) {
      // Here dump data only with Ascend.
      continue;
    }
    // If the new runtime is used, get rank_id from context via GetRankID(), else get rank_id from rank_id_.
    uint32_t rank_id = rank_id_;
    rank_id = GetRankId();
    std::string final_graph = "trace_code_graph_" + std::to_string(graph->graph_id());
    if (e2e_dump_enabled_callback()) {
      constexpr char kPath[] = "DumpJsonParserPath";
      static auto path_callback = callback::CommonCallback::GetInstance().GetCallback<std::string>(kPath);
      if (!path_callback) {
        MS_LOG(WARNING) << "Failed to get json_parser.path(), data dump function may not work.";
        return;
      }
      std::string root_dir = path_callback() + "/rank_" + std::to_string(rank_id);

      MS_LOG(INFO) << "Dump graph and exeorder for graph: " << graph->graph_id()
                   << ", root_graph_id: " << graph->root_graph_id() << ", rank_id: " << rank_id;
      std::string target_dir = root_dir + "/graphs";

      constexpr char kGenerateDumpPath[] = "GenerateDumpPath";
      static auto generate_dump_path_callback =
        callback::CommonCallback::GetInstance().GetCallback<std::string, uint32_t, uint32_t, bool>(kGenerateDumpPath);
      if (!generate_dump_path_callback) {
        MS_LOG(WARNING) << "Failed to get GenerateDumpPath, data dump function may not work.";
        return;
      }
      std::string cst_file_dir = generate_dump_path_callback(graph->root_graph_id(), rank_id, true);
      std::string ir_file_path = target_dir + "/" + "ms_output_" + final_graph + ".ir";

      constexpr char kDumpIRProtoWithSrcInfoDebugWholeStack[] = "DumpIRProtoWithSrcInfoDebugWholeStack";
      static auto dump_ir_proto_with_src_info_debug_whole_stack_callback =
        callback::CommonCallback::GetInstance()
          .GetCallback<void, const FuncGraphPtr &, const std::string &, const std::string &>(
            kDumpIRProtoWithSrcInfoDebugWholeStack);
      if (dump_ir_proto_with_src_info_debug_whole_stack_callback) {
        dump_ir_proto_with_src_info_debug_whole_stack_callback(graph, final_graph, target_dir);
      } else {
        MS_LOG(WARNING) << "Failed to get DumpIRProtoWithSrcInfoDebugWholeStack, data dump function may not work.";
      }
      DumpIR("trace_code_graph", graph, true, kWholeStack, ir_file_path);
      DumpGraphExeOrder("ms_execution_order_graph_" + std::to_string(graph->graph_id()) + ".csv", root_dir,
                        graph->execution_order());
    }
  }
#endif
}
}  // namespace session
void DumpGraphExeOrder(const std::string &file_name, const std::string &target_dir,
                       const std::vector<CNodePtr> &execution_order) {
  std::string file_path = target_dir + "/execution_order/" + file_name;
  auto realpath = Common::CreatePrefixPath(file_path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Failed to get real path: [" << file_path << "] in dump graph execution order.";
    return;
  }
  file_path = realpath.value();

  ChangeFileMode(file_path, S_IWUSR);
  // write to csv file
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Failed to open file [" << file_path
                  << "] in dump graph execution order, please check the file access permission and whether disk space "
                     "is available.";
    return;
  }
  ofs << "NodeExecutionOrder-FullNameWithScope\n";
  for (const CNodePtr &node : execution_order) {
    ofs << node->fullname_with_scope() << "\n";
  }
  ofs.close();
  // set file mode to read only by user
  ChangeFileMode(file_path, S_IRUSR);
}

uint32_t GetRankId() {
  uint32_t rank_id = 0;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  std::string world_group;
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == kAscendDevice) {
    world_group = kHcclWorldGroup;
  } else if (backend == kGPUDevice) {
    world_group = kNcclWorldGroup;
  } else {
    MS_LOG(ERROR) << "Invalid backend: " << backend;
    return rank_id;
  }
  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
      MS_LOG(INFO) << "Failed to get rank id.";
    }
  }
  return rank_id;
}
}  // namespace mindspore
