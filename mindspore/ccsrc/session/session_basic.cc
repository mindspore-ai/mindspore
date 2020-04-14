/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "session/session_basic.h"
#include <utility>
#include <algorithm>
#include <unordered_map>
#include "pipeline/parse/data_converter.h"
#include "ir/manager.h"
#include "operator/ops.h"
#include "common/trans.h"
#include "utils/context/ms_context.h"
#include "utils/config_manager.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/oplib/oplib.h"
#include "pre_activate/common/common_backend_optimization.h"
#include "pre_activate/common/helper.h"
#include "common/utils.h"
#include "ir/dtype.h"

namespace mindspore {
namespace session {
namespace {
const int kSummaryGetItem = 2;
void GetSummaryNodes(const KernelGraph *graph, std::unordered_map<std::string, std::pair<AnfNodePtr, int>> *summary) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(summary);
  summary->clear();
  auto apply_list = TopoSort(graph->get_return());
  for (auto &n : apply_list) {
    MS_EXCEPTION_IF_NULL(n);
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary)) {
      int index = 0;
      auto cnode = n->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().size() <= kSummaryGetItem) {
        MS_LOG(EXCEPTION) << "the node Summary should have 2 inputs at least!";
      }
      auto node = cnode->input(kSummaryGetItem);
      MS_EXCEPTION_IF_NULL(node);
      if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
        auto c = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(c);
        if (c->inputs().size() != kTupleGetItemInputSize) {
          MS_LOG(EXCEPTION) << "the node tuple_get_item must have 2 inputs!";
        }
        MS_EXCEPTION_IF_NULL(c->input(kInputNodeOutputIndexInTupleGetItem));
        auto value_node = c->input(kInputNodeOutputIndexInTupleGetItem)->cast<ValueNodePtr>();
        auto value = value_node->value();
        MS_EXCEPTION_IF_NULL(value);
        Int32ImmPtr int_imm_ptr = value->cast<Int32ImmPtr>();
        MS_EXCEPTION_IF_NULL(int_imm_ptr);
        index = int_imm_ptr->value();
        node = c->input(kRealInputNodeIndexInTupleGetItem);
      }
      std::pair<AnfNodePtr, int> output_pair(node, index);
      // get full name with scope will add scalar or tensor or image summary tag.
      (*summary)[n->fullname_with_scope()] = output_pair;
    }
  }
  MS_LOG(DEBUG) << "Update summary end size: " << (*summary).size();
}

bool ExistSummaryNode(const KernelGraph *graph) {
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = DeepLinkedGraphSearch(ret);
  for (auto &n : all_nodes) {
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary)) {
      return true;
    }
  }
  return false;
}

BaseRef CreateOneTensor(const AnfNodePtr &node, size_t output_index, const KernelGraph &graph,
                        const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "create tensor for output[" << node->DebugString() << "] index[" << output_index << "]";
  // if node is a value node, no need sync addr from device to host
  if (!AnfAlgo::OutputAddrExist(node, output_index)) {
    if (node->isa<ValueNode>()) {
      auto value_node = node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      return value_node->value();
    }
    if (node->isa<Parameter>()) {
      for (size_t input_idx = 0; input_idx < graph.inputs().size(); input_idx++) {
        if (input_idx > input_tensors.size()) {
          MS_LOG(EXCEPTION) << "input idx:" << input_idx << "out of range:" << input_tensors.size();
        }
        if (graph.inputs()[input_idx] == node) {
          return input_tensors[input_idx];
        }
      }
      MS_LOG(EXCEPTION) << "parameter : " << node->DebugString() << "has no output addr";
    }
  }
  // if proccess reach here,it remarks item_with_index is a real node(Parameter,or executable CNode)
  auto address = AnfAlgo::GetOutputAddr(node, output_index);
  MS_EXCEPTION_IF_NULL(address);
  auto shape = AnfAlgo::GetOutputInferShape(node, output_index);
  TypeId type_id = kNumberTypeFloat32;
  type_id = AnfAlgo::GetOutputInferDataType(node, output_index);
  std::vector<int> temp_shape;
  (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
  // if in paynative mode,data only copyed to host when user want to print data
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->enable_pynative_infer()) {
    tensor->set_device_address(AnfAlgo::GetMutableOutputAddr(node, output_index));
  } else if (!address->SyncDeviceToHost(trans::GetRuntimePaddingShape(node, output_index),
                                        LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                        tensor->data_c(true))) {
    MS_LOG(INFO) << "output sync device to host error!!!";
    tensor->set_dirty(false);
  }
  return tensor;
}

BaseRef CreatTensorForOutput(const AnfNodePtr &anf, const KernelGraph &graph,
                             const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_LOG(INFO) << "create tensor for output[" << anf->DebugString() << "]";
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  // special handle for maketuple
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    auto cnode = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    VectorRef ret;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      auto out = CreatTensorForOutput(cnode->input(i), graph, input_tensors);
      ret.push_back(out);
    }
    return ret;
  }
  // if is graph return nothing ,the function should return a null anylist
  size_t size = AnfAlgo::GetOutputTensorNum(item_with_index.first);
  if (size == 0) {
    return VectorRef();
  }
  return CreateOneTensor(item_with_index.first, item_with_index.second, graph, input_tensors);
}

BaseRef CreatTupleForOutput(const AnfNodePtr &anf, const KernelGraph &graph,
                            const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!AnfAlgo::IsRealKernel(anf)) {
    MS_LOG(EXCEPTION) << "anf[" << anf->DebugString() << "] should be a executable kernel";
  }
  if (anf->isa<ValueNode>()) {
    return CreateOneTensor(anf, 0, graph, input_tensors);
  }
  VectorRef ret;
  if (anf->isa<CNode>() && AnfAlgo::GetCNodeName(anf) != prim::kPrimMakeTuple->name()) {
    for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(anf); ++i) {
      auto out = CreateOneTensor(anf, i, graph, input_tensors);
      ret.emplace_back(out);
    }
  }
  return ret;
}

std::string FindOpInputParameterType(const std::string &op_name, kernel::OpImplyType implyType, size_t index) {
  std::string para_type;
  auto op_info = kernel::OpLib::FindOp(op_name, implyType);
  if (op_info == nullptr) {
    return para_type;
  }
  auto op_inputs_info_vec = op_info->inputs_ptr();
  if (index >= op_inputs_info_vec.size()) {
    return para_type;
  }
  auto op_io_info = op_inputs_info_vec[index];
  MS_EXCEPTION_IF_NULL(op_io_info);
  para_type = op_io_info->param_type();
  return para_type;
}

void RunOpConvertConstInputToAttr(const OpRunInfo &op_run_info, const std::shared_ptr<CNode> &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_inputs = op_run_info.op_inputs;
  // get input names vector from attrs
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_names_value = primitive->GetAttr(kAttrInputNames);
  if (input_names_value == nullptr) {
    return;
  }
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names_value);
  // convert const input to attr
  size_t input_num = op_inputs.size();
  if (input_num != input_names_vec.size()) {
    MS_LOG(EXCEPTION) << "input name number " << input_names_vec.size() << "is not equal to input value number "
                      << input_num;
  }
  for (size_t index = 0; index < input_num; ++index) {
    // skip tensor
    if (py::isinstance<tensor::Tensor>(op_inputs[index])) {
      continue;
    }
    // convert to attr
    auto para_type = FindOpInputParameterType(op_run_info.op_name, kernel::OpImplyType::kTBE, index);
    if (!para_type.empty() && para_type == kAttrDynInput) {
      auto tuple_inputs = py::cast<py::tuple>(op_inputs[index]);
      primitive->set_attr(kAttrDynInputSizes, MakeValue(std::vector<int>{SizeToInt(tuple_inputs.size())}));
      continue;
    }
    ValuePtr value = parse::data_converter::PyDataToValue(op_inputs[index]);
    MS_EXCEPTION_IF_NULL(value);
    auto input_name = input_names_vec[index];
    // set the input node as attr of the cnode, key is name of input node,value is input node's value
    primitive->set_attr(input_name, value);
  }
}

ValueNodePtr CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph) {
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<None>()) {
    return nullptr;
  }
  auto new_value_node = graph->NewValueNode(value_node);
  graph->FrontBackendlMapAdd(anf, new_value_node);
  graph->AddValueNodeToGraph(new_value_node);
  return new_value_node;
}

ParameterPtr CreateNewParameterFromParameter(const AnfNodePtr &anf, bool valid_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "anf[" << anf->DebugString() << "] is not a parameter";
  }
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  ParameterPtr new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
  graph_inputs->push_back(new_parameter);
  valid_inputs->push_back(valid_input);
  return new_parameter;
}

std::vector<AnfNodePtr> CreateParameterFromTuple(const AnfNodePtr &node, bool valid_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> parameters;
  std::vector<AnfNodePtr> pre_graph_out = AnfAlgo::GetAllOutput(node, {prim::kPrimTupleGetItem});
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  auto create_parameter = [&](const AbstractBasePtr &abstract) -> void {
    auto parameter = graph->NewParameter();
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->set_abstract(abstract);
    auto new_parameter = graph->NewParameter(parameter);
    parameters.push_back(new_parameter);
    valid_inputs->push_back(valid_input);
    graph_inputs->push_back(new_parameter);
  };
  for (const auto &out_node : pre_graph_out) {
    MS_EXCEPTION_IF_NULL(out_node);
    auto abstract = out_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    // create multiple parameters if is a tuple output real kernel
    if (abstract->isa<abstract::AbstractTuple>() && !AnfAlgo::CheckPrimitiveType(out_node, prim::kPrimTupleGetItem)) {
      auto tuple_abstract = abstract->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(tuple_abstract);
      MS_LOG(INFO) << "tuple_size [" << tuple_abstract->size() << "]";
      for (size_t output_idx = 0; output_idx < tuple_abstract->size(); output_idx++) {
        create_parameter((*tuple_abstract)[output_idx]);
      }
      continue;
    }
    // create single parameter if is a abstract real kernel
    create_parameter(out_node->abstract());
  }
  return parameters;
}

AnfNodePtr CreateNewParameterFromCNode(const AnfNodePtr &anf, bool valid_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!anf->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a cnode";
  }
  MS_LOG(INFO) << "Create a new parameter from cnode[" << anf->DebugString() << "]";
  auto parameters = CreateParameterFromTuple(anf, valid_input, graph);
  if (parameters.empty()) {
    MS_LOG(EXCEPTION) << "No parameter exist!!";
  }
  if (parameters.size() == 1) {
    return parameters[0];
  }
  std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple)};
  (void)std::copy(parameters.begin(), parameters.end(), std::back_inserter(make_tuple_input));
  auto make_tuple = graph->NewCNode(make_tuple_input);
  MS_EXCEPTION_IF_NULL(make_tuple);
  MS_LOG(INFO) << "New make tuple [" << make_tuple->DebugString() << "] of parameters";
  return make_tuple;
}

bool NeedInsertSwitch() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return (context_ptr->enable_task_sink() && context_ptr->loop_sink_flag() &&
          ConfigManager::GetInstance().iter_num() > 1);
}

size_t LoadCtrlInputTensor(const std::shared_ptr<Context> &context, std::vector<tensor::TensorPtr> *inputs) {
  MS_EXCEPTION_IF_NULL(context);
  if (!NeedInsertSwitch()) {
    (void)context->results_.erase(kInputCtrlTensors);
    return 0;
  }
  MS_LOG(INFO) << "Load kInputCtrlTensors";
  auto inputs_params =
    context->GetResult(kInputCtrlTensors).cast<const std::shared_ptr<std::vector<tensor::TensorPtr>>>();
  MS_EXCEPTION_IF_NULL(inputs_params);
  if (inputs_params->empty()) {
    MS_LOG(EXCEPTION) << "Illegal empty inputs_params";
  }
  auto tensor = (*inputs_params)[0];
  MS_EXCEPTION_IF_NULL(tensor);
  auto *val = static_cast<int32_t *>(tensor->data_c(true));
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  tensor->set_dirty(true);
  // set loop_count to zero
  MS_EXCEPTION_IF_NULL(inputs);
  inputs->push_back(tensor);
  return inputs_params->size();
}

ParameterPtr ConstructRunOpParameter(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor,
                                     bool is_weight) {
  auto param = graph->NewParameter();
  MS_EXCEPTION_IF_NULL(param);
  if (is_weight) {
    py::object obj;
    param->set_default_param(obj);
  }
  // set the kernel info of parameter
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  if (input_tensor->device_address().get() == nullptr) {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    TypeId param_init_data_type = AnfAlgo::IsParameterWeight(param) ? kTypeUnknown : input_tensor->data_type();
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{param_init_data_type});
  } else {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{input_tensor->device_address()->format()});
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{input_tensor->device_address()->type_id()});
  }
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());
  // ftruct abstract of parameter
  auto abstract = std::make_shared<abstract::AbstractTensor>(input_tensor);
  param->set_abstract(abstract);
  return param;
}

void DumpGraphOutput(const Any &any, size_t recurse_level = 0) {
  MS_LOG(INFO) << "graph outputs:";
  const size_t max_deep = 10;
  if (recurse_level > max_deep) {
    MS_LOG(INFO) << "recurse too deep";
    return;
  }
  std::string tab_str;
  for (size_t i = 0; i < recurse_level; i++) {
    tab_str = tab_str.append("  ");
  }
  if (any.is<AnyList>()) {
    (void)tab_str.append("{");
    MS_LOG(INFO) << tab_str;
    auto any_list = any.cast<AnyList>();
    for (auto &it : any_list) {
      DumpGraphOutput(it, recurse_level + 1);
    }
    (void)tab_str.append("}");
    MS_LOG(INFO) << tab_str;
  }
  (void)tab_str.append(any.ToString());
  MS_LOG(INFO) << tab_str;
}
}  // namespace

GraphId SessionBasic::graph_sum_ = 0;

CNodePtr SessionBasic::CreateNewCNode(const CNodePtr &cnode, bool valid_input, KernelGraph *graph,
                                      bool *from_other_graph,
                                      std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(from_other_graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  *from_other_graph = false;
  // get primitive of old node
  auto prim = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  // push attr to inputs[0] of new cnode
  std::vector<AnfNodePtr> cnode_inputs = {std::make_shared<ValueNode>(std::make_shared<Primitive>(*prim))};
  // if has multiple depends,only select first depend as parameter
  for (size_t input_idx = 1; input_idx < cnode->inputs().size(); input_idx++) {
    auto anf = cnode->inputs()[input_idx];
    MS_EXCEPTION_IF_NULL(anf);
    // anf has been created before
    if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
      cnode_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(anf));
      continue;
    } else if (other_graph_cnode->find(anf) != other_graph_cnode->end()) {
      cnode_inputs.push_back((*other_graph_cnode)[anf]);
      continue;
    } else if (anf->isa<ValueNode>() && !IsValueNode<FuncGraph>(anf)) {
      // if input is a value node,
      auto new_value_node = CreateNewValueNode(anf, graph);
      if (new_value_node != nullptr) {
        cnode_inputs.emplace_back(new_value_node);
      }
      continue;
    } else if (anf->isa<Parameter>()) {
      // if anf is a parameter
      auto new_parameter = CreateNewParameterFromParameter(anf, valid_input, graph);
      cnode_inputs.push_back(new_parameter);
      if (GetGraphIdByNode(anf) == kInvalidGraphId) {
        graph->FrontBackendlMapAdd(anf, new_parameter);
      } else {
        (*other_graph_cnode)[anf] = new_parameter;
      }
      continue;
    } else if (anf->isa<CNode>()) {
      *from_other_graph = true;
      // the input node is a cnode from other graph
      auto parameter_from_cnode = CreateNewParameterFromCNode(anf, valid_input, graph);
      cnode_inputs.push_back(parameter_from_cnode);
      (*other_graph_cnode)[anf] = parameter_from_cnode;
      continue;
    }
    MS_LOG(EXCEPTION) << "Unexpected input[" << anf->DebugString() << "]";
  }
  TraceManager::DebugTrace(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNode(cnode_inputs);
  TraceManager::EndTrace();
  return new_cnode;
}

KernelGraphPtr SessionBasic::ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  std::unordered_map<AnfNodePtr, AnfNodePtr> other_graph_cnode;
  auto graph = std::make_shared<KernelGraph>();
  graph->set_graph_id(graph_sum_);
  MS_LOG(INFO) << "Create graph: " << graph_sum_;
  size_t from_other_graph_depend_num = 0;
  for (const auto &node : lst) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    if (!node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " is not CNode";
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // create a new cnode object
    bool from_other_graph = false;
    // only first depend from other graph can create
    bool valid_input = true;
    if (from_other_graph_depend_num != 0 && AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend)) {
      valid_input = false;
    }
    auto new_cnode = CreateNewCNode(cnode, valid_input, graph.get(), &from_other_graph, &other_graph_cnode);
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend) && from_other_graph) {
      from_other_graph_depend_num++;
    }
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode->abstract());
    new_cnode->set_scope(cnode->scope());
    // record map relations between anf from ME and new anf node used in backend
    graph->FrontBackendlMapAdd(node, new_cnode);
  }
  // add a make_tuple at the end of graph as output
  graph->set_output(ConstructOutput(outputs, graph));
  MS_EXCEPTION_IF_NULL(context_);
  FuncGraphManagerPtr manager = context_->manager();
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  graph->SetExecOrderByDefault();
  opt::BackendCommonOptimization(graph);
  graphs_[graph_sum_++] = graph;
  return graph;
}

// run graph steps
void SessionBasic::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                 const std::vector<tensor::TensorPtr> &inputs_const) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  size_t input_ctrl_size = 1;
  MS_EXCEPTION_IF_NULL(context_);
  if (context_->HasResult(kInputCtrlTensors)) {
    input_ctrl_size = LoadCtrlInputTensor(context_, &inputs);
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto input_nodes = kernel_graph->inputs();
  if ((inputs.size() + input_ctrl_size) - 1 != input_nodes.size()) {
    MS_LOG(EXCEPTION) << "tensor input size:" << inputs.size()
                      << " is not equal graph inputs size:" << input_nodes.size()
                      << ", input_ctrl_size:" << input_ctrl_size;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto tensor = inputs[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<Parameter>() && AnfAlgo::OutputAddrExist(input_node, 0)) {
      auto pk_node = input_node->cast<ParameterPtr>();
      auto device_address = AnfAlgo::GetMutableOutputAddr(pk_node, 0);
      bool need_sync = false;
      if (ms_context->enable_pynative_infer()) {
        if (tensor->device_address().get() == nullptr || tensor->device_address() != device_address) {
          need_sync = true;
        }
      } else {
        if (tensor->is_dirty() || !AnfAlgo::IsParameterWeight(pk_node)) {
          need_sync = true;
        } else if (tensor->device_address() != device_address) {
          (void)tensor->data_sync();
          need_sync = true;
        }
      }
      if (need_sync) {
        tensor->set_device_address(device_address);
        MS_EXCEPTION_IF_NULL(device_address);
        if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(pk_node, 0),
                                              LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                              tensor->data_c(false))) {
          MS_LOG(EXCEPTION) << "SyncHostToDevice failed.";
        }
      }
    }
    tensor->set_dirty(false);
  }
}

void SessionBasic::UpdateOutputs(const std::shared_ptr<KernelGraph> &kernel_graph, VectorRef *const outputs,
                                 const std::vector<tensor::TensorPtr> &input_tensors) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_LOG(INFO) << "update output[" << item->DebugString() << "]";
    MS_EXCEPTION_IF_NULL(item);
    if (AnfAlgo::IsTupleOutput(item) && AnfAlgo::IsRealKernel(item)) {
      outputs->emplace_back(CreatTupleForOutput(item, *kernel_graph, input_tensors));
      continue;
    }
    outputs->emplace_back(CreatTensorForOutput(item, *kernel_graph, input_tensors));
  }
}

void SessionBasic::RegisterSummaryCallBackFunc(const CallBackFunc &callback) {
  MS_EXCEPTION_IF_NULL(callback);
  summary_callback_ = callback;
}

void SessionBasic::Reorder(std::vector<CNodePtr> *node_list) {
  MS_EXCEPTION_IF_NULL(node_list);
  std::vector<CNodePtr> all_opt_list;
  std::vector<CNodePtr> non_opt_list;

  for (const auto &node : *node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (kOptOperatorSet.find(AnfAlgo::GetCNodeName(node)) != kOptOperatorSet.end()) {
      all_opt_list.emplace_back(node);
    } else {
      non_opt_list.emplace_back(node);
    }
  }
  node_list->clear();
  (void)std::copy(non_opt_list.begin(), non_opt_list.end(), std::back_inserter(*node_list));
  (void)std::copy(all_opt_list.begin(), all_opt_list.end(), std::back_inserter(*node_list));
}

void SessionBasic::Summary(KernelGraph *graph) {
  if (summary_callback_ == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  bool exist_summary = ExistSummaryNode(graph);
  if (!exist_summary) {
    return;
  }
  std::unordered_map<std::string, std::pair<AnfNodePtr, int>> summary_outputs;
  GetSummaryNodes(graph, &summary_outputs);
  std::map<std::string, tensor::TensorPtr> params_list;
  // fetch outputs apply kernel in session & run callback functions
  for (auto &output_item : summary_outputs) {
    auto node = output_item.second.first;
    size_t index = IntToSize(output_item.second.second);
    auto address = AnfAlgo::GetOutputAddr(node, index);
    auto shape = AnfAlgo::GetOutputInferShape(node, index);
    TypeId type_id = AnfAlgo::GetOutputInferDataType(node, index);
    std::vector<int> temp_shape;
    (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
    tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    MS_EXCEPTION_IF_NULL(address);
    if (!address->SyncDeviceToHost(trans::GetRuntimePaddingShape(node, index), LongToSize(tensor->data().nbytes()),
                                   tensor->data_type(), tensor->data_c(true))) {
      MS_LOG(ERROR) << "Failed to sync output from device to host.";
    }
    tensor->set_dirty(false);
    params_list[output_item.first] = tensor;
  }
  // call callback function here
  summary_callback_(0, params_list);
}

void SessionBasic::ToTensorPtr(const OpRunInfo &op_run_info, std::vector<tensor::TensorPtr> *inputs,
                               std::vector<bool> *tensor_mask) {
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(tensor_mask);
  if (op_run_info.op_inputs.size() != op_run_info.inputs_mask.size()) {
    MS_LOG(EXCEPTION) << "Op input size " << op_run_info.op_inputs.size() << " should be equal to op input mask size "
                      << op_run_info.inputs_mask.size();
  }
  size_t input_num = op_run_info.op_inputs.size();
  // get tensors from op_inputs
  for (size_t i = 0; i < input_num; ++i) {
    tensor::TensorPtr tensor_ptr = nullptr;
    auto param_type = FindOpInputParameterType(op_run_info.op_name, kernel::OpImplyType::kTBE, i);
    if (py::isinstance<tensor::Tensor>(op_run_info.op_inputs[i])) {
      tensor_ptr = py::cast<tensor::TensorPtr>(op_run_info.op_inputs[i]);
    } else if (!param_type.empty() && param_type == kAttrDynInput) {
      auto tuple_inputs = py::cast<py::tuple>(op_run_info.op_inputs[i]);
      for (auto &&tuple_input : tuple_inputs) {
        tensor_ptr = py::cast<tensor::TensorPtr>(tuple_input);
        MS_EXCEPTION_IF_NULL(tensor_ptr);
        inputs->push_back(tensor_ptr);
        tensor_mask->push_back(py::cast<bool>(op_run_info.inputs_mask[i]));
      }
      continue;
    } else if (op_run_info.op_name == kApplyMomentumOpName && py::isinstance<py::float_>(op_run_info.op_inputs[i])) {
      tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::float_>(op_run_info.op_inputs[i]), kFloat32);
    }
    if (tensor_ptr != nullptr) {
      inputs->push_back(tensor_ptr);
      tensor_mask->push_back(py::cast<bool>(op_run_info.inputs_mask[i]));
    }
  }
}

CNodePtr SessionBasic::ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> output_args;
  auto FindEqu = [graph, outputs](const AnfNodePtr &out) -> AnfNodePtr {
    auto backend_anf = graph->GetBackendAnfByFrontAnf(out);
    if (backend_anf != nullptr) {
      return backend_anf;
    }
    for (const auto &output : outputs) {
      MS_LOG(INFO) << "output:" << output->DebugString();
    }
    MS_LOG(EXCEPTION) << "Can't find the node in the equiv map!";
  };
  output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_args),
                       [&](const AnfNodePtr &out) -> AnfNodePtr { return FindEqu(out); });
  return graph->NewCNode(output_args);
}

void SessionBasic::CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph) {
  MS_LOG(INFO) << "Start!";
  std::vector<AnfNodePtr> make_tuple_inputs;
  make_tuple_inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  if (AnfRuntimeAlgorithm::GetOutputTensorNum(cnode) > 1) {
    for (size_t output_index = 0; output_index < AnfRuntimeAlgorithm::GetOutputTensorNum(cnode); output_index++) {
      auto idx = NewValueNode(SizeToInt(output_index));
      MS_EXCEPTION_IF_NULL(idx);
      auto imm = std::make_shared<Int32Imm>(output_index);
      idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
      MS_EXCEPTION_IF_NULL(graph);
      auto getitem = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cnode, idx});
      std::vector<TypeId> types = {AnfAlgo::GetOutputInferDataType(cnode, output_index)};
      std::vector<std::vector<size_t>> shapes = {AnfAlgo::GetOutputInferShape(cnode, output_index)};
      AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
      make_tuple_inputs.push_back(getitem);
    }
  } else {
    make_tuple_inputs.push_back(cnode);
  }
  // create output
  auto g_output = graph->NewCNode(make_tuple_inputs);
  graph->set_output(g_output);
  // set graph manager,which now is only used to get valuenodes and hardware optimizing
  MS_EXCEPTION_IF_NULL(context_);
  FuncGraphManagerPtr manager = context_->manager();
  if (manager != nullptr) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  MS_LOG(INFO) << "Finish!";
}

std::shared_ptr<KernelGraph> SessionBasic::ConstructSingleOpGraph(const OpRunInfo &op_run_info) {
  auto graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  if (op_run_info.op_inputs.size() != op_run_info.inputs_mask.size()) {
    MS_LOG(EXCEPTION) << "op_run_info inputs.size" << op_run_info.op_inputs.size()
                      << " should be equal to parameter_mask.size " << op_run_info.inputs_mask.size();
  }
  // set input[0]
  if (op_run_info.py_primitive == nullptr) {
    inputs.push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>(op_run_info.op_name)));
  } else {
    inputs.push_back(std::make_shared<ValueNode>(op_run_info.py_primitive));
  }
  // set input parameter
  std::vector<tensor::TensorPtr> input_tensors;
  std::vector<bool> tensors_mask;
  ToTensorPtr(op_run_info, &input_tensors, &tensors_mask);
  MS_LOG(INFO) << "Input tensor size" << input_tensors.size();
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto parameter = ConstructRunOpParameter(graph, input_tensors[i], tensors_mask[i]);
    inputs.push_back(parameter);
    graph->MutableInputs()->push_back(parameter);
  }
  // set execution order
  auto cnode = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  // set abstract,which include inferred shapes and types
  cnode->set_abstract(op_run_info.abstract);
  // set const input to attr if value is not a tensor,such as scalar or tuple
  RunOpConvertConstInputToAttr(op_run_info, cnode);
  // set execution order
  std::vector<CNodePtr> exe_order = {cnode};
  graph->set_execution_order(exe_order);
  // set output
  CreateOutputNode(cnode, graph);
  return graph;
}

BaseRef SessionBasic::TransformBaseRefListToTuple(const BaseRef &base_ref) {
  if (utils::isa<VectorRef>(base_ref)) {
    auto ref_list = utils::cast<VectorRef>(base_ref);
    py::tuple output_tensors(ref_list.size());
    for (size_t i = 0; i < ref_list.size(); ++i) {
      auto output = TransformBaseRefListToTuple(ref_list[i]);  // use pyObjectRef
      if (utils::isa<tensor::TensorPtr>(output)) {
        auto tensor_ptr = utils::cast<tensor::TensorPtr>(output);
        MS_EXCEPTION_IF_NULL(tensor_ptr);
        output_tensors[i] = tensor_ptr;
      } else if (utils::isa<PyObjectRef>(output)) {
        py::object obj = utils::cast<PyObjectRef>(output).object_;
        py::tuple tensor_tuple = py::cast<py::tuple>(obj);
        output_tensors[i] = tensor_tuple;
      } else {
        MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
      }
    }
    return output_tensors;  // turn tuple to py::object and store in PyObjectRef
  } else if (utils::isa<tensor::TensorPtr>(base_ref)) {
    return base_ref;
  } else {
    MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
  }
}
}  // namespace session
}  // namespace mindspore
