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
#include "backend/session/session_basic.h"
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "pipeline/jit/parse/data_converter.h"
#include "ir/manager.h"
#include "ir/param_value.h"
#include "backend/kernel_compiler/common_utils.h"
#include "frontend/operator/ops.h"
#include "common/trans.h"
#include "utils/context/ms_context.h"
#include "utils/config_manager.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "backend/optimizer/common/common_backend_optimization.h"
#include "backend/optimizer/pass/const_input_to_attr_registry.h"
#include "backend/optimizer/common/helper.h"
#include "common/utils.h"
#include "ir/dtype.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#if (!_WIN32 && !ENABLE_GE && !ENABLE_TESTCASES)
#include "frontend/parallel/ps/worker.h"
#include "frontend/parallel/ps/common.h"
#include "frontend/parallel/ps/util.h"
#endif

namespace mindspore {
namespace session {
static std::shared_ptr<std::map<ParamValuePtr, ParameterPtr>> python_paras;
void ClearPythonParasMap() { python_paras = nullptr; }
namespace {
const int kSummaryGetItem = 2;

ParamValuePtr GetParamDefaultValue(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto parameter = node->cast<ParameterPtr>();
  if (parameter == nullptr || !parameter->has_default()) {
    return nullptr;
  }
  return parameter->default_param();
}

BaseRef CreateOneTensor(const AnfNodePtr &node, size_t output_index, const KernelGraph &graph,
                        const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "Create tensor for output[" << node->DebugString() << "] index[" << output_index << "]";
  // if node is a value node, no need sync addr from device to host
  if (!AnfAlgo::OutputAddrExist(node, output_index)) {
    if (node->isa<ValueNode>()) {
      auto value_node = node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      return value_node->value();
    }
    if (node->isa<Parameter>()) {
      for (size_t input_idx = 0; input_idx < graph.inputs().size(); input_idx++) {
        if (input_idx >= input_tensors.size()) {
          MS_LOG(EXCEPTION) << "Input idx:" << input_idx << "out of range:" << input_tensors.size();
        }
        if (graph.inputs()[input_idx] == node) {
          return input_tensors[input_idx];
        }
      }
      MS_LOG(EXCEPTION) << "Parameter : " << node->DebugString() << " has no output addr";
    }
  }
  // if proccess reach here,it remarks item_with_index is a real node(Parameter,or executable CNode)
  auto address = AnfAlgo::GetMutableOutputAddr(node, output_index);
  MS_EXCEPTION_IF_NULL(address);
  auto shape = AnfAlgo::GetOutputInferShape(node, output_index);
  TypeId type_id = kNumberTypeFloat32;
  type_id = AnfAlgo::GetOutputInferDataType(node, output_index);
  std::vector<int> temp_shape;
  if (graph.IsInternalOutput(node)) {
    temp_shape.emplace_back(1);
    tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    tensor->set_device_address(address);
    tensor->set_dirty(false);
    return tensor;
  }
  (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
  // if in paynative mode,data only copyed to host when user want to print data
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->execution_mode() == kPynativeMode || ms_context->device_target() == kGPUDevice) {
    tensor->set_device_address(address);
    tensor->set_dirty(false);
  } else if (!address->SyncDeviceToHost(trans::GetRuntimePaddingShape(node, output_index),
                                        LongToSize(tensor->data().nbytes()), tensor->data_type(), tensor->data_c())) {
    MS_LOG(INFO) << "Output sync device to host error!!!";
    tensor->set_dirty(false);
  }
  return tensor;
}

BaseRef CreateTensorForOutput(const AnfNodePtr &anf, const KernelGraph &graph,
                              const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_LOG(INFO) << "Create tensor for output[" << anf->DebugString() << "]";
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  MS_LOG(INFO) << "Create tensor for output after visit:" << item_with_index.first->DebugString();
  // special handle for maketuple
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    auto cnode = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    VectorRef ret;
    for (size_t i = 1; i < cnode->inputs().size(); ++i) {
      auto out = CreateTensorForOutput(cnode->input(i), graph, input_tensors);
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

ValueNodePtr CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
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

size_t LoadCtrlInputTensor(const std::shared_ptr<KernelGraph> &graph, std::vector<tensor::TensorPtr> *inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Load kInputCtrlTensors";
  auto inputs_params = graph->input_ctrl_tensors();
  if (inputs_params == nullptr) {
    return 0;
  }
  if (inputs_params->size() < 2) {
    MS_LOG(EXCEPTION) << "Illegal inputs_params size";
  }
  auto tensor = (*inputs_params)[0];
  MS_EXCEPTION_IF_NULL(tensor);
  auto *val = static_cast<int32_t *>(tensor->data_c());
  MS_EXCEPTION_IF_NULL(val);
  *val = 0;
  tensor->set_dirty(true);
  // set loop_count to zero
  MS_EXCEPTION_IF_NULL(inputs);
  inputs->push_back(tensor);

  auto epoch_tensor = (*inputs_params)[1];
  MS_EXCEPTION_IF_NULL(epoch_tensor);
  auto *epoch_val = static_cast<int32_t *>(epoch_tensor->data_c());
  MS_EXCEPTION_IF_NULL(epoch_val);
  *epoch_val = graph->current_epoch();
  epoch_tensor->set_dirty(true);
  inputs->push_back(epoch_tensor);
  MS_LOG(INFO) << "Load epoch_val:" << *epoch_val;

  graph->set_current_epoch(graph->current_epoch() + 1);

  return inputs_params->size();
}

ValueNodePtr ConstructRunOpValueNode(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto value_node = std::make_shared<ValueNode>(input_tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  // construct abstract of value node
  auto type_of_tensor = input_tensor->Dtype();
  auto shape_of_tensor = input_tensor->shape();
  auto abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, shape_of_tensor);
  value_node->set_abstract(abstract);
  // add value node to graph
  auto input_value_node = graph->NewValueNode(value_node);
  graph->AddValueNodeToGraph(input_value_node);
  return input_value_node;
}

ParameterPtr ConstructRunOpParameter(const std::shared_ptr<KernelGraph> &graph, const tensor::TensorPtr &input_tensor,
                                     int tensor_mask) {
  MS_EXCEPTION_IF_NULL(graph);
  auto param = graph->NewParameter();
  MS_EXCEPTION_IF_NULL(param);
  if (tensor_mask == kParameterWeightTensorMask) {
    auto param_value_new = std::make_shared<ParamValue>();
    param->set_default_param(param_value_new);
  }
  // set the kernel info of parameter
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
  if (device_address == nullptr) {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    TypeId param_init_data_type = AnfAlgo::IsParameterWeight(param) ? kTypeUnknown : input_tensor->data_type();
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{param_init_data_type});
  } else {
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{device_address->format()});
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{device_address->type_id()});
  }
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), param.get());
  // construct abstract of parameter
  auto type_of_tensor = input_tensor->Dtype();
  auto shape_of_tensor = input_tensor->shape();
  auto abstract = std::make_shared<abstract::AbstractTensor>(type_of_tensor, shape_of_tensor);
  param->set_abstract(abstract);
  return param;
}

void DumpGraphOutput(const Any &any, size_t recurse_level = 0) {
  MS_LOG(INFO) << "Graph outputs:";
  const size_t max_deep = 10;
  if (recurse_level > max_deep) {
    MS_LOG(INFO) << "Recurse too deep";
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

bool ExistSummaryNode(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = DeepLinkedGraphSearch(ret);
  for (auto &n : all_nodes) {
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary) || IsPrimitiveCNode(n, prim::kPrimHistogramSummary)) {
      return true;
    }
  }
  return false;
}
}  // namespace

GraphId SessionBasic::graph_sum_ = 0;

KernelGraphPtr SessionBasic::GetGraph(mindspore::GraphId graph_id) {
  auto it = graphs_.find(graph_id);
  if (it == graphs_.end()) {
    MS_LOG(WARNING) << "Can't find graph " << graph_id;
    return nullptr;
  }
  return it->second;
}

void SessionBasic::InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter) {
  auto graph_id = GetGraphIdByNode(out_node);
  if (graph_id == kInvalidGraphId) {
    return;
  }
  auto node_graph = GetGraph(graph_id);
  if (node_graph == nullptr) {
    return;
  }
  MS_LOG(INFO) << "Init parameter with pre graph output node: " << out_node->DebugString();
  auto ref_node = node_graph->GetInternalOutputByFrontNode(out_node);
  if (ref_node == nullptr) {
    MS_LOG(INFO) << "No corresponding internal output for output node";
    return;
  }
  size_t output_idx = 0;
  if (AnfAlgo::CheckPrimitiveType(out_node, prim::kPrimTupleGetItem)) {
    output_idx = AnfAlgo::GetTupleGetItemOutIndex(out_node->cast<CNodePtr>());
  }
  auto real_kernel = AnfAlgo::VisitKernel(ref_node, output_idx);
  auto ref_real_node = real_kernel.first;
  auto ref_real_node_index = real_kernel.second;
  if (ref_real_node->isa<CNode>() && node_graph->IsInternalOutput(ref_real_node) &&
      node_graph->IsFinalOutputKernel(ref_real_node)) {
    auto kernel_info = ref_real_node->kernel_info();
    if (kernel_info == nullptr || !kernel_info->has_build_info()) {
      MS_LOG(INFO) << "No kernel info";
      return;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(ref_real_node, ref_real_node_index);
    if (address == nullptr) {
      MS_LOG(INFO) << "No kernel address";
      return;
    }
    auto format = AnfAlgo::GetOutputFormat(ref_real_node, ref_real_node_index);
    auto type = AnfAlgo::GetOutputDeviceDataType(ref_real_node, ref_real_node_index);
    auto d_kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(d_kernel_info);
    parameter->set_kernel_info(d_kernel_info);
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsDeviceType({type});
    builder.SetOutputsFormat({format});
    d_kernel_info->set_select_kernel_build_info(builder.Build());
    AnfAlgo::SetOutputAddr(address, 0, parameter.get());
    AnfAlgo::SetOutputInferTypeAndShape({type}, {AnfAlgo::GetOutputInferShape(parameter, 0)}, parameter.get());
  }
}

std::vector<AnfNodePtr> SessionBasic::CreateParameterFromTuple(const AnfNodePtr &node, bool valid_input,
                                                               KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> parameters;
  std::vector<AnfNodePtr> pre_graph_out = {node};
  // If a cnode is a call, it's input0 is a cnode too, so it doesn't have primitive
  if (!AnfAlgo::IsRealKernel(node)) {
    pre_graph_out = AnfAlgo::GetAllOutput(node, {prim::kPrimTupleGetItem});
  }
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
      MS_LOG(INFO) << "Tuple_size [" << tuple_abstract->size() << "]";
      for (size_t output_idx = 0; output_idx < tuple_abstract->size(); output_idx++) {
        create_parameter((*tuple_abstract)[output_idx]);
      }
      continue;
    }
    // create single parameter if is a abstract real kernel
    create_parameter(out_node->abstract());
    InitInternalOutputParameter(out_node, parameters[parameters.size() - 1]);
  }
  return parameters;
}

ParameterPtr SessionBasic::CreateNewParameterFromParameter(const AnfNodePtr &anf, bool valid_input,
                                                           KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto param_value = GetParamDefaultValue(anf);
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  ParameterPtr new_parameter = nullptr;
  // if parameter's python parameter has been exist a backend parameter, reuse the exist parameter
  if (python_paras == nullptr) {
    python_paras = std::make_shared<std::map<ParamValuePtr, ParameterPtr>>();
  }
  auto iter = python_paras->find(param_value);
  if (iter != python_paras->end()) {
    new_parameter = iter->second;
  } else {
    TraceManager::DebugTrace(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
    if (param_value != nullptr) {
      (*python_paras)[param_value] = new_parameter;
    }
    TraceManager::EndTrace();
  }
  graph_inputs->push_back(new_parameter);
  valid_inputs->push_back(valid_input);
  return new_parameter;
}

AnfNodePtr SessionBasic::CreateNewParameterFromCNode(const AnfNodePtr &anf, bool valid_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
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

CNodePtr SessionBasic::CreateNewCNode(const CNodePtr &cnode, bool valid_input, KernelGraph *graph,
                                      bool *from_other_graph,
                                      std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(from_other_graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  *from_other_graph = false;
  // get primitive of old node
  std::vector<AnfNodePtr> cnode_inputs;
  auto prim = AnfAlgo::GetCNodePrimitive(cnode);
  if (prim != nullptr) {
    // push attr to inputs[0] of new cnode
    cnode_inputs.push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>(*prim)));
  } else {
    auto fg = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs.push_back(std::make_shared<ValueNode>(new_fg));
  }
  auto origin_inputs = cnode->inputs();
  bool optimize_depend = false;
  if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && origin_inputs.size() == 3 &&
      origin_inputs[kRealInputIndexInDepend]->isa<ValueNode>()) {
    optimize_depend = true;
  }
  // if has multiple depends,only select first depend as parameter
  for (size_t input_idx = 1; input_idx < origin_inputs.size(); input_idx++) {
    auto anf = origin_inputs[input_idx];
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
    } else if (anf->isa<Parameter>() && AnfAlgo::GetOutputTensorNum(anf) == 1) {
      auto new_parameter = CreateNewParameterFromParameter(anf, valid_input, graph);
      cnode_inputs.push_back(new_parameter);
      if (GetGraphIdByNode(anf) == kInvalidGraphId) {
        graph->FrontBackendlMapAdd(anf, new_parameter);
      } else {
        (*other_graph_cnode)[anf] = new_parameter;
      }
      continue;
    } else if (optimize_depend && input_idx == kDependAttachNodeIndex) {
      cnode_inputs.push_back(origin_inputs[kRealInputIndexInDepend]);
      continue;
    } else {
      *from_other_graph = true;
      // the input node is a cnode from other graph
      auto parameter_from_cnode = CreateNewParameterFromCNode(anf, valid_input, graph);
      cnode_inputs.push_back(parameter_from_cnode);
      (*other_graph_cnode)[anf] = parameter_from_cnode;
    }
  }
  TraceManager::DebugTrace(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNode(cnode_inputs);
  TraceManager::EndTrace();
  return new_cnode;
}

CNodePtr SessionBasic::CreateSwitchInput(const AnfNodePtr &node_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node_input);
  MS_EXCEPTION_IF_NULL(graph);
  // switch input generalizes partial
  if (AnfAlgo::CheckPrimitiveType(node_input, prim::kPrimPartial) ||
      AnfAlgo::CheckPrimitiveType(node_input, prim::kPrimCall)) {
    return node_input->cast<CNodePtr>();
  }
  if (node_input->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "If switch input is " << node_input->DebugString() << ", it mast be partial or call.";
  }
  std::vector<AnfNodePtr> partial_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))};
  if (node_input->isa<ValueNode>() && IsValueNode<FuncGraph>(node_input)) {
    partial_inputs.emplace_back(node_input);
    auto partial_node = graph->NewCNode(partial_inputs);
    return partial_node;
  }
  KernelGraphPtr kernel_graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->set_output(graph->GetBackendAnfByFrontAnf(node_input));
  partial_inputs.emplace_back(std::make_shared<ValueNode>(kernel_graph));
  auto partial_node = graph->NewCNode(partial_inputs);
  return partial_node;
}

CNodePtr SessionBasic::HandleSwitchInputs(const AnfNodePtr &anf_node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(graph);
  auto node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  if (node->inputs().size() < kSwitchInputSize) {
    MS_LOG(EXCEPTION) << "Switch input size less than " << kSwitchInputSize;
  }
  auto primitive = NewValueNode(std::make_shared<Primitive>(prim::kPrimSwitch->name()));
  std::vector<AnfNodePtr> switch_inputs = {primitive, node->input(1)};
  for (size_t index = 2; index < node->inputs().size(); index++) {
    auto input = CreateSwitchInput(node->input(index), graph);
    switch_inputs.emplace_back(input);
  }
  auto switch_node = graph->NewCNode(switch_inputs);
  return switch_node;
}

std::vector<AnfNodePtr> SessionBasic::CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  // create primitive of cnode:call(partial or switch)
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  if (cnode_input == nullptr) {
    MS_LOG(EXCEPTION) << "CNode input[0] is CNode:" << attr_input->DebugString()
                      << ", but input[0] has not been created.";
  }
  // if the node is partial, insert the inputs of partial to the call
  if (AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimPartial)) {
    auto partial_node = attr_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_node);
    auto partial_inputs = partial_node->inputs();
    std::transform(partial_inputs.begin() + kFirstDataInputIndex, partial_inputs.end(),
                   std::back_inserter(cnode_inputs), [&graph](const AnfNodePtr &node) {
                     MS_EXCEPTION_IF_NULL(graph->GetBackendAnfByFrontAnf(node));
                     return graph->GetBackendAnfByFrontAnf(node);
                   });
    return cnode_inputs;
  } else if (AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitch)) {
    auto switch_node = HandleSwitchInputs(cnode_input, graph);
    cnode_inputs.emplace_back(switch_node);
    return cnode_inputs;
  }
  MS_LOG(EXCEPTION) << "CNode input[0] must be partial or switch.";
}

CNodePtr SessionBasic::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (AnfAlgo::IsGraphKernel(cnode)) {
    auto fg = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs.push_back(std::make_shared<ValueNode>(new_fg));
  } else if (IsValueNode<FuncGraph>(attr_input)) {
    // create primitive of cnode:call
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
    // create a ValueNode<KernelGraph> as input of cnode:call
    if (graph->GetBackendAnfByFrontAnf(attr_input) != nullptr) {
      cnode_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(attr_input));
    } else {
      auto new_value_node = CreateValueNodeKernelGraph(attr_input, graph);
      if (new_value_node != nullptr) {
        cnode_inputs.emplace_back(new_value_node);
      }
    }
  } else if (attr_input->isa<CNode>()) {
    cnode_inputs = CreateSwitchOrPartialNode(cnode, graph);
  } else {
    // get primitive of old node
    auto prim = AnfAlgo::GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    // push attr to inputs[0] of new cnode
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(*prim)))};
  }

  for (size_t input_idx = 1; input_idx < cnode->inputs().size(); input_idx++) {
    auto anf = cnode->input(input_idx);
    MS_EXCEPTION_IF_NULL(anf);
    // anf has been created before
    if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
      cnode_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(anf));
      continue;
    } else if (IsValueNode<None>(anf)) {
      continue;
    }
    MS_LOG(EXCEPTION) << "Unexpected input[" << anf->DebugString() << "]";
  }
  TraceManager::DebugTrace(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNode(cnode_inputs);
  TraceManager::EndTrace();
  return new_cnode;
}

ValueNodePtr SessionBasic::CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto sub_func_graph = AnfAlgo::GetValueNodeFuncGraph(anf);
  MS_EXCEPTION_IF_NULL(sub_func_graph);
  if (front_backend_graph_map_.find(sub_func_graph) == front_backend_graph_map_.end()) {
    MS_LOG(EXCEPTION) << "FuncGraph: " << sub_func_graph->ToString() << " has not been transformed to KernelGraph.";
  }
  auto sub_kernel_graph = front_backend_graph_map_[sub_func_graph];

  ValueNodePtr new_value_node = std::make_shared<ValueNode>(sub_kernel_graph);
  new_value_node->set_abstract(value_node->abstract());
  // create new kernel_info of new value_node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  kernel_info->SetFeatureMapFlag(false);
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  AnfAlgo::SetGraphId(graph->graph_id(), new_value_node.get());

  graph->FrontBackendlMapAdd(anf, new_value_node);

  return new_value_node;
}

ParameterPtr SessionBasic::CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }

  auto param_value = GetParamDefaultValue(anf);
  ParameterPtr new_parameter = nullptr;
  if (python_paras == nullptr) {
    python_paras = std::make_shared<std::map<ParamValuePtr, ParameterPtr>>();
  }
  auto iter = python_paras->find(param_value);
  if (iter != python_paras->end()) {
    new_parameter = iter->second;
  } else {
    TraceManager::DebugTrace(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
    if (param_value != nullptr) {
      (*python_paras)[param_value] = new_parameter;
    }
    TraceManager::EndTrace();
  }

  return new_parameter;
}

KernelGraphPtr SessionBasic::ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  std::unordered_map<AnfNodePtr, AnfNodePtr> other_graph_cnode;
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
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
  FuncGraphManagerPtr manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  graph->SetExecOrderByDefault();
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
  opt::BackendCommonOptimization(graph);
  return graph;
}

void SessionBasic::CreateCNodeKernelGraph(const AnfNodePtr node, KernelGraphPtr graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // create a new cnode object
  auto new_cnode = CreateNewCNode(cnode, graph.get());
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  new_cnode->set_scope(cnode->scope());
  graph->FrontBackendlMapAdd(node, new_cnode);
  if (AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimReturn)) {
    graph->set_return(new_cnode);
  }
}
std::shared_ptr<KernelGraph> SessionBasic::ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                                std::vector<KernelGraphPtr> *all_out_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(all_out_graph);
  auto node_list = TopoSort(func_graph->get_return());
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  front_backend_graph_map_[func_graph] = graph;
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();

  bool is_trace_back = false;
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    if (node->isa<Parameter>()) {
      auto graph_inputs = graph->MutableInputs();
      MS_EXCEPTION_IF_NULL(graph_inputs);
      auto new_parameter = CreateNewParameter(node, graph.get());
      graph_inputs->push_back(new_parameter);
      graph->FrontBackendlMapAdd(node, new_parameter);
      continue;
    } else if (node->isa<ValueNode>()) {
      if (!IsValueNode<FuncGraph>(node)) {
        // if input is a common value node,
        (void)CreateNewValueNode(node, graph.get());
      } else {
        // if input is a ValueNode<FuncGraph>
        FuncGraphPtr child_graph = AnfAlgo::GetValueNodeFuncGraph(node);
        if (front_backend_graph_map_.find(child_graph) != front_backend_graph_map_.end()) {
          is_trace_back = true;
        } else {
          (void)ConstructKernelGraph(child_graph, all_out_graph);
        }
        (void)CreateValueNodeKernelGraph(node, graph.get());
      }
      continue;
    } else {
      CreateCNodeKernelGraph(node, graph);
    }
  }
  // if a graph jump back unconditionally, return op of this graph will never be executed, so output is null.
  graph->set_output_null(is_trace_back);
  AddParameterToGraphInputs(func_graph->parameters(), graph.get());
  graph->SetExecOrderByDefault();
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
  all_out_graph->push_back(graph);
  return graph;
}

void SessionBasic::AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  graph_inputs->clear();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter);
    auto backend_parameter = graph->GetBackendAnfByFrontAnf(parameter);
    if (backend_parameter == nullptr) {
      // for example "def f(x,y,z) {return x + y}", parameter z in unused
      auto new_parameter = CreateNewParameter(parameter, graph);
      graph_inputs->push_back(new_parameter);
      MS_LOG(INFO) << "Can't find parameter:" << parameter->DebugString();
      continue;
    }
    MS_LOG(INFO) << "Graph[" << graph->graph_id() << "],parameter:" << parameter->DebugString();
    graph_inputs->push_back(backend_parameter);
  }
}

// run graph steps
void SessionBasic::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                                 const std::vector<tensor::TensorPtr> &inputs_const) const {
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  size_t input_ctrl_size = 2;
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->input_ctrl_tensors()) {
    input_ctrl_size = LoadCtrlInputTensor(kernel_graph, &inputs);
  }
  auto input_nodes = kernel_graph->inputs();
  if ((inputs.size() + input_ctrl_size) - 2 != input_nodes.size()) {
    MS_LOG(EXCEPTION) << "Tensor input:" << inputs.size() << " is not equal graph inputs:" << input_nodes.size()
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
        if (tensor->is_dirty()) {
          need_sync = true;
        } else if (tensor->device_address() != device_address) {
          (void)tensor->data_sync();
          need_sync = true;
        }
      }
      if (need_sync) {
        if (ms_context->execution_mode() == kPynativeMode || AnfAlgo::IsParameterWeight(pk_node)) {
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
    tensor->set_dirty(false);
  }
}

void SessionBasic::UpdateOutputs(const std::shared_ptr<KernelGraph> &kernel_graph, VectorRef *const outputs,
                                 const std::vector<tensor::TensorPtr> &input_tensors) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  auto anf_outputs = kernel_graph->outputs();
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Update output[" << item->DebugString() << "]";
    outputs->emplace_back(CreateTensorForOutput(item, *kernel_graph, input_tensors));
  }
}

void SessionBasic::RegisterSummaryCallBackFunc(const CallBackFunc &callback) {
  MS_EXCEPTION_IF_NULL(callback);
  summary_callback_ = callback;
}

void SessionBasic::Reorder(std::vector<CNodePtr> *node_list) { AnfAlgo::ReorderExecList(NOT_NULL(node_list)); }

void SessionBasic::GetSummaryNodes(KernelGraph *graph) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->summary_node_exist()) {
    return;
  }
  auto summary = graph->summary_nodes();
  auto apply_list = TopoSort(graph->get_return());
  for (auto &n : apply_list) {
    MS_EXCEPTION_IF_NULL(n);
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary) || IsPrimitiveCNode(n, prim::kPrimHistogramSummary)) {
      auto cnode = n->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().size() <= kSummaryGetItem) {
        MS_LOG(EXCEPTION) << "The node Summary should have 2 inputs at least!";
      }
      auto node = cnode->input(kSummaryGetItem);
      MS_EXCEPTION_IF_NULL(node);
      auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
      MS_EXCEPTION_IF_NULL(item_with_index.first);
      if (!AnfAlgo::IsRealKernel(item_with_index.first)) {
        MS_LOG(EXCEPTION) << "Unexpected node:" << item_with_index.first->DebugString();
      }
      summary[n->fullname_with_scope()] = item_with_index;
    }
  }
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
}

void SessionBasic::Summary(KernelGraph *graph) {
  if (summary_callback_ == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  bool exist_summary = graph->summary_node_exist();
  if (!exist_summary) {
    return;
  }
  GetSummaryNodes(graph);
  auto summary_outputs = graph->summary_nodes();
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
    if (!address->GetPtr()) {
      continue;
    }
    if (!address->SyncDeviceToHost(trans::GetRuntimePaddingShape(node, index), LongToSize(tensor->data().nbytes()),
                                   tensor->data_type(), tensor->data_c())) {
      MS_LOG(ERROR) << "Failed to sync output from device to host.";
    }
    tensor->set_dirty(false);
    params_list[output_item.first] = tensor;
  }
  // call callback function here
  summary_callback_(0, params_list);
}

CNodePtr SessionBasic::ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> output_args;
  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    MS_LOG(INFO) << "Output:" << output->DebugString();
  }
  auto FindEqu = [graph, outputs](const AnfNodePtr &out) -> AnfNodePtr {
    auto backend_anf = graph->GetBackendAnfByFrontAnf(out);
    if (backend_anf != nullptr) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->execution_mode() == kPynativeMode) {
        return backend_anf;
      }
      auto front_real_kernel = AnfAlgo::VisitKernel(out, 0);
      auto backend_real_kernel = AnfAlgo::VisitKernel(backend_anf, 0);
      MS_EXCEPTION_IF_NULL(out);
      auto out_func_graph = out->func_graph();
      MS_EXCEPTION_IF_NULL(out_func_graph);
      auto out_func_graph_manager = out_func_graph->manager();
      if (out_func_graph_manager == nullptr) {
        return backend_anf;
      }
      auto node_users = out_func_graph_manager->node_users();
      auto users = node_users[out];
      bool internal_output = true;
      std::string kernel_target = GetCNodeTarget(front_real_kernel.first);
      for (auto user : users) {
        auto cnode = user.first->cast<CNodePtr>();
        if (cnode == nullptr) {
          internal_output = false;
          break;
        }
        auto prim = cnode->input(kAnfPrimitiveIndex);
        if (prim == nullptr || !prim->isa<ValueNode>()) {
          internal_output = false;
          break;
        }
        if (!AnfAlgo::IsRealKernel(user.first) || kernel_target != GetCNodeTarget(user.first)) {
          internal_output = false;
          break;
        }
      }
      if (internal_output) {
        MS_LOG(INFO) << "Internal output1: " << out->DebugString() << "To " << backend_real_kernel.first->DebugString();
        graph->AddInternalOutput(out, backend_real_kernel.first);
      }
      return backend_anf;
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
  MS_EXCEPTION_IF_NULL(graph);
  if (AnfRuntimeAlgorithm::GetOutputTensorNum(cnode) > 1) {
    for (size_t output_index = 0; output_index < AnfRuntimeAlgorithm::GetOutputTensorNum(cnode); output_index++) {
      auto idx = NewValueNode(SizeToInt(output_index));
      MS_EXCEPTION_IF_NULL(idx);
      auto imm = std::make_shared<Int32Imm>(output_index);
      idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
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

std::shared_ptr<KernelGraph> SessionBasic::ConstructSingleOpGraph(const OpRunInfo &op_run_info,
                                                                  const std::vector<tensor::TensorPtr> &input_tensors,
                                                                  const std::vector<int> &tensors_mask) {
  auto graph = std::make_shared<KernelGraph>();
  std::vector<AnfNodePtr> inputs;
  // set input[0]
  PrimitivePtr op_prim = op_run_info.py_primitive;
  MS_EXCEPTION_IF_NULL(op_prim);
  inputs.push_back(std::make_shared<ValueNode>(op_prim));
  // set input parameter
  MS_LOG(INFO) << "Input tensor size: " << input_tensors.size();
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    if (tensors_mask[i] == kValueNodeTensorMask) {
      auto value_node = ConstructRunOpValueNode(graph, input_tensors[i]);
      inputs.push_back(value_node);
      continue;
    }
    auto parameter = ConstructRunOpParameter(graph, input_tensors[i], tensors_mask[i]);
    inputs.push_back(parameter);
    auto mutable_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(mutable_inputs);
    mutable_inputs->push_back(parameter);
  }
  // set execution order
  auto cnode = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(cnode);
  // set abstract,which include inferred shapes and types
  cnode->set_abstract(op_run_info.abstract);
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

KernelGraphPtr SessionBasic::NewKernelGraph() {
  auto graph = std::make_shared<KernelGraph>();
  graph->set_graph_id(graph_sum_);
  graphs_[graph_sum_++] = graph;
  return graph;
}

AnfNodePtr SessionBasic::FindPullNode(const AnfNodePtr &push_node, const std::vector<AnfNodePtr> &node_list) {
  MS_EXCEPTION_IF_NULL(push_node);
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>()) {
      for (auto input : node->cast<CNodePtr>()->inputs()) {
        if (push_node == AnfAlgo::VisitKernel(input, 0).first) {
          if (AnfAlgo::GetCNodeName(node) != kPullOpName) {
            MS_LOG(EXCEPTION) << "The edge between Push and Pull node is invalid.";
          }
          return node;
        }
      }
    }
  }
  return nullptr;
}

#if (!_WIN32 && !ENABLE_GE && !ENABLE_TESTCASES)
void SessionBasic::AssignParamKey(const KernelGraphPtr &kernel_graph) {
  if (!parallel::ps::Util::IsRoleOfWorker()) {
    MS_LOG(INFO) << "Not parameter server mode.";
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(kernel_graph->get_return());
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>()) {
      // Assign key for forward kernel EmbeddingLookup.
      // The key will be assigned to embedding table ande Push kernel as well.
      if (AnfAlgo::GetCNodeName(node) == kEmbeddingLookupOpName) {
        size_t embedding_table_idx = 0;
        auto embedding_table = AnfAlgo::GetInputNode(node->cast<CNodePtr>(), embedding_table_idx);
        size_t key = parallel::ps::Worker<float>::GetInstance().SetParamKey(embedding_table->fullname_with_scope());
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), node);
      } else if (AnfAlgo::GetCNodeName(node) == kPushOpName) {
        auto pull_node = FindPullNode(node, node_list);
        if (!pull_node) {
          MS_LOG(EXCEPTION) << "Assigning parameter key failed: can't find Pull node of the Push node.";
        }

        // Second input of Pull node is the trainable parameter.
        size_t parameter_index = 1;
        auto parameter_node = AnfAlgo::GetInputNode(pull_node->cast<CNodePtr>(), parameter_index);
        size_t key = parallel::ps::Worker<float>::GetInstance().SetParamKey(parameter_node->fullname_with_scope());
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), node);
        AnfAlgo::SetNodeAttr(kAttrPsKey, MakeValue(key), pull_node);

        std::string optimizer_name = AnfAlgo::GetNodeAttr<std::string>(node, kAttrOptimizerType);
        parallel::ps::Worker<float>::GetInstance().SetKeyOptimId(key, optimizer_name);
      }
    }
  }
}

void SessionBasic::InitPSParamAndOptim(const KernelGraphPtr &kernel_graph,
                                       const std::vector<tensor::TensorPtr> &inputs_const) {
  if (!parallel::ps::Util::IsRoleOfWorker()) {
    return;
  }
  std::vector<tensor::TensorPtr> inputs(inputs_const);
  size_t input_ctrl_size = 1;
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->input_ctrl_tensors()) {
    input_ctrl_size = LoadCtrlInputTensor(kernel_graph, &inputs);
  }
  auto input_nodes = kernel_graph->inputs();
  if ((inputs.size() + input_ctrl_size) - 1 != input_nodes.size()) {
    MS_LOG(EXCEPTION) << "Tensor input:" << inputs.size() << " is not equal graph inputs:" << input_nodes.size()
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
      mindspore::parallel::ps::Worker<float>::GetInstance().InitPSParamAndOptim(
        pk_node->fullname_with_scope(), tensor->data_c(), LongToSize(tensor->data().nbytes()));
    }
  }
  ps_init_ = true;
}
#endif
}  // namespace session
}  // namespace mindspore
