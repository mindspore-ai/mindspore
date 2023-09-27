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

#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>
#include <memory>

#include "src/extendrt/utils/func_graph_utils.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/convert_utils.h"
#include "mindspore/ccsrc/include/backend/optimizer/helper.h"

#include "ops/op_name.h"
#include "tools/optimizer/format/to_nhwc_format.h"
#include "tools/optimizer/graph/decrease_transpose_algo.h"

namespace mindspore {
const PrimitivePtr kPrimMakeTupleV2 = std::make_shared<Primitive>("make_tuple");
ValuePtr FuncGraphUtils::GetNodeValuePtr(AnfNodePtr input_node) {
  if (input_node == nullptr) {
    return nullptr;
  }
  if (IsPrimitiveCNode(input_node, prim::kPrimDepend)) {
    input_node = AnfUtils::VisitKernel(input_node, 0).first;
  }
  ValuePtr value = nullptr;
  if (input_node->isa<ValueNode>() && !HasAbstractMonad(input_node)) {
    auto value_node = input_node->cast<ValueNodePtr>();
    if (value_node) {
      value = value_node->value();
    }
  } else if (input_node->isa<Parameter>()) {
    auto parameter = input_node->cast<ParameterPtr>();
    if (parameter->has_default()) {
      value = parameter->default_param();
    }
  }
  return value;
}

tensor::TensorPtr FuncGraphUtils::GetConstNodeValue(AnfNodePtr input_node) {
  ValuePtr value = GetNodeValuePtr(input_node);
  if (value == nullptr) {
    return nullptr;
  }
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor == nullptr || tensor->data().const_data() == nullptr) {
      return nullptr;
    }
    return tensor;
  }
  if (value->isa<Scalar>()) {
    return ScalarToTensor(value->cast<ScalarPtr>());
  }
  if (value->isa<ValueTuple>()) {
    return opt::CreateTupleTensor(value->cast<ValueTuplePtr>());
  }
  if (value->isa<Type>()) {
    auto type_ptr = value->cast<TypePtr>();
    if (type_ptr == nullptr) {
      return nullptr;
    }
    return std::make_shared<tensor::Tensor>(static_cast<int64_t>(type_ptr->type_id()), type_ptr->type());
  }
  MS_LOG(WARNING) << "Unexpected value type " << value->type_name() << " for " << input_node->fullname_with_scope();
  return nullptr;
}

bool FuncGraphUtils::GetCNodeOperator(const mindspore::CNodePtr &cnode,
                                      mindspore::kernel::BaseOperatorPtr *base_operator) {
  if (!cnode || !base_operator) {
    MS_LOG(ERROR) << "Input cnode or base_operator cannot be nullptr";
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim) {
    MS_LOG(ERROR) << "Primitive of cnode " << cnode->fullname_with_scope() << " cannot be nullptr";
    return false;
  }
  auto kernel_name = prim->name();
  ops::PrimitiveCPtr primc_ptr = nullptr;
  static auto &primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  auto primc_it = primc_fns.find(kernel_name);
  if (primc_it != primc_fns.end() && primc_it->second) {
    primc_ptr = primc_it->second();
  }
  if (primc_ptr == nullptr) {
    MS_LOG(ERROR) << "OpPrimCRegister can not find " << kernel_name;
    return false;
  }
  (void)primc_ptr->SetAttrs(prim->attrs());

  *base_operator = nullptr;
  static auto &operator_fns = ops::OperatorRegister::GetInstance().GetOperatorMap();
  auto op_it = operator_fns.find(kernel_name);
  if (op_it != operator_fns.end() && op_it->second) {
    *base_operator = op_it->second(primc_ptr);
  }
  if (*base_operator == nullptr) {
    MS_LOG(ERROR) << "Failed to create operator of type " << kernel_name;
    return false;
  }
  return true;
}

bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  if (node == nullptr || primitive_type == nullptr) {
    return false;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    return IsPrimitive(cnode->input(kAnfPrimitiveIndex), primitive_type);
  } else if (node->isa<ValueNode>()) {
    return IsPrimitive(node, primitive_type);
  }
  return false;
}

std::vector<common::KernelWithIndex> FuncGraphUtils::GetNodeInputs(const AnfNodePtr &anf_node) {
  if (anf_node == nullptr) {
    return {};
  }
  if (!anf_node->isa<CNode>()) {
    return {{anf_node, 0}};
  }
  auto cnode = anf_node->cast<CNodePtr>();
  std::vector<common::KernelWithIndex> inputs;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t input_idx = 0; input_idx < input_num; ++input_idx) {
    const auto &pre_node_output = common::AnfAlgo::GetPrevNodeOutput(cnode, input_idx);
    auto pre_node = pre_node_output.first;
    if (CheckPrimitiveType(pre_node, prim::kPrimMakeTuple) || CheckPrimitiveType(pre_node, kPrimMakeTupleV2)) {
      auto tuple_inputs = GetNodeInputs(pre_node);
      std::copy(tuple_inputs.begin(), tuple_inputs.end(), std::back_inserter(inputs));
    } else if (CheckPrimitiveType(pre_node, prim::kPrimSplit) &&
               CheckPrimitiveType(cnode->input(1), prim::kPrimSplit)) {
      inputs = common::AnfAlgo::GetAllOutputWithIndex(pre_node);
    } else {
      inputs.push_back(pre_node_output);
    }
  }
  return inputs;
}

bool FuncGraphUtils::GetCNodeInputsOutputs(const mindspore::CNodePtr &cnode,
                                           std::vector<AnfWithOutIndex> *input_tensors,
                                           std::vector<AnfWithOutIndex> *output_tensors) {
  if (!cnode || !input_tensors || !output_tensors) {
    MS_LOG(ERROR) << "Input cnode, input_tensors or output_tensors cannot be nullptr";
    return false;
  }
  // Makeup input tensors.
  *input_tensors = GetNodeInputs(cnode);
  // Makeup output tensors.
  output_tensors->clear();
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  for (size_t output_idx = 0; output_idx < output_num; ++output_idx) {
    session::KernelWithIndex tensor_id = {cnode, output_idx};
    output_tensors->push_back(tensor_id);
  }
  return true;
}

bool FuncGraphUtils::GetFuncGraphInputs(const FuncGraphPtr &func_graph, std::vector<AnfWithOutIndex> *inputs) {
  if (!func_graph || !inputs) {
    MS_LOG(ERROR) << "Input func_graph or inputs cannot be nullptr";
    return false;
  }
  auto graph_inputs = func_graph->get_inputs();
  // find parameters of graph inputs
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    auto input = graph_inputs[i];
    if (input == nullptr) {
      MS_LOG(ERROR) << "Input " << i << " of FuncGraph is nullptr.";
      return false;
    }
    auto parameter = input->cast<ParameterPtr>();
    if (!parameter) {
      MS_LOG(ERROR) << "Input " << input->fullname_with_scope() << " of FuncGraph is not type of Parameter.";
      return false;
    }
    if (common::AnfAlgo::IsParameterWeight(parameter)) {
      continue;
    }
    inputs->push_back(std::make_pair(input, 0));
  }
  return true;
}

bool FuncGraphUtils::GetFuncGraphOutputs(const FuncGraphPtr &func_graph, std::vector<AnfWithOutIndex> *outputs) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Input func_graph cannot be nullptr!";
    return false;
  }

  if (outputs == nullptr) {
    MS_LOG(ERROR) << "Outputs cannot be nullptr!";
    return false;
  }

  *outputs = GetNodeInputs(func_graph->get_return());
  return true;
}

DataType FuncGraphUtils::GetTensorDataType(const AnfWithOutIndex &tensor) {
  auto node = tensor.first;
  auto output_idx = tensor.second;
  auto tensor_val = GetConstNodeValue(node);
  TypeId type_id;
  if (tensor_val) {
    type_id = tensor_val->Dtype()->type_id();
  } else {
    type_id = common::AnfAlgo::GetOutputInferDataType(node, output_idx);
  }
  return static_cast<enum DataType>(type_id);
}

ShapeVector FuncGraphUtils::GetTensorShape(const AnfWithOutIndex &tensor) {
  auto node = tensor.first;
  auto output_idx = tensor.second;
  auto tensor_val = GetConstNodeValue(node);
  ShapeVector shape;
  if (tensor_val) {
    shape = tensor_val->shape_c();
  } else {
    shape = common::AnfAlgo::GetOutputInferShape(node, output_idx);
  }
  return shape;
}

Status FuncGraphUtils::UnifyGraphToNHWCFormat(const FuncGraphPtr &graph) {
  auto value = graph->get_attr(ops::kFormat);
  if (value != nullptr && GetValue<int64_t>(value) != mindspore::NHWC) {
    auto format_pass = std::make_shared<opt::ToNHWCFormat>();
    MS_CHECK_TRUE_RET(format_pass != nullptr, kLiteNullptr);
    if (!format_pass->Run(graph)) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::Partition Run ToNHWCFormat pass failed";
      return kLiteNullptr;
    }
    auto transpose_pass = std::make_shared<opt::DecreaseTransposeAlgo>();
    MS_CHECK_TRUE_RET(transpose_pass != nullptr, kLiteNullptr);
    if (!transpose_pass->Run(graph)) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::Partition Run DecreaseTransposeAlgo pass failed";
      return kLiteNullptr;
    }
  }
  return kSuccess;
}

std::string FuncGraphUtils::GetTensorName(const AnfWithOutIndex &tensor) {
  auto node = tensor.first;
  auto idx = tensor.second;
  MS_EXCEPTION_IF_NULL(node);
  AbstractBasePtr abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  if (utils::isa<abstract::AbstractTuplePtr>(abstract)) {
    auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(abstract);
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    auto abstract_list = abstract_tuple->elements();
    if (abstract_list.size() <= idx) {
      MS_LOG(ERROR) << "AbstractTuple's size[" << abstract_list.size() << "] is smaller than expect size[" << idx
                    << "]";
      return "";
    }
    abstract = abstract_list[idx];
    MS_EXCEPTION_IF_NULL(abstract);
  }
  MS_EXCEPTION_IF_NULL(abstract);
  std::string output_name;
  if (!abstract->name().empty()) {
    output_name = abstract->name();
  } else if (idx > 0) {
    output_name = node->fullname_with_scope() + ":" + std::to_string(idx);
  } else {
    output_name = node->fullname_with_scope();
  }
  return output_name;
}

AbstractBasePtr FuncGraphUtils::GetAbstract(const AnfWithOutIndex &tensor) {
  auto node = tensor.first;
  auto idx = tensor.second;
  MS_EXCEPTION_IF_NULL(node);
  AbstractBasePtr abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  return common::AnfAlgo::FetchAbstractByIndex(node->abstract(), idx);
}

void FuncGraphUtils::GetFuncGraphInputsInfo(const FuncGraphPtr &func_graph, std::vector<tensor::TensorPtr> *inputs,
                                            std::vector<std::string> *inputs_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(inputs_name);
  std::vector<AnfWithOutIndex> input_idxs;
  if (!GetFuncGraphInputs(func_graph, &input_idxs)) {
    MS_LOG(ERROR) << "Failed to get input infos from graph";
    return;
  }
  inputs->clear();
  inputs_name->clear();
  for (auto &tensor : input_idxs) {
    auto name = FuncGraphUtils::GetTensorName(tensor);
    auto data_type = FuncGraphUtils::GetTensorDataType(tensor);
    auto shape = FuncGraphUtils::GetTensorShape(tensor);
    auto ms_tensor = std::make_shared<tensor::Tensor>(static_cast<TypeId>(data_type), shape);
    ms_tensor->set_name(name);
    inputs->push_back(ms_tensor);
    inputs_name->push_back(name);
  }
}

void FuncGraphUtils::GetFuncGraphOutputsInfo(const FuncGraphPtr &func_graph, std::vector<tensor::TensorPtr> *outputs,
                                             std::vector<std::string> *output_names) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_names);
  std::vector<AnfWithOutIndex> output_idxs;
  if (!GetFuncGraphOutputs(func_graph, &output_idxs)) {
    MS_LOG(ERROR) << "Failed to get input infos from graph";
    return;
  }
  outputs->clear();
  output_names->clear();
  for (auto &tensor : output_idxs) {
    auto name = FuncGraphUtils::GetTensorName(tensor);
    auto data_type = FuncGraphUtils::GetTensorDataType(tensor);
    auto shape = FuncGraphUtils::GetTensorShape(tensor);
    auto ms_tensor = std::make_shared<tensor::Tensor>(static_cast<TypeId>(data_type), shape);
    ms_tensor->set_name(name);
    outputs->push_back(ms_tensor);
    output_names->push_back(name);
  }
}

std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> FuncGraphUtils::TransformSegmentToAnfGraph(
  const AnfNodePtrList &lst) {
  if (lst.empty()) {
    MS_LOG(EXCEPTION) << "Input anf node list is empty";
  }
  FuncGraphPtr fg = nullptr;
  {
    // limit the lifetime of guard.
    MS_EXCEPTION_IF_NULL(lst[0]);
    MS_EXCEPTION_IF_NULL(lst[0]->cast<CNodePtr>());
    MS_EXCEPTION_IF_NULL(lst[0]->cast<CNodePtr>()->func_graph());
    TraceGuard guard(std::make_shared<TraceSegmentTransform>(lst[0]->cast<CNodePtr>()->func_graph()->debug_info()));
    fg = std::make_shared<FuncGraph>();
  }
  AnfNodePtrList inputs;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> eqv;
  // Merge CNodes into a AnfGraph that represents a linear instruction segment
  for (auto n : lst) {
    MS_EXCEPTION_IF_NULL(n);
    if (!n->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Inst is not CNode";
    }
    auto &inps = n->cast<CNodePtr>()->inputs();
    if (inps.empty()) {
      MS_LOG(EXCEPTION) << "Input is empty";
    }
    if (!IsValueNode<Primitive>(inps[0]) &&
        !(IsValueNode<FuncGraph>(inps[0]) &&
          inps[0]->cast<ValueNodePtr>()->value()->cast<FuncGraphPtr>()->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL))) {
      MS_LOG(EXCEPTION) << "Input[0] must be a Primitive ValueNode";
    }
    auto fn = inps[0];
    std::vector<AnfNodePtr> args{fn};
    if (IsPrimitive(fn, prim::kPrimDepend) && inps.size() >= kDependInputSize &&
        eqv.find(inps[kDependAttachNodeIndex]) == eqv.end()) {
      args.emplace_back(RefSubGraphNode(fg, inps[kRealInputIndexInDepend], &inputs, &eqv));
      const size_t value_start_index = 2;
      for (size_t i = value_start_index; i < inps.size(); ++i) {
        args.emplace_back(NewValueNode(MakeValue(0)));
      }
    } else {
      (void)std::transform(std::begin(inps) + 1, std::end(inps), std::back_inserter(args),
                           [&fg, &inputs, &eqv](const AnfNodePtr &a) { return RefSubGraphNode(fg, a, &inputs, &eqv); });
    }
    TraceGuard tg(std::make_shared<TraceSegmentTransform>(n->debug_info()));
    MS_EXCEPTION_IF_NULL(fg);
    eqv[n] = fg->NewCNode(args);
    eqv[n]->set_abstract(n->abstract());
    eqv[n]->set_kernel_info(n->kernel_info_ptr());
  }
  mindspore::HashSet<AnfNodePtr> eqv_keys;
  for (auto &e : eqv) {
    (void)eqv_keys.emplace(e.first);
  }
  auto mgr = lst[0]->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  auto outputs = GetOutput(lst, mgr->node_users(), eqv_keys);
  AnfNodePtr fg_output;
  if (outputs.size() > 1) {
    std::vector<AnfNodePtr> output_args;
    output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
    (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_args),
                         [&eqv](const AnfNodePtr &o) -> AnfNodePtr { return eqv[o]; });
    // Set output for AnfGraph
    fg_output = fg->NewCNode(output_args);
  } else {
    if (outputs.empty()) {
      MS_LOG(EXCEPTION) << "Output is empty.";
    }
    fg_output = eqv[outputs[0]];
  }
  fg->set_output(fg_output);
  return std::make_tuple(fg, inputs, outputs);
}

AnfNodePtrList FuncGraphUtils::GetOutput(const AnfNodePtrList &nodes, const NodeUsersMap &users,
                                         const mindspore::HashSet<AnfNodePtr> &seen) {
  AnfNodePtrList output;
  if (users.size() == 0) {
    return output;
  }
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto iter = users.find(node);
    if (iter == users.end()) {
      continue;
    }
    auto &node_users = iter->second;
    const bool has_outer_user = std::any_of(std::begin(node_users), std::end(node_users),
                                            [&seen](const std::pair<AnfNodePtr, int64_t> &u) -> bool {
                                              const bool is_outer_user = (seen.find(u.first) == seen.end());
                                              return is_outer_user;
                                            });
    if (has_outer_user) {
      output.emplace_back(node);
    }
  }
  return output;
}

AnfNodePtr FuncGraphUtils::RefSubGraphNode(const FuncGraphPtr &fg, const AnfNodePtr &node, AnfNodePtrList *inputs_ptr,
                                           mindspore::HashMap<AnfNodePtr, AnfNodePtr> *eqv_ptr) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(inputs_ptr);
  MS_EXCEPTION_IF_NULL(eqv_ptr);
  MS_EXCEPTION_IF_NULL(node);
  auto &inputs = *inputs_ptr;
  auto &eqv = *eqv_ptr;
  if (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) {
    eqv[node] = node;
  } else if (eqv.find(node) == eqv.end()) {
    inputs.push_back(node);
    eqv[node] = fg->add_parameter();
    eqv[node]->set_abstract(node->abstract());
    eqv[node]->set_kernel_info(node->kernel_info_ptr());
  }
  return eqv[node];
}
}  // namespace mindspore
