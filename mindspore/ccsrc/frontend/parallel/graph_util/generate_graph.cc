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

#include "frontend/parallel/graph_util/generate_graph.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "base/base.h"
#include "include/utils/python_adapter.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "ir/tensor_new.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"
#include "utils/log_adapter.h"
#include "ops_utils/op_utils.h"
#include "ops/op_def.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace parallel {
namespace {
ValuePtr CreateOpPrimtiveWithAttrs(const OperatorAttrs &attrs, const OperatorName &op_name,
                                   const std::string &instance_name) {
  auto op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    return CreateOpInstance(attrs, op_name, instance_name);
  }

  auto prim = std::make_shared<Primitive>(op_name);
  MS_EXCEPTION_IF_NULL(prim);
  prim->set_instance_name(instance_name);
  for (const auto &[name, value] : attrs) {
    prim->set_attr(name, value);
  }

  return prim;
}

std::vector<AnfNodePtr> RectifyInputsForNewCNode(const std::vector<AnfNodePtr> &inputs) {
  if (inputs.size() <= 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "For NewCNode, the inputs should not less than two!";
  }

  auto value_node = inputs[0]->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto prim = value->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);

  auto op_name = prim->name();
  auto op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    return inputs;
  }

  std::vector<AnfNodePtr> new_inputs(inputs.begin(), inputs.end());
  auto op_inputs_num = op_def->indexes_.size();
  auto graph_view_prim = op_def->is_graph_view_;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto ge_mode = context->GetBackend() == "GE";
  if (graph_view_prim == true && !ge_mode) {
    op_inputs_num = op_inputs_num + 1;  // for umonad input
    auto monad_input = NewValueNode(kUMonad);
    monad_input->set_abstract(kUMonad->ToAbstract());
    (void)new_inputs.emplace_back(monad_input);
  }
  new_inputs.resize(op_inputs_num + 1);  // 1 for primitive.

  // For new defined op, almost all old attrs is changed to inputs.
  std::vector<std::string> latter_erase;
  auto attrs = prim->attrs();
  for (const auto &[name, value] : attrs) {
    auto [is_input, node_input_idx] = CheckAndGetValidIdxByOpDef(op_def, op_name, name, new_inputs.size());
    if (!is_input) {
      continue;
    }
    new_inputs[node_input_idx] = NewValueNode(value);
    latter_erase.push_back(name);
  }

  for (const auto &name : latter_erase) {
    prim->EraseAttr(name);
  }

  return new_inputs;
}
}  // namespace

std::pair<bool, size_t> CheckAndGetValidIdxByOpDef(const ops::OpDefPtr &op_def, const std::string &op_name,
                                                   const std::string &attr_name, size_t limit_size) {
  auto ks_iter = op_def->indexes_.find(attr_name);
  if (ks_iter == op_def->indexes_.end()) {
    MS_LOG(DEBUG) << "For " << op_name << ", cannot find a valid index for input " << attr_name
                  << " in operator-definition.";
    return std::make_pair(false, SIZE_MAX);
  }

  auto idx = ks_iter->second;
  auto real_idx = idx + 1;
  if (real_idx >= limit_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << op_name << ", " << idx << " is not a valid index for input " << attr_name;
  }
  return std::make_pair(true, real_idx);
}

const char *GetOpPythonPath(const char *op_name) {
  static const py::module inner_mod = py::module::import(INNER_OP_PATH);
  if (py::hasattr(inner_mod, op_name)) {
    return INNER_OP_PATH;
  }

  static const py::module mod = py::module::import(OP_PATH);
  if (py::hasattr(mod, op_name)) {
    return OP_PATH;
  }

  static const py::module grad_mod = py::module::import(GRAD_OP_PATH);
  if (py::hasattr(grad_mod, op_name)) {
    return GRAD_OP_PATH;
  }

  static const py::module nn_mod = py::module::import(NN_OPS_PATH);
  if (py::hasattr(nn_mod, op_name)) {
    return NN_OPS_PATH;
  }

  static const py::module functional_mod = py::module::import(FUNCTIONAL_OP_PATH);
  if (!py::hasattr(functional_mod, op_name)) {
    MS_LOG(EXCEPTION) << OP_PATH << " and " << INNER_OP_PATH << " and " << GRAD_OP_PATH << " and " << NN_OPS_PATH
                      << "and" << FUNCTIONAL_OP_PATH << " don't have op:" << op_name;
  }
  return FUNCTIONAL_OP_PATH;
}

ValuePtr CreateOpInstance(const OperatorAttrs &attrs, const OperatorName &op_name, const std::string &instance_name) {
  const auto op_path = GetOpPythonPath(op_name.c_str());
  std::vector<py::object> arg_list;
  (void)std::transform(attrs.begin(), attrs.end(), std::back_inserter(arg_list),
                       [](const Attr &attr) { return ValueToPyData(attr.second); });
  py::object obj =
    python_adapter::CallPyFn(GET_OP_FUNCTION_PATH, GET_OP_FUNCTION, op_name, op_path, instance_name, arg_list);
  ValuePtr op_instance = nullptr;
  bool succ = parse::ConvertData(obj, &op_instance);
  if (!succ) {
    MS_LOG(ERROR) << "Failure: get Python op " << op_path << " from " << op_name << " fail";
    return nullptr;
  }
  return op_instance;
}

std::vector<AnfNodePtr> ConvertToRealInputs(const OperatorName &op_name, const std::string &instance_name,
                                            const AnfNodePtrList &inputs, const OperatorAttrs &attrs) {
  auto op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    // Create old op from python for creating some attr in __init__.
    auto prim_value = CreateOpInstance(attrs, op_name, instance_name);
    AnfNodePtrList node_inputs = {NewValueNode(prim_value)};
    node_inputs.insert(node_inputs.end(), inputs.begin(), inputs.end());
    return node_inputs;
  }

  size_t op_inputs_num = inputs.size() + attrs.size();
  if (op_inputs_num != op_def->indexes_.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << op_name << ", inputs should be " << op_def->indexes_.size()
                               << ", but got given inputs num " << inputs.size() << " and attrs num " << attrs.size();
  }

  auto prim = std::make_shared<Primitive>(op_name);
  MS_EXCEPTION_IF_NULL(prim);
  prim->set_instance_name(instance_name);

  AnfNodePtrList node_inputs;
  node_inputs.resize(1 + op_inputs_num);  // 1 for primitive value node.
  node_inputs[0] = NewValueNode(prim);

  for (size_t i = 0; i < inputs.size(); ++i) {
    node_inputs[i + 1] = inputs[i];
  }

  // For new-defined op, almost all attrs are inputs now, here should insert the value as input in right position.
  for (size_t i = 0; i < attrs.size(); ++i) {
    auto [attr_name, attr_value] = attrs[i];
    auto [is_input, node_input_idx] = CheckAndGetValidIdxByOpDef(op_def, op_name, attr_name, node_inputs.size());
    if (!is_input) {
      continue;
    }
    node_inputs[node_input_idx] = NewValueNode(attr_value);
  }

  return node_inputs;
}

CNodePtr CreateCNodeByInputsAndAttr(const FuncGraphPtr &func_graph, const OperatorName &op_name,
                                    const std::string &instance_name, const AnfNodePtrList &inputs,
                                    const OperatorAttrs &attrs) {
  auto real_inputs = ConvertToRealInputs(op_name, instance_name, inputs, attrs);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto cnode = func_graph->NewCNode(real_inputs);
  return cnode;
}

CNodePtr CreateNewCNodeForReplace(const CNodePtr &origin_node, const PrimitivePtr &new_prim) {
  MS_EXCEPTION_IF_NULL(origin_node);
  auto func_graph = origin_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto inputs = origin_node->inputs();
  AnfNodePtrList new_inputs(inputs.begin(), inputs.end());

  MS_EXCEPTION_IF_NULL(new_prim);
  auto op_name = new_prim->name();
  auto op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def != nullptr) {
    // For new defined op, almost all old attrs is changed to inputs.
    std::vector<std::string> latter_erase;
    auto attrs = new_prim->attrs();
    for (const auto &[name, value] : attrs) {
      auto [is_input, node_input_idx] = CheckAndGetValidIdxByOpDef(op_def, op_name, name, inputs.size());
      if (!is_input) {
        continue;
      }
      if (!inputs[node_input_idx]->isa<ValueNode>()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, origin_node)
          << "For auto parallel, the " << (node_input_idx - 1) << " input of " << op_name << " must be a value node!";
      }

      inputs[node_input_idx] = NewValueNode(value);
      latter_erase.push_back(name);
    }

    for (const auto &name : latter_erase) {
      new_prim->EraseAttr(name);
    }
  }

  new_inputs[0] = NewValueNode(new_prim);
  return func_graph->NewCNode(new_inputs);
}

AnfNodePtr ValuePtrToAnfNodePtr(const ValuePtr &value_ptr) {
  auto value_node = NewValueNode(value_ptr);
  MS_EXCEPTION_IF_NULL(value_node);
  return value_node->cast<AnfNodePtr>();
}

AnfNodePtr CreateInt32Tensor(int64_t value, bool int64_type) {
  mindspore::tensor::TensorPtr tensor_ptr;
  auto dtype = int64_type ? kInt64 : kInt32;
  tensor_ptr = tensor::from_scalar(value, dtype);

  ValuePtr value_ptr = MakeValue(tensor_ptr);
  auto anf_node_ptr = ValuePtrToAnfNodePtr(value_ptr);
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  anf_node_ptr->set_abstract(std::make_shared<abstract::AbstractTensor>(dtype, Shape{}));
  return anf_node_ptr;
}

AnfNodePtr CreateFP32Tensor(float value) {
  mindspore::tensor::TensorPtr tensor_ptr = tensor::from_scalar(value, kFloat32);
  ValuePtr value_ptr = MakeValue(tensor_ptr);
  auto anf_node_ptr = ValuePtrToAnfNodePtr(value_ptr);
  return anf_node_ptr;
}

AnfNodePtr CreateTypeInt(int64_t nbits) {
  ValuePtr value_ptr = MakeValue(std::make_shared<Int>(nbits));
  return ValuePtrToAnfNodePtr(value_ptr);
}

AnfNodePtr CreateTypeFloat(int64_t nbits) {
  ValuePtr value_ptr = MakeValue(std::make_shared<Float>(nbits));
  return ValuePtrToAnfNodePtr(value_ptr);
}

AnfNodePtr CreatInt64Imm(int64_t value) {
  ValuePtr value_ptr = MakeValue(std::make_shared<Int64Imm>(value));
  return ValuePtrToAnfNodePtr(value_ptr);
}

AnfNodePtr CreateFP32Imm(float value) {
  ValuePtr value_ptr = MakeValue(std::make_shared<FP32Imm>(value));
  return ValuePtrToAnfNodePtr(value_ptr);
}

AnfNodePtr CreateBoolImm(bool value) {
  ValuePtr value_ptr = MakeValue(std::make_shared<BoolImm>(value));
  return ValuePtrToAnfNodePtr(value_ptr);
}

AnfNodePtr CreateStringImm(std::string value) {
  ValuePtr value_ptr = MakeValue(std::make_shared<StringImm>(value));
  return ValuePtrToAnfNodePtr(value_ptr);
}

AnfNodePtr CreateTuple(const std::vector<int64_t> &tuple) {
  std::vector<ValuePtr> value_list;
  (void)std::transform(tuple.begin(), tuple.end(), std::back_inserter(value_list),
                       [](const int64_t value) { return MakeValue(value); });
  ValueTuplePtr value_tuple_ptr = std::make_shared<ValueTuple>(value_list);
  return ValuePtrToAnfNodePtr(value_tuple_ptr);
}

std::string GetInstanceNameByCNode(const CNodePtr &cnode) {
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (!prim) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The first input of the cnode is not a PrimitivePtr.";
  }
  std::string instance_name = prim->instance_name();
  return HashInstanceName(instance_name);
}

std::string HashInstanceName(const std::string &name) {
  auto using_hash_name = common::GetEnv(USING_HASH_NAME);
  std::string instance_name;
  if ((using_hash_name.empty()) || (using_hash_name == "on")) {
    instance_name = HashName(name);
  } else {
    instance_name = name;
  }
  return instance_name;
}

void InsertVirtualPipelineEndNode(const CNodePtr &cnode, const FuncGraphManagerPtr &manager, size_t index,
                                  std::string end_flag) {
  auto pre_cnode = cnode->input(index)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(pre_cnode);
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  OperatorAttrs attrs_;
  auto op = CreateOpInstance(attrs_, "_VirtualPipelineEnd", "end_node");
  auto value_node = NewValueNode(op);
  auto virtual_end = graph->NewCNode({value_node, pre_cnode});
  virtual_end->set_abstract(pre_cnode->abstract());
  virtual_end->AddPrimalAttr(end_flag, pre_cnode->GetPrimalAttr(MICRO));
  virtual_end->AddPrimalAttr(MICRO, pre_cnode->GetPrimalAttr(MICRO));
  manager->SetEdge(cnode, SizeToInt(index), virtual_end);
  if (ParallelContext::GetInstance()->enable_fold_pipeline()) {
    auto seg = ParallelContext::GetInstance()->pipeline_segment_split_num();
    virtual_end->AddPrimalAttr(SEGMENT, MakeValue(seg - 1));
  }
}

CNodePtr CreateVirtualConverterBeginNode(const AnfNodePtr &input_cnode, size_t output_nums) {
  auto graph = input_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  Attr output_nums_attr = {"output_nums", MakeValue(output_nums)};
  OperatorAttrs attrs_ = {output_nums_attr};
  auto op = CreateOpInstance(attrs_, "_VirtualConverterBegin", "virtual_converter_begin");
  auto value_node = NewValueNode(op);
  auto virtual_begin = graph->NewCNode({value_node, input_cnode});
  return virtual_begin;
}

CNodePtr CreateVirtualConverterEndNode(const FuncGraphPtr &graph, const std::vector<CNodePtr> &input_cnodes) {
  if (input_cnodes.empty()) {
    MS_LOG(EXCEPTION) << "input cnodes for _VirtualConverterEnd is empty.";
  }
  Attr input_nums_attr = {"input_nums", MakeValue(input_cnodes.size())};
  OperatorAttrs attrs_ = {input_nums_attr};
  auto op = CreateOpInstance(attrs_, "_VirtualConverterEnd", "virtual_converter_End");
  auto value_node = NewValueNode(op);
  std::vector<AnfNodePtr> virtual_end_input = {value_node};
  std::copy(input_cnodes.begin(), input_cnodes.end(), std::back_inserter(virtual_end_input));
  auto virtual_end = graph->NewCNode(virtual_end_input);
  return virtual_end;
}

Status GenerateGraph::Init(const CNodePtr &cnode) {
  if (!cnode) {
    MS_LOG(ERROR) << "Init:cnode is nullptr";
    return FAILED;
  }
  cnode_ = cnode;
  func_graph_ = cnode->func_graph();
  if (!func_graph_) {
    MS_LOG(ERROR) << "Init:func_graph_ is nullptr";
    return FAILED;
  }
  manager_ = func_graph_->manager();
  if (!manager_) {
    MS_LOG(ERROR) << "Init:manager_ is nullptr";
    return FAILED;
  }
  scope_ = cnode_->scope();
  if (!scope_) {
    MS_LOG(ERROR) << "Init:scope_ is nullptr";
    return FAILED;
  }
  virtual_input_node_ = std::make_shared<AnfNode>(nullptr);
  virtual_input_node_->set_scope(scope_);
  instance_name_base_ = GetInstanceNameByCNode(cnode_);
  name_idx_ = 0;
  return SUCCESS;
}

AnfNodePtr GenerateGraph::PushBack(const std::vector<AnfNodePtr> &inputs) {
  auto new_inputs = RectifyInputsForNewCNode(inputs);
  for (auto &input : new_inputs) {
    MS_EXCEPTION_IF_NULL(input);  // if error raise here, check if inputs need include attrs
  }
  CNodePtr cnode = func_graph_->NewCNode(new_inputs);  // using NewCNode to create anfnode
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  SetUserAttrs(origin_attrs_, prim);
  if (cnode_->HasPrimalAttr(PARAMETER_START_SHARE_CELL)) {
    cnode->AddPrimalAttr(PARAMETER_START_SHARE_CELL, cnode_->GetPrimalAttr(PARAMETER_START_SHARE_CELL));
  }
  cnode->set_scope(scope_);
  auto new_anf_node_ptr = cnode->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(new_anf_node_ptr);
  return new_anf_node_ptr;
}

AnfNodePtr GenerateGraph::NewOpInst(const OperatorName &op_name, const OperatorAttrs &attrs) {
  name_idx_++;
  ValuePtr op_prim_instance =
    CreateOpPrimtiveWithAttrs(attrs, op_name, instance_name_base_ + op_name + std::to_string(name_idx_));
  if (op_prim_instance == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:" << op_name << " NewOpInst failed";
  }
  auto value_node = NewValueNode(op_prim_instance);
  return value_node->cast<AnfNodePtr>();
}

AnfNodePtr GenerateGraph::NewOpInst(const OperatorName &op_name, const OperatorAttrs &attrs,
                                    const OperatorAttrs &prim_attrs) {
  name_idx_++;
  ValuePtr op_prim_instance =
    CreateOpPrimtiveWithAttrs(attrs, op_name, instance_name_base_ + op_name + std::to_string(name_idx_));
  if (op_prim_instance == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:" << op_name << " NewOpInst failed";
  }
  auto value_node = NewValueNode(op_prim_instance);
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_EXCEPTION_IF_NULL(prim);
  for (const auto &[name, value] : prim_attrs) {
    prim->set_attr(name, value);
  }
  return value_node->cast<AnfNodePtr>();
}

AnfNodePtr GenerateGraph::NewOpInst(const OperatorName &op_name) {
  name_idx_++;
  OperatorAttrs attrs;
  ValuePtr op_prim_instance =
    CreateOpPrimtiveWithAttrs(attrs, op_name, instance_name_base_ + std::to_string(name_idx_));
  if (op_prim_instance == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:" << op_name << " CreateOpInstance failed";
  }
  auto value_node = NewValueNode(op_prim_instance);
  return value_node->cast<AnfNodePtr>();
}
}  // namespace parallel
}  // namespace mindspore
