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

#include "frontend/parallel/graph_util/generate_graph.h"

#include <algorithm>
#include <memory>
#include <string>

#include "pipeline/jit/parse/python_adapter.h"
#include "utils/convert_utils_py.h"

using mindspore::tensor::Tensor;

namespace mindspore {
namespace parallel {
std::string GetOpPythonPath(const OperatorName &op_name) {
  // almost all ops are defined in two main paths
  const std::string ops_module = OP_PATH;
  const std::string inner_ops_module = INNER_OP_PATH;
  const std::string functional_op_module = FUNCTIONAL_OP_PATH;
  py::module mod = py::module::import(common::SafeCStr(ops_module));
  py::module inner_mod = py::module::import(common::SafeCStr(inner_ops_module));
  py::module functional_mod = py::module::import(common::SafeCStr(functional_op_module));

  if (py::hasattr(inner_mod, common::SafeCStr(op_name))) {
    return inner_ops_module;
  }
  if (py::hasattr(mod, common::SafeCStr(op_name))) {
    return ops_module;
  }
  if (!py::hasattr(functional_mod, common::SafeCStr(op_name))) {
    MS_LOG(EXCEPTION) << ops_module << " and " << inner_ops_module << " and " << functional_op_module
                      << " don't have op:" << op_name;
  }
  return functional_op_module;
}

ValuePtr CreatOpInstance(const OperatorAttrs &attrs, const OperatorName &op_name, const std::string &instance_name) {
  std::string op_path = GetOpPythonPath(op_name);
  py::module mod = py::module::import(common::SafeCStr(op_path));
  if (!py::hasattr(mod, common::SafeCStr(op_name))) {
    MS_LOG(ERROR) << "Failure: op_path:" << op_path << " don't have attr " << op_name;
    return nullptr;
  }
  std::vector<py::object> arg_list;
  (void)std::transform(attrs.begin(), attrs.end(), std::back_inserter(arg_list),
                       [](const Attr &attr) { return ValuePtrToPyData(attr.second); });
  py::object obj =
    parse::python_adapter::CallPyFn(GET_OP_FUNCTION_PATH, GET_OP_FUNCTION, op_name, op_path, instance_name, arg_list);
  ValuePtr op_instance = nullptr;
  bool succ = parse::ConvertData(obj, &op_instance);
  if (!succ) {
    MS_LOG(ERROR) << "Failure:get Python op " << op_path << " from " << op_name << " fail";
    return nullptr;
  }
  return op_instance;
}

AnfNodePtr ValuePtrToAnfNodePtr(const ValuePtr &value_ptr) {
  auto value_node = NewValueNode(value_ptr);
  MS_EXCEPTION_IF_NULL(value_node);
  return value_node->cast<AnfNodePtr>();
}

static std::unordered_map<int64_t, AnfNodePtr> int_tensor_map = {};
AnfNodePtr CreateInt32Tensor(int64_t value) {
  auto it = int_tensor_map.find(value);
  if (it != int_tensor_map.end()) {
    return it->second;
  }
  mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<tensor::Tensor>(value, kInt32);
  ValuePtr value_ptr = MakeValue(tensor_ptr);
  auto anf_node_ptr = ValuePtrToAnfNodePtr(value_ptr);
  int_tensor_map[value] = anf_node_ptr;
  return anf_node_ptr;
}

AnfNodePtr CreatTypeInt(int64_t value) {
  ValuePtr value_ptr = MakeValue(std::make_shared<Int>(value));
  return ValuePtrToAnfNodePtr(value_ptr);
}

AnfNodePtr CreatInt64Imm(int64_t value) {
  ValuePtr value_ptr = MakeValue(std::make_shared<Int64Imm>(value));
  return ValuePtrToAnfNodePtr(value_ptr);
}

std::string GetInstanceNameByCNode(const CNodePtr &cnode) {
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (!prim) {
    MS_LOG(EXCEPTION) << "The first input of the cnode is not a PrimitivePtr.";
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
  CNodePtr cnode = func_graph_->NewCNode(inputs);  // using NewCNode to create anfnode
  MS_EXCEPTION_IF_NULL(cnode);
  cnode->set_scope(scope_);
  if (inputs.size() < 2) {
    MS_LOG(EXCEPTION) << "inputs.size() must be more than 1";
  }
  (void)manager_->Replace(inputs.at(1), cnode);  // using Replace function to insert cnode after inputs[0]
  auto new_anf_node_ptr = cnode->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(new_anf_node_ptr);
  return new_anf_node_ptr;
}

AnfNodePtr GenerateGraph::NewOpInst(const OperatorName &op_name, const OperatorAttrs &attrs) {
  name_idx_++;
  ValuePtr pyop_instance = CreatOpInstance(attrs, op_name, instance_name_base_ + op_name + std::to_string(name_idx_));
  if (pyop_instance == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:" << op_name << " CreatOpInstance failed";
  }
  auto value_node = NewValueNode(pyop_instance);
  return value_node->cast<AnfNodePtr>();
}

AnfNodePtr GenerateGraph::NewOpInst(const OperatorName &op_name) {
  name_idx_++;
  OperatorAttrs attrs;
  ValuePtr pyop_instance = CreatOpInstance(attrs, op_name, instance_name_base_ + std::to_string(name_idx_));
  if (pyop_instance == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:" << op_name << " CreatOpInstance failed";
  }
  auto value_node = NewValueNode(pyop_instance);
  return value_node->cast<AnfNodePtr>();
}
}  // namespace parallel
}  // namespace mindspore
