/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/op_def_utils.h"
#include <vector>
#include <string>
#include <memory>

#include <set>

#include "ir/primitive.h"
#include "ir/func_graph.h"
#include "ops/op_def.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore::ops {
constexpr auto kAttrMeOpName = "me_op_name";

std::set<int64_t> GetInputDependValueList(const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_prim);
  std::set<int64_t> depend_list;
  mindspore::ops::OpDefPtr op_def = GetOpDef(op_prim->name());
  if (op_def == nullptr) {
    // Use old Primitive infer.
    auto op_infer_opt = abstract::GetPrimitiveInferImpl(op_prim);
    if (!op_infer_opt.has_value()) {
      if (op_prim->HasAttr(kAttrMeOpName)) {
        auto ori_prim_name = GetValue<std::string>(op_prim->GetAttr(kAttrMeOpName));
        op_infer_opt = abstract::GetPrimitiveInferImpl(std::make_shared<Primitive>(ori_prim_name));
      }
    }
    if (op_infer_opt.has_value()) {
      auto op_infer = op_infer_opt.value().Get();
      if (op_infer != nullptr && depend_list.empty()) {
        depend_list = op_infer->GetValueDependArgIndices();
      }
    }
    return depend_list;
  }

  depend_list = op_def->func_impl_.GetValueDependArgIndices();
  if (!depend_list.empty()) {
    return depend_list;
  }
  // if not defined the GetValueDependArgIndices() func in infer, consider all the no-Tensor
  // input as value depend.
  auto args = op_def->args_;
  for (size_t i = 0; i < args.size(); i++) {
    if (args[i].arg_dtype_ != mindspore::ops::OP_DTYPE::DT_TENSOR &&
        args[i].arg_dtype_ != mindspore::ops::OP_DTYPE::DT_TUPLE_TENSOR &&
        args[i].arg_dtype_ != mindspore::ops::OP_DTYPE::DT_LIST_TENSOR) {
      (void)depend_list.insert(i);
    }
  }
  return depend_list;
}

size_t GetInputIndexByName(const std::string &op_name, const std::string &input_name) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(INFO) << op_name << " is not defined in opdef.";
    return SIZE_MAX;
  }
  auto ks_iter = op_def->indexes_.find(input_name);
  if (ks_iter != op_def->indexes_.end()) {
    size_t index = ks_iter->second;
    MS_LOG(DEBUG) << "Find " << input_name << "in " << index << "th input of OP " << op_name;
    return index;
  }
  MS_LOG(INFO) << "Not Find " << input_name << "in OP " << op_name;
  return SIZE_MAX;
}

std::string GetInputNameByIndex(const std::string &op_name, size_t index) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    return "";
  }
  if (index >= op_def->args_.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Get input name by index out of range, index: " << index
                               << ", size: " << op_def->args_.size() << ", op name: " << op_name;
  }
  auto input = op_def->args_[index];
  return input.arg_name_;
}

bool HasOpDef(const std::string &op_name) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  return op_def != nullptr;
}

size_t GetOpInputsNum(const std::string &op_name) {
  mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(INFO) << op_name << " is not defined in opdef.";
    return SIZE_MAX;
  }
  return op_def->indexes_.size();
}

// This is used to convert arg with 'prim_init' of cnode convert to attr of primitive.
// CNode in new mindir can be converted to old mindir by this function.
// For example, {PrimAvgPool, x, kernel_size, strides, pad_mode, data_format} =>
//              {PrimAvgPool, x}
CNodePtr ConvertArgsToAttr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto op_def = mindspore::ops::GetOpDef(prim_name);
  if (op_def == nullptr) {
    MS_LOG(DEBUG) << "Prim:" << prim->ToString()
                  << "is not a primitive defined in yaml, cannot convert args to attr, cnode:" << cnode->DebugString();
    return nullptr;
  }
  prim = std::make_shared<Primitive>(prim_name);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(prim)};
  for (size_t arg_index = 0; arg_index < op_def->args_.size(); ++arg_index) {
    auto arg = op_def->args_[arg_index];
    if (!arg.as_init_arg_) {
      // origin is input , put the node input into new node inputs vector
      (void)new_node_inputs.emplace_back(cnode->input(arg_index + 1));
      continue;
    }

    auto arg_input_node = cnode->input(arg_index + 1);
    if (!arg_input_node->isa<ValueNode>()) {
      // arg is not ValueNode, Network has dynamic args, not support
      MS_LOG(INTERNAL_EXCEPTION) << "Node " << cnode->DebugString() << " with arg " << arg_input_node->DebugString()
                                 << " is dynamic, not supported now.";
      continue;
    }
    auto arg_value_node = arg_input_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(arg_value_node);
    auto arg_value = arg_value_node->value();
    MS_EXCEPTION_IF_NULL(arg_value);
    prim->AddAttr(arg.arg_name_, arg_value);
  }

  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto new_node = func_graph->NewCNode(new_node_inputs);
  new_node->set_abstract(cnode->abstract());
  new_node->set_fullname_with_scope(cnode->fullname_with_scope());
  return new_node;
}
}  // namespace mindspore::ops
