/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GET_GRAD_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GET_GRAD_ELIMINATE_H_

#include <vector>
#include <algorithm>
#include <string>

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
class GetGradEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimGetGrad, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimGetGrad, {IsCNode, IsCNode})(node);
    FindGradByNameOrId(grad_tuple_);
    if (result_ == nullptr) {
      MS_LOG(EXCEPTION) << "Can not find the gradient for position or Parameter provided";
    }
    return result_;
  }

  void FindGradByNameOrId(const CNodePtr &node) {
    if (got_) {
      return;
    }
    ValueNodePtr name_or_id = nullptr;
    AnfNodePtrList inputs;
    constexpr int64_t name_index = 1;
    constexpr int64_t value_index = 2;
    if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      inputs = node->inputs();
      if (inputs.size() == 1) {
        return;
      }
      auto first = inputs[name_index];
      name_or_id = first->cast<ValueNodePtr>();
    } else {
      return;
    }
    if (name_or_id != nullptr && name_or_id->value()->isa<Int64Imm>()) {
      if (GetValueNode<Int64ImmPtr>(name_or_id)->value() == id_) {
        result_ = inputs[value_index];
        got_ = true;
        return;
      }
    } else if (name_or_id != nullptr && name_or_id->value()->isa<StringImm>()) {
      if (GetValueNode<StringImmPtr>(name_or_id)->value() == name_) {
        result_ = inputs[value_index];
        got_ = true;
        return;
      }
    } else {
      for (size_t i = 1; i < inputs.size(); i++) {
        CNodePtr child = inputs[i]->cast<CNodePtr>();
        FindGradByNameOrId(child);
      }
      return;
    }
  }

  void Visit(const CNodePtr &node) override {
    if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      grad_tuple_ = node;
    } else if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
      auto &input = node->inputs();
      MS_EXCEPTION_IF_NULL(input[1]);
      if (input[1]->isa<Parameter>()) {
        auto para = dyn_cast<Parameter>(input[1]);
        name_ = para->name();
      } else if (IsValueNode<tensor::Tensor>(input[1])) {
        auto tensor = GetValueNode<tensor::TensorPtr>(input[1]);
        auto param_inf = tensor->param_info();
        if (param_inf != nullptr) {
          name_ = param_inf->name();
        }
      } else if (IsPrimitiveCNode(input[1], prim::kPrimListGetItem)) {
        auto cnode = input[1]->cast<CNodePtr>();
        auto inner = cnode->inputs();
        constexpr auto number_two = 2;
        MS_EXCEPTION_IF_NULL(inner[1]);
        MS_EXCEPTION_IF_NULL(inner[number_two]);
        int64_t pos = GetValueNode<Int64ImmPtr>(inner[number_two])->value();
        auto list = GetValueNode<ValueListPtr>(inner[1])->value();
        MS_EXCEPTION_IF_NULL(list[pos]);
        auto tensor = list[pos]->cast<tensor::TensorPtr>();
        auto param_inf = tensor->param_info();
        MS_EXCEPTION_IF_NULL(param_inf);
        name_ = param_inf->name();
      } else {
        MS_LOG(EXCEPTION) << "suppose to get tensor or parameter, but got: " << input[1]->DebugString();
      }
    }
  }

  void Visit(const ValueNodePtr &node) override { id_ = GetValueNode<Int64ImmPtr>(node)->value(); }

  void Reset() {
    id_ = -1;
    name_ = "";
    grad_tuple_ = nullptr;
    result_ = nullptr;
    got_ = false;
  }

 private:
  int64_t id_;
  std::string name_;
  CNodePtr grad_tuple_{nullptr};
  AnfNodePtr result_{nullptr};
  bool got_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_TRANSPOSE_ELIMINATE_H_
