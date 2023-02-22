/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BRANCH_CULLING_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BRANCH_CULLING_H_

#include <vector>
#include <algorithm>

#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "ir/pattern_matcher.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimSwitch, true, X, Y}
// {prim::kPrimSwitch, false, X, Y}
class SwitchSimplify : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    static const auto support_fallback = common::GetEnv("MS_DEV_ENABLE_FALLBACK");
    static const auto use_fallback = (support_fallback != "0");
    PatternNode<AnfNodePtr> cond, true_br, false_br;
    auto SwitchSimplLambda = [&node, &cond, &true_br, &false_br]() -> AnfNodePtr {
      auto value_ptr = GetValueNode(cond.GetNode(node));
      bool cond_value;
      if (value_ptr->isa<BoolImm>()) {
        cond_value = GetValue<bool>(value_ptr);
      } else if (use_fallback && value_ptr->isa<parse::InterpretedObject>()) {
        // {prim::kPrimSwitch, InterpretObject: 'True', X, Y}
        // {prim::kPrimSwitch, InterpretObject: 'False', X, Y}
        auto interpreted_obj = value_ptr->cast<parse::InterpretedObjectPtr>();
        py::object obj = interpreted_obj->obj();
        constexpr char PYTHON_MOD_PARSE_MODULE[] = "mindspore._extends.parse";
        constexpr char PYTHON_MOD_CHECK_OBJ_BOOL[] = "check_obj_bool";
        py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
        cond_value = python_adapter::CallPyModFn(mod, PYTHON_MOD_CHECK_OBJ_BOOL, obj).cast<bool>();
      } else {
        MS_LOG(EXCEPTION) << "The condition of branch must be a bool tensor value or a bool scalar value,"
                          << " not support this condition value: " << value_ptr->ToString();
      }

      MS_LOG(DEBUG) << "condition value: " << value_ptr->ToString() << ", cond: " << cond_value;
      if (cond_value) {
        return true_br.GetNode(node);
      }
      return false_br.GetNode(node);
    };

    auto IsDeterminateCondition = [](const AnfNodePtr &node) -> bool {
      static const auto support_fallback = common::GetEnv("MS_DEV_ENABLE_FALLBACK");
      static const auto use_fallback = (support_fallback != "0");
      auto &abs = node->abstract();
      bool is_interpret_object = false;
      if (use_fallback && abs != nullptr) {
        ValuePtr value = abs->BuildValue();
        MS_EXCEPTION_IF_NULL(value);
        is_interpret_object = value->isa<parse::InterpretedObject>();
      }
      return IsValueNode<BoolImm>(node) || is_interpret_object;
    };
    MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), SwitchSimplLambda,
                            cond.CheckFunc(IsDeterminateCondition, node));

    return nullptr;
  }
};

// {prim::kPrimLess, Value1, Value2}
// {prim::kPrimSwitch, Less, X, Y}
// {prim::kPrimGreater, Value1, Value2}
// {prim::kPrimSwitch, Greater, X, Y}
class CompareSwitchSimplify : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> cond, true_br, false_br;
    auto CompareSwitchSimplifyLambda = [&node, &cond, &true_br, &false_br]() -> AnfNodePtr {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto compare_cnode = cnode->input(kIndex1)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(compare_cnode);
      auto cond_tensor1 = GetValue<tensor::TensorPtr>(GetValueNode(compare_cnode->input(kIndex1)));
      auto cond_tensor2 = GetValue<tensor::TensorPtr>(GetValueNode(compare_cnode->input(kIndex2)));
      auto cond_value1 = reinterpret_cast<float *>(cond_tensor1->data_c());
      auto cond_value2 = reinterpret_cast<float *>(cond_tensor2->data_c());
      bool flag = false;
      if (IsPrimitiveCNode(compare_cnode, prim::kPrimLess) && (*cond_value1 < *cond_value2)) {
        flag = true;
      } else if (IsPrimitiveCNode(compare_cnode, prim::kPrimGreater) && (*cond_value1 > *cond_value2)) {
        flag = true;
      }
      if (flag) {
        return true_br.GetNode(node);
      }
      return false_br.GetNode(node);
    };

    auto ConstantCompareLambda = [](const AnfNodePtr &node) -> bool {
      if (!node->isa<CNode>()) {
        return false;
      }
      auto cnode = node->cast<CNodePtr>();
      if (!IsPrimitiveCNode(cnode, prim::kPrimLess) && !IsPrimitiveCNode(cnode, prim::kPrimGreater)) {
        return false;
      }
      bool has_no_value =
        std::any_of(cnode->inputs().begin() + kIndex1, cnode->inputs().end(), [](const AnfNodePtr &node) {
          if (!IsValueNode<tensor::Tensor>(node)) {
            return true;
          }
          auto value = GetValue<tensor::TensorPtr>(GetValueNode(node));
          if (value->device_address() != nullptr) {
            return true;
          }
          if (value->DataSize() > 1) {
            return true;
          }
          auto type_id = value->Dtype()->type_id();
          if (type_id != TypeId::kNumberTypeFloat32 && type_id != TypeId::kNumberTypeFloat) {
            return true;
          }
          return false;
        });
      return !has_no_value;
    };

    MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), CompareSwitchSimplifyLambda,
                            cond.CheckFunc(ConstantCompareLambda, node));

    return nullptr;
  }
};

// {prim::kPrimTupleGetItem, {prim::kPrimSwitch, X0, X1, X2}, C} =>
// {prim::kPrimSwitch, X0, {prim::kPrimTupleGetItem, X1, C}, {prim::kPrimTupleGetItem, X2, C}}
class FloatTupleGetItemSwitch : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> cond, true_br, false_br, x;
    MATCH_REPLACE_IF(node,
                     PPrimitive(prim::kPrimTupleGetItem, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), x),
                     PPrimitive(prim::kPrimSwitch, cond, PPrimitive(prim::kPrimTupleGetItem, true_br, x),
                                PPrimitive(prim::kPrimTupleGetItem, false_br, x)),
                     x.CheckFunc(IsVNode, node));
    return nullptr;
  }
};

// {prim::kPrimEnvironGet, {prim::kPrimSwitch, X1, X2, X3}, X4, X5} =>
// {prim::kPrimSwitch, X1, {prim::kPrimEnvironGet, X2, X4, X5}, {prim::kPrimEnvironGet, X3, X4, X5}}
class FloatEnvironGetSwitch : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> cond, true_br, false_br, x, x2;
    MATCH_REPLACE(node,
                  PPrimitive(prim::kPrimEnvironGet, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), x, x2),
                  PPrimitive(prim::kPrimSwitch, cond, PPrimitive(prim::kPrimEnvironGet, true_br, x, x2),
                             PPrimitive(prim::kPrimEnvironGet, false_br, x, x2)));

    return nullptr;
  }
};

namespace internal {
FuncGraphPtr TransformGraphCondTrueBranchNodes(const FuncGraphPtr &graph, const AnfNodePtr &cond);
FuncGraphPtr TransformGraphCondFalseBranchNodes(const FuncGraphPtr &graph, const AnfNodePtr &cond);
// block_nodes[0]: condition node
// block_nodes[1]: true branch node
// block_nodes[2]: false branch node
// branch_output_abs[0]: true branch abstract
// branch_output_abs[1]: false branch abstract
AnfNodePtr TransformMergeBranches(const std::vector<AnfNodePtr> &block_nodes,
                                  const std::vector<AbstractBasePtr> &branch_output_abs,
                                  const FuncGraphPtr &func_graph);
}  // namespace internal

// {{prim::kPrimSwitch, X, G1, G2}, Xs}
class ConvertSwitchReplacement {
 public:
  ConvertSwitchReplacement() = default;
  virtual ~ConvertSwitchReplacement() = default;

  bool operator()(const FuncGraphPtr &root, const OptimizerPtr &) const {
    AnfNodePtr ret = root->get_return();
    MS_EXCEPTION_IF_NULL(ret);
    std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);

    bool change = false;
    for (auto &node : all_nodes) {
      if (CheckSwitchWrapNode(node)) {
        TransformSwitchBranchReplace(node);
        change = true;
      }
    }
    return change;
  }

 private:
  // Determine whether there are graphs inside the branch graph.
  bool CheckSwitchBranch(const AnfNodePtr &node) const;
  // Determine whether node matches {{prim::kPrimSwitch, X, G1, G2}, Xs}.
  bool CheckSwitchWrapNode(const AnfNodePtr &node) const;
  // Replace switch branch.
  void TransformSwitchBranchReplace(const AnfNodePtr &node) const;
};

// {prim::kPrimSwitch, {prim::kPrimDepend, ValueNode, X}, G1, G2} ->
// {prim::kPrimDepend, {prim::kPrimSwitch, ValueNode, G1, G2}, X}
class ExchangeSwitchDependValue : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!node->isa<CNode>() || node->func_graph() == nullptr) {
      return nullptr;
    }
    ScopePtr scope = node->cast<CNodePtr>()->scope();
    ScopeGuard scope_guard(scope);

    PatternNode<AnfNodePtr> cond, true_br, false_br, v, x;
    MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSwitch, PPrimitive(prim::kPrimDepend, v, x), true_br, false_br),
                     PPrimitive(prim::kPrimDepend, PPrimitive(prim::kPrimSwitch, v, true_br, false_br), x),
                     IsVNode(v.GetNode(node)));
    return nullptr;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BRANCH_CULLING_H_
