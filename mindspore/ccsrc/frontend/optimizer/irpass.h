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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_H_

#include <memory>

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
// the collection of irpass for optimie action
class OptimizeIRPassLib {
 public:
  OptimizeIRPassLib();
  ~OptimizeIRPassLib() = default;

  SubstitutionPtr arithmetic_simplify_;
  SubstitutionPtr arithmetic_simplify2_;
  SubstitutionPtr special_op_eliminate_;
  SubstitutionPtr zero_like_fill_zero_;
  SubstitutionPtr adjust_all_reduce_mul_add_;
  SubstitutionPtr float_depend_g_call_;
  //  ops eliminate
  SubstitutionPtr item_tuple_or_list_eliminate_;
  SubstitutionPtr tile_eliminate_;
  SubstitutionPtr cast_eliminate_;
  SubstitutionPtr reshape_eliminate_;
  SubstitutionPtr transpose_eliminate_;
  SubstitutionPtr reduce_eliminate_;
  SubstitutionPtr partial_eliminate_;
  SubstitutionPtr same_eliminate_;
  SubstitutionPtr check_bprop_eliminate_;
  SubstitutionPtr reset_defer_inline_;
  SubstitutionPtr depend_value_elim_;
  SubstitutionPtr all_reduce_const_elim_;
  SubstitutionPtr mirror_mini_step_elim_;
  SubstitutionPtr virtual_add_elim_;
  SubstitutionPtr mini_step_allgather_replace_;

  // Env Item Eliminate
  SubstitutionPtr env_get_item_eliminate_;
  SubstitutionPtr new_env_get_item_;
  SubstitutionPtr incorporate_env_getitem_;
  SubstitutionPtr incorporate_env_getitem_bypass_recursive_;
  SubstitutionPtr incorporate_env_getitem_switch_;
  SubstitutionPtr incorporate_env_getitem_switch_layer_;

  // Ref eliminate
  SubstitutionPtr make_ref_eliminate_;
  SubstitutionPtr get_ref_param_eliminate_;
  SubstitutionPtr get_make_ref_eliminate_;
  SubstitutionPtr replace_refkey_by_param_;
  SubstitutionPtr replace_old_param_;

  // Branch culling
  SubstitutionPtr switch_simplify_;
  SubstitutionPtr float_tuple_getitem_switch_;
  SubstitutionPtr float_env_getitem_switch_;
  SubstitutionPtr convert_switch_replacement_;
  SubstitutionPtr exchange_switch_depend_value_;

  // AddN
  SubstitutionPtr merge_addn_;
  SubstitutionPtr addn_zero_filter_;

  // AccumulateNV2
  SubstitutionPtr accumulaten_eliminater_;

  // Gradient irpasses
  SubstitutionPtr minmaximum_grad_;

  // inline
  SubstitutionPtr inline_;
  SubstitutionPtr inline_without_move_;
  SubstitutionPtr replace_applicator_;
  SubstitutionPtr specialize_transform_;

  // Auto-monad related eliminaters.
  SubstitutionPtr updatestate_eliminater_;
  SubstitutionPtr switch_call_monad_eliminater_;
  SubstitutionPtr stopgrad_eliminater_;
  SubstitutionPtr load_eliminater_;

  // Incorporation
  SubstitutionPtr incorporate_getitem_set_;
  SubstitutionPtr incorporate_getitem_from_param_;
  SubstitutionPtr incorporate_call_;
  SubstitutionPtr incorporate_call_switch_;

  // virtual dataset
  SubstitutionPtr virtual_dataset_eliminate_;

  // Receive
  SubstitutionPtr receive_eliminate_;

  // Convert
  SubstitutionPtr print_tuple_wrapper_;

  // Unused parameter eliminate
  SubstitutionPtr unused_parameter_eliminate_;
  SubstitutionPtr unused_output_eliminate_;

  // tuple parameter graph transform
  SubstitutionPtr call_graph_tuple_transform_;

  // AddN eliminate
  SubstitutionPtr addn_eliminate_;

  // Fusion
  SubstitutionPtr mark_interface_fusion_;

  // RowTensor Eliminate
  SubstitutionPtr row_tensor_eliminate_;

  // RowTensorAddZerosLike Eliminate
  SubstitutionPtr row_tensor_add_zeros_like_;

  // SparseTensor Eliminate
  SubstitutionPtr sparse_tensor_eliminate_;

  // Value_Based Eliminate
  SubstitutionPtr value_based_eliminate_;

  // SwitchLayer defer inline
  SubstitutionPtr switch_layer_defer_inline_;

  // Pynative Eliminate
  SubstitutionPtr pynative_eliminate_;
};

// the collection of irpass for resolve action
class ResolveIRPassLib {
 public:
  ResolveIRPassLib();
  ~ResolveIRPassLib() = default;

  SubstitutionPtr resolver_resolve_and_getattr_;
  SubstitutionPtr resolver_resolve_;
  SubstitutionPtr resolver_getattr_;
  SubstitutionPtr resolver_getattr_resolve_;
};

class InferenceOptPrepareLib {
 public:
  InferenceOptPrepareLib();
  ~InferenceOptPrepareLib() = default;
  SubstitutionPtr grad_var_prepare_;
};

// predicate functions
inline bool IsNode(const AnfNodePtr &) { return true; }

inline bool IsCNode(const AnfNodePtr &node) {
  if (node != nullptr) {
    return node->isa<CNode>();
  }
  return false;
}

inline bool IsVNode(const AnfNodePtr &node) {
  if (node != nullptr) {
    return node->isa<ValueNode>();
  }
  return false;
}

inline bool IsParam(const AnfNodePtr &node) {
  if (node != nullptr) {
    return node->isa<Parameter>();
  }
  return false;
}

inline bool IsLoad(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  return IsPrimitiveCNode(node, prim::kPrimLoad);
}

// Check if CNode Input 0 is Func Graph
inline bool IsCNodeGraph(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }

  auto inp0 = node->cast<CNodePtr>()->input(0);
  return IsValueNode<FuncGraph>(inp0);
}

// Check if CNode Input 0 is Func Graph of graph kernel.
inline bool IsCNodeGraphKernel(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }

  auto inp0 = node->cast<CNodePtr>()->input(0);
  if (IsValueNode<FuncGraph>(inp0)) {
    auto fg = GetValueNode<FuncGraphPtr>(inp0);
    if (fg == nullptr) {
      return false;
    }
    return fg->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
  }
  return false;
}

// Check if CNode Input 0 is CNode
inline bool IsCNodeDup(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }

  auto inp0 = node->cast<CNodePtr>()->input(0);
  return (inp0 != nullptr) && inp0->isa<CNode>();
}

// check if the cnode is a switch cnode
inline bool IsCNodeSwitch(const AnfNodePtr &node) {
  if (node != nullptr) {
    if (node->isa<CNode>()) {
      return IsPrimitiveCNode(node, prim::kPrimSwitch);
    }
  }
  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_H_
