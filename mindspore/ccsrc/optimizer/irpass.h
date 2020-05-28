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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_H_

#include <memory>

#include "optimizer/optimizer.h"
#include "optimizer/opt.h"
#include "ir/visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
// the collection of irpass for optimie action
class OptimizeIRPassLib {
 public:
  OptimizeIRPassLib();
  ~OptimizeIRPassLib() = default;

  SubstitutionPtr arithmetic_simplify_;
  SubstitutionPtr special_op_eliminate_;
  SubstitutionPtr zero_like_fill_zero_;
  SubstitutionPtr adjust_all_reduce_mul_add_;

  //  ops eliminate
  SubstitutionPtr item_tuple_eliminate_;
  SubstitutionPtr tile_eliminate_;
  SubstitutionPtr cast_eliminate_;
  SubstitutionPtr reshape_eliminate_;
  SubstitutionPtr transpose_eliminate_;
  SubstitutionPtr reduce_eliminate_;
  SubstitutionPtr partial_eliminate_;
  SubstitutionPtr same_eliminate_;
  SubstitutionPtr check_bprop_eliminate_;
  SubstitutionPtr reset_defer_inline_;

  // Env Item Eliminate
  SubstitutionPtr new_env_get_item_;
  SubstitutionPtr add_env_get_item_;
  SubstitutionPtr env_get_set_item_;
  SubstitutionPtr incorporate_env_getitem_;
  SubstitutionPtr incorporate_env_getitem_switch_;

  // Ref eliminate
  SubstitutionPtr make_ref_eliminate_;
  SubstitutionPtr get_make_ref_eliminate_;
  SubstitutionPtr replace_refkey_by_param_;
  SubstitutionPtr replace_old_param_;

  // Branch culling
  SubstitutionPtr switch_simplify_;
  SubstitutionPtr float_tuple_getitem_switch_;
  SubstitutionPtr float_env_getitem_switch_;
  SubstitutionPtr convert_switch_replacement_;

  // AddN
  SubstitutionPtr merge_addn_;
  SubstitutionPtr addn_zero_filter_;

  // Gradient irpasses
  SubstitutionPtr expand_jprim_;
  SubstitutionPtr stop_gradient_eliminate_;
  SubstitutionPtr minmaximum_grad_;

  // inline
  SubstitutionPtr inline_;
  SubstitutionPtr replace_applicator_;
  SubstitutionPtr specialize_transform_;

  // Incorporation
  SubstitutionPtr incorporate_getitem_;
  SubstitutionPtr incorporate_getitem_switch_;
  SubstitutionPtr incorporate_call_;
  SubstitutionPtr incorporate_call_switch_;

  // virtual dataset
  SubstitutionPtr virtual_dataset_eliminate_;

  // Convert
  SubstitutionPtr print_tuple_wrapper_;
};

// the collection of irpass for resolve action
class ResolveIRPassLib {
 public:
  ResolveIRPassLib();
  ~ResolveIRPassLib() = default;

  SubstitutionPtr resolver_resolve_;
  SubstitutionPtr resolver_getattr_;
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

// Check if CNode Input 0 is Func Graph
inline bool IsCNodeGraph(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }

  auto inp0 = node->cast<CNodePtr>()->input(0);
  if (IsValueNode<FuncGraph>(inp0)) {
    return true;
  }
  return false;
}

// Check if CNode Input 0 is CNode
inline bool IsCNodeDup(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }

  auto inp0 = node->cast<CNodePtr>()->input(0);
  if (inp0 != nullptr && inp0->isa<CNode>()) {
    return true;
  }
  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_H_
