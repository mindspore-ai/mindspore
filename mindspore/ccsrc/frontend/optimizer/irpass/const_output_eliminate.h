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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONST_OUTPUT_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONST_OUTPUT_ELIMINATE_H_

#include <memory>
#include <vector>
#include "ir/anf.h"
#include "ir/tensor_new.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/optimizer/irpass.h"
#include "primitive/array_ops.h"
#include "include/utils/anfalgo.h"
#include "utils/ordered_map.h"

namespace mindspore::opt::irpass {
// {a=makeTule(0, 0, 0);return a;} --> {a=makeTuple(0,0,0); b=depend(0, a); return b;}
// {a=makeTule(0, 0, 0, grad);return a;} --> {a=makeTuple(0,0,0); c=makeTuple(0, grad); b=depend(c, a); return b;}
class ConstOutputEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    auto scheduler = parallel::ParallelContext::GetInstance()->pipeline_scheduler();
    if (scheduler == parallel::ZBV) {
      return nullptr;
    }
    auto flag = IsEliminate(node);
    if (!flag) {
      return nullptr;
    }

    MS_LOG(INFO) << "const output eliminater process";

    auto fg = GetValueNode<FuncGraphPtr>(node);
    MS_EXCEPTION_IF_NULL(fg);
    auto output = fg->output();
    const size_t min_input_size = 3;
    const auto &inputs = output->cast<CNodePtr>()->inputs();
    if (inputs.size() < min_input_size) {
      MS_LOG(INFO) << "maketuple input size small, size=" << inputs.size();
      return nullptr;
    }

    if (!grad_mode_) {
      const auto const_data = Tensor0Builder();
      new_out_abstract_ = const_data->ToAbstract();
      auto new_value_node = NewValueNode(const_data);
      new_value_node->set_abstract(new_out_abstract_);

      auto depend = fg->NewCNode({NewValueNode(prim::kPrimDepend), new_value_node, output});
      MS_EXCEPTION_IF_NULL(depend);
      depend->set_abstract(new_out_abstract_);
      fg->set_output(depend);
    } else {
      // Zeros + grad
      std::vector<AnfNodePtr> zero_inputs(inputs.begin() + 1, inputs.end() - 1);
      auto grad_input = inputs.back();

      std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
      make_tuple_inputs.insert(make_tuple_inputs.end(), zero_inputs.begin(), zero_inputs.end());
      auto tuple_zero_node_abstract = GetTupleAbstract(zero_inputs);
      auto tuple_zero_node = fg->NewCNode(make_tuple_inputs);
      tuple_zero_node->set_abstract(tuple_zero_node_abstract);

      const auto const_data = Tensor0Builder();
      auto abstract_tensor = const_data->ToAbstract();
      auto new_value_node = NewValueNode(const_data);
      new_value_node->set_abstract(abstract_tensor);
      auto depend = fg->NewCNode({NewValueNode(prim::kPrimDepend), new_value_node, tuple_zero_node});
      depend->set_abstract(abstract_tensor);

      new_out_abstract_ = GetTupleAbstract({new_value_node, grad_input});
      auto new_out = fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), depend, grad_input});
      new_out->set_abstract(new_out_abstract_);
      fg->manager()->Replace(output, new_out);
    }
    fg->return_node()->set_abstract(new_out_abstract_);

    (void)DoProcess(fg, true);

    return nullptr;
  }

 private:
  bool grad_mode_ = false;
  size_t grad_index_ = 0;
  AbstractBasePtr new_out_abstract_ = nullptr;
  OrderedMap<TypeId, AnfNodePtr> type_node_pair_;

  void Reset() {
    grad_mode_ = false;
    grad_index_ = 0;
    new_out_abstract_ = nullptr;
    type_node_pair_.clear();
  }

  AbstractBasePtr GetTupleAbstract(const std::vector<AnfNodePtr> &inputs) const {
    AbstractBasePtrList new_sep_abstracts;
    for (const auto &input : inputs) {
      new_sep_abstracts.push_back(input->abstract());
    }

    return std::make_shared<abstract::AbstractTuple>(new_sep_abstracts);
  }

  bool IsTupleAllZero(const AnfNodePtr &node) {
    auto tuple = node->abstract()->cast<abstract::AbstractTuplePtr>();
    if (tuple == nullptr) {
      return false;
    }
    size_t element_cnt = 0;
    for (const auto &element : tuple->elements()) {
      element_cnt++;
      if (element->isa<abstract::AbstractTensor>()) {
        const auto &tensor_abstract = element->cast<abstract::AbstractTensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor_abstract);
        auto dim_zero = tensor_abstract->BuildShape()->IsDimZero();
        auto value_any = tensor_abstract->BuildValue()->isa<ValueAny>();
        if (!value_any) {
          return false;
        }

        if (element_cnt == tuple->elements().size()) {
          grad_mode_ = dim_zero ? false : true;
          grad_index_ = tuple->elements().size() - 1;
          continue;
        }

        if (!dim_zero) {
          return false;
        }
        continue;
      }

      if (!element->isa<abstract::AbstractScalar>()) {
        return false;
      }
      const auto &scalar_abstract = element->cast<abstract::AbstractScalarPtr>();
      MS_EXCEPTION_IF_NULL(scalar_abstract);
      auto abs_value = scalar_abstract->BuildValue();
      MS_EXCEPTION_IF_NULL(abs_value);
      auto abs_int32 = dyn_cast<Int32Imm>(abs_value);
      if (abs_int32 != nullptr) {
        if (abs_int32->value() != 0) {
          return false;
        }
        continue;
      }

      auto abs_int64 = dyn_cast<Int64Imm>(abs_value);
      if (abs_int64 == nullptr) {
        return false;
      }

      if (abs_int64->value() != 0) {
        return false;
      }
    }

    // If only one zero, no need to eliminate
    constexpr size_t kTwoElements = 2;
    constexpr size_t kOneElement = 1;
    if (grad_mode_) {
      return tuple->elements().size() > kTwoElements;
    } else {
      return tuple->elements().size() > kOneElement;
    }
  }

  bool IsEliminate(const AnfNodePtr &node) {
    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg == nullptr) {
      return false;
    }
    auto output = fg->output();
    if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
      return false;
    }

    // Check whether the output is 0
    if (!IsTupleAllZero(output)) {
      return false;
    }

    // Check output users
    return DoProcess(fg);
  }

  bool DoProcess(const FuncGraphPtr &func, bool is_replace = false) {
    MS_EXCEPTION_IF_NULL(func);
    auto &fg_use_map = func->func_graph_cnodes_index();
    if (fg_use_map.empty()) {
      return false;
    }

    for (auto &fg_use : fg_use_map) {
      auto use_node = fg_use.first->first->cast<CNodePtr>();
      if (!IsPrimitiveCNode(use_node, prim::kPrimMakeTuple)) {
        return false;
      }
      auto use_node_graph = use_node->func_graph();
      if (use_node_graph->output() != use_node) {
        return false;
      }
      auto &fg_use_map_sub = use_node_graph->func_graph_cnodes_index();
      auto mng_sub = use_node_graph->manager();
      for (auto &fg_use_sub : fg_use_map_sub) {
        auto fg_use_node = fg_use_sub.first->first->cast<CNodePtr>();
        if (fg_use_node == nullptr) {
          return false;
        }
        auto users_sub = mng_sub->node_users()[fg_use_node];

        auto ret = SubUsersProcess(users_sub, is_replace);
        if (!ret) {
          return false;
        }
      }
    }

    return true;
  }

  bool SubUsersProcess(const AnfNodeIndexSet &users, bool is_replace) {
    for (auto &user : users) {
      if (IsPrimitiveCNode(user.first, prim::kPrimDepend) && user.second == kDependAttachNodeIndex) {
        continue;
      }

      if (!IsPrimitiveCNode(user.first, prim::kPrimTupleGetItem)) {
        return false;
      }

      auto index = common::AnfAlgo::GetTupleGetItemOutIndex(user.first->cast<CNodePtr>());
      if (index != kIndex1) {
        continue;
      }

      auto mng_sub = user.first->func_graph()->manager();
      auto users_sub = mng_sub->node_users()[user.first];
      for (auto &user_sub : users_sub) {
        if (is_replace) {
          user_sub.first->set_abstract(new_out_abstract_);
        }

        auto ret = ConstNodeRealUserProcess(user_sub.first, user_sub.first->func_graph(), is_replace);
        if (!ret) {
          return false;
        }
      }
    }

    return true;
  }

  bool ConstNodeRealUserProcess(const AnfNodePtr &node, const FuncGraphPtr &func, bool is_replace) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(func);

    auto mng = func->manager();
    auto users = mng->node_users()[node];
    if (users.empty()) {
      return false;
    }

    for (auto &user : users) {
      if (IsPrimitiveCNode(user.first, prim::kPrimDepend) && user.second == kDependAttachNodeIndex) {
        continue;
      }

      if (!IsPrimitiveCNode(user.first, prim::kPrimTupleGetItem)) {
        return false;
      }
      AnfNodePtr new_node;
      const auto element = user.first->abstract();
      if (element->isa<abstract::AbstractTensor>()) {
        const auto &tensor_abstract = element->cast<abstract::AbstractTensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor_abstract);
        const auto &tensor_type = tensor_abstract->element()->BuildType();
        auto iter = type_node_pair_.find(tensor_type->type_id());
        bool find = iter != type_node_pair_.end();
        if (!find) {
          const auto const_data = Tensor0Builder(tensor_type);
          new_out_abstract_ = const_data->ToAbstract();
          new_node = NewValueNode(const_data);
          new_node->set_abstract(const_data->ToAbstract());
          type_node_pair_.emplace(tensor_type->type_id(), new_node);
        } else {
          new_node = iter->second;
        }
      }

      if (!is_replace) {
        // Check
        auto ret = RealUserCallerCheck(user.first, user.first->func_graph());
        if (!ret) {
          return false;
        }
      }

      if (is_replace) {
        // Real caller
        if (!grad_mode_) {
          auto depend = func->NewCNode({NewValueNode(prim::kPrimDepend), new_node, node});
          MS_EXCEPTION_IF_NULL(depend);
          depend->set_abstract(new_node->abstract());
          mng->Replace(user.first, depend);
        } else {
          auto index = common::AnfAlgo::GetTupleGetItemOutIndex(user.first->cast<CNodePtr>());
          auto real_input = common::AnfAlgo::GetTupleGetItemRealInput(user.first->cast<CNodePtr>());
          size_t new_index = index == grad_index_ ? 1 : 0;
          if (new_index == 1) {
            auto new_index_value = NewValueNode(MakeValue(SizeToLong(new_index)));
            auto new_grad_node = func->NewCNode({NewValueNode(prim::kPrimTupleGetItem), real_input, new_index_value});
            new_grad_node->set_abstract(user.first->abstract());
            mng->Replace(user.first, new_grad_node);
          } else {
            auto depend = func->NewCNode({NewValueNode(prim::kPrimDepend), new_node, node});
            MS_EXCEPTION_IF_NULL(depend);
            depend->set_abstract(new_node->abstract());
            mng->Replace(user.first, depend);
          }
        }
      }
    }

    return true;
  }

  tensor::TensorPtr Tensor0Builder(TypePtr element_type = nullptr) const {
    if (element_type != nullptr) {
      return tensor::from_scalar(0.0, element_type);
    } else {
      return tensor::from_scalar(0.0);
    }
  }

  bool RealUserCallerCheck(const AnfNodePtr &node, const FuncGraphPtr &func) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(func);

    auto mng = func->manager();
    auto &users = mng->node_users()[node];

    if (users.empty()) {
      return false;
    }

    for (auto &user : users) {
      if (IsPrimitiveCNode(user.first, prim::kPrimDepend) && user.second == kDependAttachNodeIndex) {
        continue;
      }

      if (IsPrimitiveCNode(user.first, prim::kPrimDepend) && user.second == kRealInputIndexInDepend && grad_mode_) {
        continue;
      }

      if (IsPrimitiveCNode(user.first, prim::kPrimSend) && grad_mode_) {
        continue;
      }

      if (!IsPrimitiveCNode(user.first, prim::kPrimMakeTuple)) {
        return false;
      }

      MS_EXCEPTION_IF_NULL(user.first->abstract());
      auto tuple = user.first->abstract()->cast<abstract::AbstractTuplePtr>();
      if (!tuple) {
        return false;
      }

      // Check whether the element of tuple is empty tensor
      for (const auto &element : tuple->elements()) {
        if (!element->isa<abstract::AbstractTensor>()) {
          return false;
        }

        const auto &tensor_abstract = element->cast<abstract::AbstractTensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor_abstract);
        if (!(tensor_abstract->BuildShape()->IsDimZero() && tensor_abstract->BuildValue()->isa<ValueAny>())) {
          return false;
        }
      }
    }

    return true;
  }
};
}  // namespace mindspore::opt::irpass

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONST_OUTPUT_ELIMINATE_H_
