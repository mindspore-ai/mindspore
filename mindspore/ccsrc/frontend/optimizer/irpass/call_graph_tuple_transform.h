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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CALL_GRAPH_TRANSFORM_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CALL_GRAPH_TRANSFORM_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/hash_set.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/graph_transform.h"

namespace mindspore {
namespace opt {
namespace irpass {
bool IsFuncGraphCallNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  return !IsValueNode<Primitive>(cnode->input(kAnfPrimitiveIndex));
}

bool FlattenArgs(const FuncGraphPtr &fg, const AnfNodePtrList &args, size_t start_idx, AnfNodePtrList *new_args) {
  bool change = false;
  for (size_t i = start_idx; i < args.size(); i++) {
    const auto &arg = args[i];
    auto abs = arg->abstract();
    if (abs == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Null abs of arg:" << arg->DebugString();
    }
    // Dynamic length sequence input can not be flattened.
    if (!IsSequenceExpandable(arg->abstract())) {
      new_args->push_back(arg);
      continue;
    }
    auto new_arg = TransformSequenceArgument(fg, arg, abs->cast<abstract::AbstractSequencePtr>());
    (void)new_args->insert(new_args->cend(), new_arg.cbegin(), new_arg.cend());
    change = true;
  }
  return change;
}

// fg(param1_sequence, param2)
// =>
// fg(param1_1, param1_2, ..., param1_n, param2)
// Transform graph call sequence inputs to flat inputs.
class GraphSequenceTransform : public AnfVisitor {
 public:
  GraphSequenceTransform() = default;
  ~GraphSequenceTransform() override = default;
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    if (!IsValueNode<FuncGraph>(node)) {
      return nullptr;
    }
    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (!FuncGraphHasConstantSequenceInput(fg)) {
      return nullptr;
    }
    fg = graph_transform_(fg, optimizer->manager());
    // Can't set abstract of the value node, otherwise the renormalize process won't be executed.
    return NewValueNode(fg);
  }

 private:
  GraphSequenceParamTransform graph_transform_;
};

// {PrimPartial, G, Sequence_Xs}
// =>
// {kPrimPartial, G, TupleGetItem{Sequence_Xs,0}, SequenceGetItem{Sequence_Xs,1}, ..., TupleGetItem{Sequence_Xs,n}}
// transform partial's sequence binding args to flat inputs.
class PartialSequenceArgTransform : public AnfVisitor {
 public:
  PartialSequenceArgTransform() = default;
  ~PartialSequenceArgTransform() override = default;
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPartial)) {
      return nullptr;
    }
    auto partial = node->cast<CNodePtr>();
    const auto &partial_inputs = partial->inputs();
    const auto &fg = partial->func_graph();
    constexpr auto kPartialFirstArgIndex = 2;
    // Put ValueNode<kPrimPartial> and ValueNode<FuncGraph> into new_inputs.
    auto new_inputs = AnfNodePtrList(partial_inputs.begin(), partial_inputs.begin() + kPartialFirstArgIndex);
    auto change = FlattenArgs(fg, partial_inputs, kPartialFirstArgIndex, &new_inputs);
    if (change) {
      auto new_partial = fg->NewCNode(new_inputs);
      new_partial->set_abstract(partial->abstract());
      return new_partial;
    }
    return nullptr;
  }
};

// {G,Sequence_Xs}
// =>
// {G, TupleGetItem{Sequence_Xs,0}, TupleGetItem{Sequence_Xs,1}, ..., TupleGetItem{Sequence_Xs,n}}
// Transform call's sequence args to flat inputs.
class CallSequenceArgTransform : public AnfVisitor {
 public:
  CallSequenceArgTransform() = default;
  ~CallSequenceArgTransform() override = default;
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    if (!IsFuncGraphCallNode(node)) {
      return nullptr;
    }

    auto call_node = node->cast<CNodePtr>();
    const auto &call_inputs = call_node->inputs();
    const auto &fg = call_node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // Put ValueNode<FuncGraph> into inputs.
    auto new_inputs = AnfNodePtrList(call_inputs.begin(), call_inputs.begin() + 1);
    auto change = FlattenArgs(fg, call_inputs, 1, &new_inputs);
    if (change) {
      auto new_call = fg->NewCNode(new_inputs);
      new_call->set_abstract(call_node->abstract());
      return new_call;
    }
    return nullptr;
  }
};

class CallGraphSequenceTransform : public OptimizerCaller {
 public:
  CallGraphSequenceTransform() {
    (void)transformers_.emplace_back(std::make_shared<GraphSequenceTransform>());
    (void)transformers_.emplace_back(std::make_shared<PartialSequenceArgTransform>());
    (void)transformers_.emplace_back(std::make_shared<CallSequenceArgTransform>());
  }
  ~CallGraphSequenceTransform() override = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    if (AlreadyHasSparseComponent(node)) {
      return nullptr;
    }
    for (auto &transform : transformers_) {
      auto new_node = (*transform)(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  bool has_sparse_tensor_ = false;
  bool AlreadyHasSparseComponent(const AnfNodePtr &node) {
    if (has_sparse_tensor_) {
      return true;
    }
    if (IsFuncGraphCallNode(node) || IsPrimitiveCNode(node, prim::kPrimPartial)) {
      auto call_node = node->cast<CNodePtr>();
      const auto &call_inputs = call_node->inputs();
      for (auto input_node : call_inputs) {
        auto abs = input_node->abstract();
        // If SparseTensor, Tuple(SparseTensor,...) or Tuple(...,(..., SparseTensor)), return false and skip this pass.
        if (abs != nullptr && ContainSparseTensor(abs)) {
          has_sparse_tensor_ = true;
          return true;
        }
      }
    } else if (IsValueNode<FuncGraph>(node)) {
      auto fg = GetValueNode<FuncGraphPtr>(node);
      if (std::any_of(fg->parameters().cbegin(), fg->parameters().cend(), ParamContainSparseTensor)) {
        has_sparse_tensor_ = true;
        return true;
      }
    }
    return false;
  }
  std::vector<OptimizerCallerPtr> transformers_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CALL_GRAPH_TRANSFORM_H_
