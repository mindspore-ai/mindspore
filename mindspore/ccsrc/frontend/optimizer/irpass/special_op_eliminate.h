/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIAL_OP_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIAL_OP_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "include/securec.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "primitive/structure_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "frontend/optimizer/pattern_matcher.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/prim_eliminate.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/comm_manager.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/jit/ps/parse/resolve.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
class SpecialOpEliminater : public OptimizerCaller {
 public:
  SpecialOpEliminater();
  ~SpecialOpEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;

 private:
  OptimizerCallerPtr insert_gradient_of_, stop_gradient_, hook_backward_, print_shape_type_, mirror_, virtual_div_,
    mutable_;
  std::vector<OptimizerCallerPtr> eliminaters_{};
};

// {PrimVirtualDataset, X} -> X
// {PrimVirtualDataset, Xs} -> {prim::kPrimMakeTuple, Xs}
class VirtualDatasetEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// {prim::kPrimVirtualOutput, X} -> X
class VirtualOutputEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// {prim::kPrimAShardIdentity, X} -> X
class AShardIdentityEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// {ParallelVirtualNode, X, Y...} -> X
class ParallelVirtualNodeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// {prim::kPrimSameTypeShape, X, Y} -> X
class SameEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
  void Visit(const AnfNodePtr &node) override;

 private:
  AnfNodePtr x_{nullptr};
};

// {prim::kPrimCheckBprop, X, Y} -> X
class CheckBpropEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;

 private:
  AnfNodePtr x_{nullptr};
};

// {prim::DumpGradient, X, Y} -> Y
class DumpGradientEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// {prim::kPrimMiniStepAllGather, X, Z} -> {prim::kPrimAllGather, X}
class MiniStepAllGatherPass : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// {prim::kPrimMicroStepAllGather, X, Z} -> {prim::kPrimAllGather, X}
class MicroStepAllGatherPass : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// Reset defer_inline flag
class ResetDeferInline : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// {PrimZerosLike, Y} ->
// {PrimFill, {PrimDType, Y}, {PrimShape, Y}, 0}
class ZeroLikeFillZero : public AnfVisitor {
 public:
  ZeroLikeFillZero();
  ~ZeroLikeFillZero() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;

 private:
  AnfNodePtr y_{nullptr};
  PrimitivePtr PrimFill_, PrimShape_, PrimDType_;
};

// {prim::kPrimDepend, X, ValueCond} -> X
// {prim::kPrimDepend, {prim, X, ...}, X} -> {prim, X, ...}
class DependValueElim : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  bool IsUsedByOther(const AnfNodePtr &node, const AnfNodePtr &user_node) const;
};

// {{prim:getattr, {prim::resolve, SymbolStr, C}, zeros_like}, Xy} ->Tensor(0, shape(Xy))
// {prim:getattr, {prim::resolve, SymbolStr, zeros_like}, Xy} ->Tensor(0, shape(Xy))
// {{prim::resolve, CommonOPS, getitem}, (tensor0, tensor1,...), 0} -> tensor0
class PynativeEliminater : public OptimizerCaller {
  bool CheckNameSpaceVNode(const AnfNodePtr &node, const std::string &str_value) const;

  bool CheckSymbolVNode(const AnfNodePtr &node, const std::string &str_value) const;
  bool CheckStrVNode(const AnfNodePtr &node, const std::string &str_value) const;

  ValuePtr FillGetItem(const ValuePtr &value, const ValuePtr &idx, const AnfNodePtr &node) const;

  ValuePtr FillZero(const ValuePtr &value, const AnfNodePtr &node);

 private:
  AnfNodePtr OperatorHandle1(const PatternNode<AnfNodePtr> &arg, const AnfNodePtr &node);

  AnfNodePtr OperatorHandle2(const PatternNode<AnfNodePtr> &arg, const AnfNodePtr &node);

  void OperatorHandle3(const std::vector<PatternNode<AnfNodePtr>> &args, const AnfNodePtr &node) const;

  AnfNodePtr OperatorHandle4(const PatternNode<AnfNodePtr> &arg, const PatternNode<AnfNodePtr> &arg1,
                             const AnfNodePtr &node) const;

 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

class AllReduceConstElim : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// This pattern introduced by Depend(CollectCNodeWithIsolateNodes) in program_specialize.cc
// {{prim::kPrimDepend, X, Y}, Xs}->{prim::kPrimDepend, {X, Xs}, Y}
class FloatDependGCall : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

class PynativeGradjitPrimitivePyEliminater : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SPECIAL_OP_ELIMINATE_H_
