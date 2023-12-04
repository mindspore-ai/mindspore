/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_INFERSHAPE_OP_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_INFERSHAPE_OP_H_
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "abstract/abstract_value.h"
#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/utils.h"
#include "backend/common/graph_kernel/symbol_engine/operations/operation.h"

namespace mindspore::graphkernel::symbol {
namespace ops::infershape {
class InferShapeOp : public Operation {
 public:
  using Operation::Operation;
  ~InferShapeOp() override = default;
  MS_DECLARE_PARENT(InferShapeOp, Operation)

 protected:
  void UpdateMathInfo() override { SetPositive(output_as<ListSymbol>()); }
  void SetPositive(ListSymbol *list);
};

class RealShape : public InferShapeOp {
 public:
  struct ShapeHint {
    size_t input_index;
    SymbolPtrList cnode_inputs;
    SymbolPtrList param_inputs;
  };
  explicit RealShape(const SymbolPtr &inp, const ShapeHint *shape_hint = nullptr)
      : InferShapeOp({inp}), shape_hint_(shape_hint) {}
  ~RealShape() override = default;
  MS_DECLARE_PARENT(RealShape, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  SymbolPtr ParseTensorShape(const abstract::TensorShapePtr &shape_ptr);
  SymbolPtr ParseBaseShape(const BaseShapePtr &base_shape_ptr);
  SymbolPtr SearchPrevSymbols(ListSymbol *cur, size_t axis);
  const ShapeHint *shape_hint_{nullptr};
};

class BinElemwise : public InferShapeOp {
 public:
  BinElemwise(const SymbolPtr &lhs, const SymbolPtr &rhs) : InferShapeOp({lhs, rhs}) {}
  ~BinElemwise() override = default;
  MS_DECLARE_PARENT(BinElemwise, InferShapeOp)

  static SymbolPtrList Process(const SymbolPtrList &lhs, const SymbolPtrList &rhs, const Emitter &e, size_t shift = 0);

 protected:
  SymbolPtr Eval() override;
};

class Reduce : public InferShapeOp {
 public:
  Reduce(const SymbolPtr &inp, const SymbolPtr &axis, const SymbolPtr &keepdims, const SymbolPtr &skip_mode)
      : InferShapeOp({inp, axis, keepdims, skip_mode}) {}
  ~Reduce() override = default;
  MS_DECLARE_PARENT(Reduce, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  bool GetAxisSet(const SymbolPtr &axis, int64_t rank, bool skip_mode, HashSet<int64_t> *axis_set) const;
};

class Reshape : public InferShapeOp {
 public:
  Reshape(const SymbolPtr &input, const SymbolPtr &shape) : InferShapeOp({input, shape}) {}
  ~Reshape() override = default;
  MS_DECLARE_PARENT(Reshape, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override;
  bool ProductShape(const IListSymbol *shape);
  std::pair<SymbolPtr, int64_t> ProductData(const IListSymbol *data);
  void UpdateMathInfo() override;

  int64_t shape_size_{1LL};
  int unknown_dim_idx_{-1};
  bool shape_all_have_data_on_building_{false};
  OpPtrList inner_ops_;
};

class Transpose : public InferShapeOp {
 public:
  Transpose(const SymbolPtr &data, const SymbolPtr &perm) : InferShapeOp({data, perm}) {}
  ~Transpose() override = default;
  MS_DECLARE_PARENT(Transpose, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override;

  inline SymbolPtrList GenResult(const ListSymbol *inp, const IListSymbol *perm) const {
    MS_EXCEPTION_IF_CHECK_FAIL(inp->size() == perm->size(), "size of input and perm should be equal.");
    SymbolPtrList result(inp->size());
    for (size_t i = 0; i < result.size(); i++) {
      result[i] = inp->symbols()[LongToSize(NormAxis(perm->item(i), result.size()))];
    }
    return result;
  }
};

class MatMul : public InferShapeOp {
 public:
  MatMul(const SymbolPtr &a, const SymbolPtr &b, const SymbolPtr &transpose_a, const SymbolPtr &transpose_b,
         bool has_batch = false)
      : InferShapeOp({a, b, transpose_a, transpose_b}), has_batch_(has_batch) {}
  ~MatMul() override = default;
  MS_DECLARE_PARENT(MatMul, InferShapeOp)
  std::string name() const override { return has_batch_ ? "BatchMatMul" : "MatMul"; }

 protected:
  SymbolPtr Eval() override;
  bool has_batch_;
};

class ExpandDims : public InferShapeOp {
 public:
  ExpandDims(const SymbolPtr &input, const SymbolPtr &axis) : InferShapeOp({input, axis}) {}
  ~ExpandDims() override = default;
  MS_DECLARE_PARENT(ExpandDims, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
};

class BiasAddGrad : public InferShapeOp {
 public:
  BiasAddGrad(const SymbolPtr &x, const SymbolPtr &fmt) : InferShapeOp({x, fmt}) {}
  ~BiasAddGrad() override = default;
  MS_DECLARE_PARENT(BiasAddGrad, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

class LayerNorm : public InferShapeOp {
 public:
  LayerNorm(const SymbolPtr &x, const SymbolPtr &begin_axis) : InferShapeOp({x, begin_axis}) {}
  ~LayerNorm() override = default;
  MS_DECLARE_PARENT(LayerNorm, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

class Gather : public InferShapeOp {
 public:
  Gather(const SymbolPtr &param, const SymbolPtr &indices, const SymbolPtr &axis, const SymbolPtr &batch_dims)
      : InferShapeOp({param, indices, axis, batch_dims}) {}
  ~Gather() override = default;
  MS_DECLARE_PARENT(Gather, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

class OneHot : public InferShapeOp {
 public:
  OneHot(const SymbolPtr &indices, const SymbolPtr &depth, const SymbolPtr &axis)
      : InferShapeOp({indices, depth, axis}) {}
  ~OneHot() override = default;
  MS_DECLARE_PARENT(OneHot, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
};

class StridedSlice : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ~StridedSlice() override = default;
  MS_DECLARE_PARENT(StridedSlice, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
  SymbolPtr ComputeInferShape(const ListSymbol *x_shape, const IListSymbol *begin_v, const IListSymbol *end_v,
                              const IListSymbol *strides_v);
  SymbolPtr GetSlicingLengthForPositiveStrides(IntSymbolPtr start, IntSymbolPtr end, IntSymbolPtr strides,
                                               IntSymbolPtr x_dim);

  bool begin_mask(int bit) const { return ((begin_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  bool end_mask(int bit) const { return ((end_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  bool ellipsis_mask(int bit) const { return ((ellipsis_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  bool new_axis_mask(int bit) const { return ((new_axis_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  bool shrink_axis_mask(int bit) const { return ((shrink_axis_mask_ >> static_cast<size_t>(bit)) & 1) == 1; }
  size_t begin_mask_{0};
  size_t end_mask_{0};
  size_t ellipsis_mask_{0};
  size_t new_axis_mask_{0};
  size_t shrink_axis_mask_{0};
  const IListSymbol *out_hint_{nullptr};
};

class Switch : public InferShapeOp {
 public:
  Switch(const SymbolPtr &cond, const SymbolPtr &true_branch, const SymbolPtr &false_branch)
      : InferShapeOp({cond, true_branch, false_branch}) {}
  ~Switch() override = default;
  MS_DECLARE_PARENT(Switch, InferShapeOp)
 protected:
  SymbolPtr Eval() override;
  SymbolPtr ShapeJoin(const SymbolPtr &tb, const SymbolPtr &fb);
  SymbolPtr ItemJoin(const SymbolPtr &tb, const SymbolPtr &fb);
};
}  // namespace ops::infershape
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_INFERSHAPE_OP_H_
