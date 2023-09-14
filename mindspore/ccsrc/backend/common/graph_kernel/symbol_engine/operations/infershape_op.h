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

#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/utils.h"
#include "backend/common/graph_kernel/symbol_engine/operations/operation.h"

namespace mindspore::graphkernel::symbol {
namespace ops::infershape {
inline int64_t NormAxis(int64_t axis, size_t rank) { return axis >= 0 ? axis : axis + static_cast<int64_t>(rank); }
class RealShape : public Operation {
 public:
  explicit RealShape(const SymbolPtr &inp) : Operation({inp}) {}
  ~RealShape() override = default;
  MS_DECLARE_PARENT(RealShape, Operation)

 protected:
  SymbolPtr Eval() override;
};

class BinElemwise : public Operation {
 public:
  BinElemwise(const SymbolPtr &lhs, const SymbolPtr &rhs) : Operation({lhs, rhs}) {}
  ~BinElemwise() override = default;
  MS_DECLARE_PARENT(BinElemwise, Operation)

  static SymbolPtrList Process(const SymbolPtrList &lhs, const SymbolPtrList &rhs, const Emitter &e, size_t shift = 0);

 protected:
  SymbolPtr Eval() override;
};

class Reduce : public Operation {
 public:
  Reduce(const SymbolPtr &inp, const SymbolPtr &axis, const SymbolPtr &keepdims, const SymbolPtr &skip_mode)
      : Operation({inp, axis, keepdims, skip_mode}) {}
  ~Reduce() override = default;
  MS_DECLARE_PARENT(Reduce, Operation)

 protected:
  SymbolPtr Eval() override;
  bool GetAxisSet(const SymbolPtr &axis, int64_t rank, bool skip_mode, HashSet<int64_t> *axis_set) const;
};

class Reshape : public Operation {
 public:
  Reshape(const SymbolPtr &input, const SymbolPtr &shape) : Operation({input, shape}) {}
  ~Reshape() override = default;
  MS_DECLARE_PARENT(Reshape, Operation)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override;
  bool ProductShape(const IListSymbol *shape);

  int64_t shape_size_{1LL};
  int unknown_dim_idx_{-1};
  bool shape_all_have_data_on_building_{false};
  OpPtrList inner_ops_;
};

class Transpose : public Operation {
 public:
  Transpose(const SymbolPtr &data, const SymbolPtr &perm) : Operation({data, perm}) {}
  ~Transpose() override = default;
  MS_DECLARE_PARENT(Transpose, Operation)

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

class MatMul : public Operation {
 public:
  MatMul(const SymbolPtr &a, const SymbolPtr &b, const SymbolPtr &transpose_a, const SymbolPtr &transpose_b,
         bool has_batch = false)
      : Operation({a, b, transpose_a, transpose_b}), has_batch_(has_batch) {}
  ~MatMul() override = default;
  MS_DECLARE_PARENT(MatMul, Operation)
  std::string name() const override { return has_batch_ ? "BatchMatMul" : "MatMul"; }

 protected:
  SymbolPtr Eval() override;
  bool has_batch_;
};
}  // namespace ops::infershape
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_INFERSHAPE_OP_H_
