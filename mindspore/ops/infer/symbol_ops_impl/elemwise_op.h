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
#ifndef MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_ELEMWISE_OP_H_
#define MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_ELEMWISE_OP_H_

#include "mindspore/ops/infer/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API ElemwiseBinop : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  ElemwiseBinop(const SymbolPtr &lhs, const SymbolPtr &rhs) : InferShapeOp({lhs, rhs}) {}
  ~ElemwiseBinop() override = default;
  MS_DECLARE_PARENT(ElemwiseBinop, InferShapeOp)

  static SymbolPtrList Process(const SymbolPtrList &lhs, const SymbolPtrList &rhs, const OperationEmitter &e,
                               size_t shift = 0);

 protected:
  SymbolPtr Eval() override;
};
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_ELEMWISE_OP_H_
