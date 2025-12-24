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
#ifndef MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_REDUCE_H_
#define MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_REDUCE_H_

#include "mindspore/ops/infer/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Reduce : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Reduce(const SymbolPtr &inp, const SymbolPtr &axis, const SymbolPtr &keepdims, const SymbolPtr &skip_mode)
      : InferShapeOp({inp, axis, keepdims, skip_mode}) {}
  ~Reduce() override = default;
  MS_DECLARE_PARENT(Reduce, InferShapeOp)

 protected:
  SymbolPtr Eval() override;
  SymbolPtr DynEval(size_t rank, bool keep_dims, SymbolPtr axis);
  bool GetAxisSet(const SymbolPtr &axis, int64_t rank, bool skip_mode, HashSet<int64_t> *axis_set) const;
};
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_REDUCE_H_
