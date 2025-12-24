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
#include <string>
#include "mindspore/ops/infer/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API Reshape : public InferShapeOp {
 public:
  using InferShapeOp::InferShapeOp;
  Reshape(const SymbolPtr &input, const SymbolPtr &shape) : InferShapeOp({input, shape}) {}
  ~Reshape() override = default;
  MS_DECLARE_PARENT(Reshape, InferShapeOp)
  std::string DumpText() const override;

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override;

  /// \brief find the index of unknown dim ("-1" dim)
  size_t FindUnknownDim(const SymbolPtrList &symbols);
  /// \brief remove the symbol both exist in input shape and output shape.
  void RemoveSameSymbol(SymbolPtrList *inp_symbols, SymbolPtrList *out_symbols);

  OpPtrList inner_ops_;
};
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
