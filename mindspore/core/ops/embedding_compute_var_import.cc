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

#include "ops/embedding_compute_var_import.h"

#include <memory>

#include "mindapi/src/helper.h"
#include "mindapi/base/macros.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "ops/op_utils.h"
#include "ops/nn_ops.h"
#include "ops/primitive_c.h"
#include "ops/ops_func_impl/embedding_compute_var_import.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(EmbeddingComputeVarImport, BaseOperator);
class MIND_API EmbeddingComputeVarImportInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    (void)infer_impl_->CheckValidation(primitive, input_args);
    return infer_impl_->InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return infer_impl_->InferType(primitive, input_args);
  }

 private:
  std::shared_ptr<EmbeddingComputeVarImportFuncImpl> infer_impl_ =
    std::make_shared<EmbeddingComputeVarImportFuncImpl>();
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(EmbeddingComputeVarImport, prim::kPrimEmbeddingComputeVarImport,
                                 EmbeddingComputeVarImportInfer, false);
}  // namespace ops
}  // namespace mindspore
