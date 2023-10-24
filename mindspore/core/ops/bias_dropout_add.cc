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

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops/nn_ops.h"
#include "ops/fusion/bias_dropout_add_fusion.h"

namespace mindspore {
namespace ops {
class MIND_API BiasDropoutAddInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto x = input_args[0];
    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(x->GetShape());

    ShapeVector shape = x->GetShape()->GetShapeVector();
    auto output_shape = std::make_shared<abstract::Shape>(shape);
    return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{output_shape, output_shape});
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    size_t input_size = 3;
    auto op_name = primitive->name();
    CheckArgsSize(op_name, input_args, input_size);
    auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    auto out_type = x->GetType()->Clone();
    return std::make_shared<Tuple>(TypePtrList{out_type, out_type});
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BiasDropoutAdd, prim::kPrimBiasDropoutAdd, BiasDropoutAddInfer, false);
}  // namespace ops
}  // namespace mindspore
