/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/matrix_inverse.h"

#include <set>
#include <memory>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MatrixInverse, BaseOperator);

void MatrixInverse::Init(const bool adjoint) { this->set_adjoint(adjoint); }

void MatrixInverse::set_adjoint(const bool adjoint) { (void)this->AddAttr(kAdjoint, api::MakeValue(adjoint)); }

bool MatrixInverse::get_adjoint() const {
  auto value_ptr = GetAttr(kAdjoint);
  return GetValue<bool>(value_ptr);
}

class MatrixInverseInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto prim_name = primitive->name();
    auto x_shape_ptr = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
    auto x_rank = SizeToLong(x_shape.size());
    const constexpr int64_t kNumber1 = 1;
    const constexpr int64_t kNumber2 = 2;
    if (!x_shape_ptr->IsDynamic()) {
      (void)CheckAndConvertUtils::CheckInteger("x rank", x_rank, kGreaterEqual, kNumber2, prim_name);
      CheckAndConvertUtils::Check("row size", x_shape[LongToSize(x_rank - kNumber1)], kEqual,
                                  x_shape[LongToSize(x_rank - kNumber2)], prim_name);
      (void)CheckAndConvertUtils::CheckInteger("row size", x_shape[LongToSize(x_rank - kNumber1)], kGreaterEqual,
                                               kNumber2, prim_name);
      (void)CheckAndConvertUtils::CheckInteger("column size", x_shape[LongToSize(x_rank - kNumber2)], kGreaterEqual,
                                               kNumber2, prim_name);
    }
    return std::make_shared<abstract::Shape>(x_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const int64_t input_num = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim->name());
    const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
    auto infer_type = input_args[kInputIndex0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, prim->name());
    return infer_type;
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(MatrixInverse, prim::kPrimMatrixInverse, MatrixInverseInfer, false);
}  // namespace ops
}  // namespace mindspore
