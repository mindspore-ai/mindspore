/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "ops/cholesky_inverse_.h"

#include <map>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CholeskyInverseInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kDimNum = 2;
  auto op_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  if (x_shape.size() != kDimNum && !IsDynamic(x_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', The dimension of x must be equal to 2, but got: " << x_shape.size() << ".";
  }
  if (!IsDynamic(x_shape) && x_shape[x_shape.size() - 1] != x_shape[x_shape.size() - kDimNum]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', input must be square matrix, "
                             << "while row is " << x_shape[x_shape.size() - kDimNum] << ", col is "
                             << x_shape[x_shape.size() - 1];
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr CholeskyInverseInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  auto out_type =
    CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  return out_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(CholeskyInverse, BaseOperator);
void CholeskyInverse::Init(const bool upper) { set_upper(upper); }

void CholeskyInverse::set_upper(const bool upper) { (void)this->AddAttr(kUpper, api::MakeValue(upper)); }

bool CholeskyInverse::get_upper() const { return GetValue<bool>(GetAttr(kUpper)); }

AbstractBasePtr CholeskyInverseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = CholeskyInverseInferType(primitive, input_args);
  auto infer_shape = CholeskyInverseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGCholeskyInverseInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyInverseInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyInverseInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyInverseInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CholeskyInverse, prim::kPrimCholeskyInverse, AGCholeskyInverseInfer, false);
}  // namespace ops
}  // namespace mindspore
