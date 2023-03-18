/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/polygamma.h"

#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PolygammaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[kInputIndex1]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  int64_t input_a;
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto a_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (a_shape.size() != 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', 'a' should be a 0-dim Tensor, but got rank: " << a_shape.size() << ".";
  }
  if (input_args[kInputIndex0]->isa<abstract::AbstractTensor>()) {
    auto input_a_ptr = input_args[kInputIndex0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(input_a_ptr);
    auto input_a_value_ptr = input_a_ptr->BuildValue();
    MS_EXCEPTION_IF_NULL(input_a_value_ptr);
    if (input_a_value_ptr->isa<tensor::Tensor>()) {
      auto input_a_tensor = input_a_value_ptr->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(input_a_tensor);
      input_a = *static_cast<int64_t *>(input_a_tensor->data_c());
    } else {
      return std::make_shared<abstract::Shape>(x_shape);
    }
  } else if (input_args[kInputIndex0]->isa<abstract::AbstractScalar>()) {
    auto input_a_ptr = input_args[kInputIndex0]->cast<abstract::AbstractScalarPtr>();
    MS_EXCEPTION_IF_NULL(input_a_ptr);
    auto input_value = input_a_ptr->BuildValue();
    if (input_value->isa<ValueAny>()) {
      return std::make_shared<abstract::Shape>(x_shape);
    }
    input_a = GetValue<int64_t>(input_value);
  } else {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', the input a type should be tensor or scalar, but got invalid abstract type:"
                      << input_args[kInputIndex0]->type_name() << ".";
  }
  (void)CheckAndConvertUtils::CheckInteger("input_a", input_a, kGreaterEqual, 1, prim_name);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr PolygammaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto a_type = input_args[kInputIndex0]->BuildType();
  auto x_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> a_valid_types = {kInt32, kInt64};
  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("a", a_type, a_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, x_valid_types, prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Polygamma, BaseOperator);
AbstractBasePtr PolygammaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto infer_shape = PolygammaInferShape(primitive, input_args);
  auto infer_type = PolygammaInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGPolygammaInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PolygammaInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PolygammaInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PolygammaInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Polygamma, prim::kPrimPolygamma, AGPolygammaInfer, false);
}  // namespace ops
}  // namespace mindspore
