/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/avg_pool_grad.h"

#include <set>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/conv_pool_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kXFromTensor = "x_from_tensor";
abstract::ShapePtr AvgPoolGradInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  // input_args: x, y, dout, return x.shape
  auto prim_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x_shape = x->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto value = x->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  if (primitive->HasAttr(kXFromTensor) && GetValue<bool>(primitive->GetAttr(kXFromTensor))) {
    if (x->isa<abstract::AbstractTensor>() && value->isa<tensor::Tensor>()) {
      // The first input is Tensor(convert from a tuple), the value of Tuple is the real "x_origin" shape.
      auto check_shape = x_shape->cast<abstract::ShapePtr>()->shape();
      auto x_origin = CheckAndConvertUtils::CheckTensorIntValue("x_origin", value, prim_name);
      if (check_shape != x_origin) {
        return std::make_shared<abstract::Shape>(x_origin);
      }
    }
  }
  auto shape_element = x_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr AvgPoolGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  // input_args: x, y, dout, return x.shape
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(AvgPoolGrad, PoolGrad);
AbstractBasePtr AvgPoolGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto types = AvgPoolGradInferType(primitive, input_args);
  auto shapes = AvgPoolGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGAvgPoolGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolGradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AvgPoolGrad, prim::kPrimAvgPoolGrad, AGAvgPoolGradInfer, false);
}  // namespace ops
}  // namespace mindspore
