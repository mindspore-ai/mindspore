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

#include "ops/pdist.h"

#include <algorithm>
#include <memory>
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
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PdistInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto x_size = x_shape.size();
  const int64_t input_dim = 2;
  (void)CheckAndConvertUtils::CheckInteger("x dim", SizeToLong(x_size), kEqual, input_dim, prim_name);

  auto input_x = input_args[0];
  MS_EXCEPTION_IF_NULL(input_x);
  if (x_shape[x_size - input_dim] >= 0) {
    int64_t dim_R = x_shape[x_size - input_dim];
    const float out_shape_used = 0.5;
    dim_R = FloatToLong(dim_R * (dim_R - 1) * out_shape_used);
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < x_size - input_dim; i++) {
      out_shape.push_back(x_shape[i]);
    }
    out_shape.push_back(dim_R);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(-1);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr PdistInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat64, kFloat32, kFloat16};
  auto x_dtype = input_args[0]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, primitive->name());
}
}  // namespace

float Pdist::get_p() const {
  auto value_ptr = this->GetAttr(kP);
  return GetValue<float>(value_ptr);
}
void Pdist::set_p(const float p) { (void)this->AddAttr(kP, api::MakeValue(p)); }

MIND_API_OPERATOR_IMPL(Pdist, BaseOperator);
AbstractBasePtr PdistInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = PdistInferType(primitive, input_args);
  auto infer_shape = PdistInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGPdistInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PdistInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PdistInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PdistInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Pdist, prim::kPrimPdist, AGPdistInfer, false);
}  // namespace ops
}  // namespace mindspore
