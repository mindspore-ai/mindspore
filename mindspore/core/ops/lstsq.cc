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

#include <memory>
#include <set>
#include <string>
#include <map>

#include "ops/lstsq.h"
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
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LstsqInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t x_dim_num = 2;
  const int64_t a_dim_num_1 = 1;
  const int64_t a_dim_num_2 = 2;

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto a_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto a_shape = a_shape_map[kShape];

  if (IsDynamicRank(x_shape) || IsDynamicRank(a_shape)) {
    return std::make_shared<abstract::Shape>(
      std::vector<int64_t>{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny});
  }

  if (x_shape.size() != x_dim_num) {
    MS_EXCEPTION(ValueError) << "For 'Lstsq', the dimension of x must be equal to 2, but got x_dim: " << x_shape.size()
                             << ".";
  }
  if (a_shape.size() != a_dim_num_2 && a_shape.size() != a_dim_num_1) {
    MS_EXCEPTION(ValueError) << "For 'Lstsq', the dimension of 'a' must be equal to 2 or 1, but got a_dim: "
                             << a_shape.size() << ".";
  }
  if (!IsDynamicShape(x_shape) && !IsDynamicShape(a_shape) && x_shape[0] != a_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For 'Lstsq', the length of x_dim[0] must be equal to the length of a_dims[0]. But got x_dim[0]: "
      << x_shape[0] << ",  a_dims[0]: " << a_shape[0] << ".";
  }
  ShapeVector y_shape;
  if (a_shape.size() == a_dim_num_1) {
    y_shape.push_back(x_shape[1]);
    y_shape.push_back(1);
  } else {
    y_shape.push_back(x_shape[1]);
    y_shape.push_back(a_shape[1]);
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr LstsqInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("a", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(Lstsq, BaseOperator);
AbstractBasePtr LstsqInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto infer_type = LstsqInferType(primitive, input_args);
  auto infer_shape = LstsqInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLstsqInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LstsqInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LstsqInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LstsqInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Lstsq, prim::kPrimLstsq, AGLstsqInfer, false);
}  // namespace ops
}  // namespace mindspore
