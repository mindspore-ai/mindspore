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
#include "ops/geqrf.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
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
abstract::TupleShapePtr GeqrfInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const size_t kTwo = 2;
  const std::vector<int64_t> UNKNOWN_RANK = {-2};
  auto a_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (IsDynamicRank(a_shape) || IsDynamic(a_shape)) {
    ShapeVector dyn_shape{UNKNOWN_RANK};
    std::vector<abstract::BaseShapePtr> shape_tuple;
    (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(dyn_shape));
    (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(dyn_shape));
    return std::make_shared<abstract::TupleShape>(shape_tuple);
  }
  auto ndim = a_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("ndim", ndim, kGreaterEqual, kTwo, primitive->name());
  auto m = a_shape[ndim - 2];
  auto n = a_shape[ndim - 1];
  auto p = std::min(m, n);
  std::vector<int64_t> tau_shape;
  for (size_t i = 0; i < ndim - kDim2; i++) {
    tau_shape.emplace_back(a_shape[i]);
  }
  tau_shape.emplace_back(p);

  std::vector<abstract::BaseShapePtr> shape_tuple;
  (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(a_shape));
  (void)shape_tuple.emplace_back(std::make_shared<abstract::Shape>(tau_shape));
  return std::make_shared<abstract::TupleShape>(shape_tuple);
}

TypePtr GeqrfInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, prim->name());

  std::vector<TypePtr> type_tuple = {type, type};
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

AbstractBasePtr GeqrfInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());

  auto infer_type = GeqrfInferType(primitive, input_args);
  auto infer_shape = GeqrfInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Geqrf, BaseOperator);

// AG means auto generated
class MIND_API AGGeqrfInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GeqrfInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GeqrfInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GeqrfInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Geqrf, prim::kPrimGeqrf, AGGeqrfInfer, false);
}  // namespace ops
}  // namespace mindspore
