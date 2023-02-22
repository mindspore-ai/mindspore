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

#include "ops/square_sum_all.h"

#include <map>
#include <string>
#include <memory>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
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
abstract::TupleShapePtr SquareSumAllInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto input_y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("x", input_x_shape, kEqual, input_y_shape, prim_name, ValueError);
  if (IsDynamicRank(input_x_shape)) {
    auto output_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{output_shape, output_shape});
  }
  std::vector<int64_t> shape_vec;
  if (primitive->HasAttr(kBatchRank)) {
    int64_t batch_rank = GetValue<int64_t>(primitive->GetAttr(kBatchRank));
    for (int64_t index = 0; index < batch_rank; index++) {
      shape_vec.push_back(input_x_shape[LongToSize(index)]);
    }
  }
  auto output_shape = std::make_shared<abstract::Shape>(shape_vec);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{output_shape, output_shape});
}

TuplePtr SquareSumAllInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  // x must have the same type as y and is either float16 or float32.
  auto input_x_type = input_args[kInputIndex0]->BuildType();
  auto input_y_type = input_args[kInputIndex1]->BuildType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_x_type);
  (void)types.emplace("y", input_y_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  auto output_type = input_x_type->cast<TensorTypePtr>();
  return std::make_shared<Tuple>(std::vector<TypePtr>{output_type, output_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SquareSumAll, BaseOperator);
AbstractBasePtr SquareSumAllInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = SquareSumAllInferType(primitive, input_args);
  auto shapes = SquareSumAllInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGSquareSumAllInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SquareSumAllInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SquareSumAllInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SquareSumAllInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SquareSumAll, prim::kPrimSquareSumAll, AGSquareSumAllInfer, false);
}  // namespace ops
}  // namespace mindspore
