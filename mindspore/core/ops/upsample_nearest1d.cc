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
 * distributed under the License is distributed on an "ASF IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "upsample_nearest1d.h"

namespace mindspore {
namespace ops {
// upsample_nearest1d
namespace {

abstract::ShapePtr UpsampleNearest1dInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != 3) {
    MS_LOG(EXCEPTION) << "input args size should be 3, but got " << input_args.size();
  }
  auto prim_name = primitive->name();
  auto input_shape_ptr = input_args[0]->BuildShape()->cast<abstract::ShapePtr>();

  auto output_size_value = input_args[1]->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(output_size_value);
  std::vector<int64_t> output_size_shape;
  std::transform(output_size_value->value().begin(), output_size_value->value().end(),
                 std::back_inserter(output_size_shape), [](const ValuePtr &value) { return GetValue<int64_t>(value); });
  MS_EXCEPTION_IF_NULL(input_shape_ptr);

  auto input_shape = input_shape_ptr->shape();
  if (input_shape.size() != 3) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input tensor must be a 3D Tensor, but got:" << input_shape.size();
  }

  ShapeVector ret_shape{input_shape[0], input_shape[1], output_size_shape[0]};
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr UpsampleNearest1dInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  TypePtr input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);

  const std::set<TypePtr> valid_types = {kFloat32, kFloat16, kUInt8};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types, prim_name);

  return input_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(UpsampleNearest1d, BaseOperator);

AbstractBasePtr UpsampleNearest1dInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = UpsampleNearest1dInferType(primitive, input_args);
  auto infer_shape = UpsampleNearest1dInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGUpsampleNearest1dInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleNearest1dInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleNearest1dInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleNearest1dInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleNearest1d, prim::kPrimUpsampleNearest1d, AGUpsampleNearest1dInfer, false);
}  // namespace ops
}  // namespace mindspore
