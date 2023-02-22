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

#include "ops/bounding_box_encode.h"

#include <set>
#include <map>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BoundingBoxEncodeInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto anchor_box = input_args[0]->BuildShape();
  auto groundtruth_box = input_args[1]->BuildShape();

  MS_EXCEPTION_IF_NULL(anchor_box);
  MS_EXCEPTION_IF_NULL(groundtruth_box);

  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("arg size", SizeToLong(input_args.size()), kEqual, input_num, prim_name);

  auto anchor_box_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto groundtruth_box_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];

  const int64_t kShapeSize = 2;
  (void)CheckAndConvertUtils::CheckInteger("anchor box rank", SizeToLong(anchor_box_shape.size()), kEqual, kShapeSize,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("groundtruth box rank", SizeToLong(groundtruth_box_shape.size()), kEqual,
                                           kShapeSize, prim_name);

  if (anchor_box_shape[0] != groundtruth_box_shape[0]) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', 'anchor_box' and 'groundtruth_box' must have the same first dimension. But got anchor_box_shape[0]: "
      << anchor_box_shape[0] << ", groundtruth_box_shape[0]: " << groundtruth_box_shape[0] << ".";
  }

  const int64_t last_dimension = 4;
  if (anchor_box_shape[1] > 0 && anchor_box_shape[1] != last_dimension) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', 'anchor_box' last dimension must be 4, but got: " << anchor_box_shape[1] << ".";
  }
  if (groundtruth_box_shape[1] > 0 && groundtruth_box_shape[1] != last_dimension) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', 'groundtruth_box' last dimension must be 4, but got: " << groundtruth_box_shape[1]
                             << ".";
  }

  auto x_shape = anchor_box->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(x_shape);
  return x_shape;
}

TypePtr BoundingBoxEncodeInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  std::set<TypePtr> valid_x_type;
  (void)valid_x_type.emplace(kFloat16);
  (void)valid_x_type.emplace(kFloat32);

  for (size_t i = 0; i < input_args.size(); i++) {
    auto x_type = input_args[i]->BuildType();
    MS_EXCEPTION_IF_NULL(x_type);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_type, valid_x_type, prim_name);
  }

  std::map<std::string, TypePtr> types;
  (void)types.emplace("anchor_box", input_args[0]->BuildType());
  (void)types.emplace("groundtruth_box", input_args[1]->BuildType());

  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(BoundingBoxEncode, BaseOperator);
AbstractBasePtr BoundingBoxEncodeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);

  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());

  auto infer_type = BoundingBoxEncodeInferType(primitive, input_args);
  auto infer_shape = BoundingBoxEncodeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGBoundingBoxEncodeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BoundingBoxEncodeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BoundingBoxEncodeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BoundingBoxEncodeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BoundingBoxEncode, prim::kPrimBoundingBoxEncode, AGBoundingBoxEncodeInfer, false);
}  // namespace ops
}  // namespace mindspore
