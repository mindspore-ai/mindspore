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

#include "ops/bn_training_update.h"

#include <set>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/format.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kBNTrainingUpdateInputNum = 8;

void BNTrainingUpdateCheckFormat(const PrimitivePtr &primitive, const mindspore::Format format) {
  static std::vector<mindspore::Format> valid_formats{Format::NHWC, Format::NCHW, Format::NCDHW};
  auto CheckFormat = [format](const mindspore::Format other) { return format == other; };
  bool is_valid = std::any_of(valid_formats.begin(), valid_formats.end(), CheckFormat);
  if (MS_UNLIKELY(!is_valid)) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', data format must be NCHW, NHWC and NCDHW, but got "
                      << FormatEnumToString(format) << ".";
  }
}

void BNTrainingUpdateCheckShapes(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::vector<ShapeVector> shapes{};
  for (size_t i = kInputIndex0; i < kInputIndex7; ++i) {
    auto shape = input_args[i]->GetShape()->GetShapeVector();
    shapes.emplace_back(std::move(shape));
  }
  const auto &input_x_shape = shapes[kInputIndex0];
  const auto &sum_shape = shapes[kInputIndex1];
  const auto &square_sum_shape = shapes[kInputIndex2];
  const auto &scale_shape = shapes[kInputIndex3];
  const auto &offset_shape = shapes[kInputIndex4];
  const auto &mean_shape = shapes[kInputIndex5];
  const auto &variance_shape = shapes[kInputIndex6];

  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input_x rank", SizeToLong(input_x_shape.size()), kGreaterThan, 1,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("sum rank", SizeToLong(sum_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("square_sum rank", SizeToLong(square_sum_shape.size()), kEqual, 1,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("scale rank", SizeToLong(scale_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("offset rank", SizeToLong(offset_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("mean rank", SizeToLong(mean_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("variance rank", SizeToLong(variance_shape.size()), kEqual, 1, prim_name);

  if (std::any_of(shapes.begin(), shapes.end(), IsDynamic)) {
    return;
  }
  // get format
  auto format_opt = GetScalarValue<int64_t>(input_args[kInputIndex7]->GetValue());
  if (MS_UNLIKELY(!format_opt.has_value())) {
    MS_LOG(EXCEPTION) << "For " << prim_name << ", failed to get format's value.";
  }
  auto format = static_cast<mindspore::Format>(format_opt.value());
  BNTrainingUpdateCheckFormat(primitive, format);

  auto c_axis = format == Format::NHWC ? input_x_shape.size() - kInputIndex1 : kInputIndex1;
  CheckAndConvertUtils::Check("sum shape", sum_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  CheckAndConvertUtils::Check("square_sum shape", square_sum_shape[0], kEqual, input_x_shape[c_axis], prim_name,
                              TypeError);

  CheckAndConvertUtils::Check("scale shape", scale_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  CheckAndConvertUtils::Check("offset shape", offset_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  CheckAndConvertUtils::Check("mean shape", mean_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  CheckAndConvertUtils::Check("variance shape", variance_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
}

abstract::TupleShapePtr BNTrainingUpdateInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kBNTrainingUpdateInputNum, prim_name);
  BNTrainingUpdateCheckShapes(primitive, input_args);

  // get out_shapes
  auto input_x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto variance_shape_ptr = input_args[kInputIndex6]->GetShape();
  std::vector<abstract::BaseShapePtr> out_shapes{input_x_shape_ptr->Clone(), variance_shape_ptr->Clone(),
                                                 variance_shape_ptr->Clone(), variance_shape_ptr->Clone(),
                                                 variance_shape_ptr->Clone()};

  return std::make_shared<abstract::TupleShape>(std::move(out_shapes));
}

TuplePtr BNTrainingUpdateInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kBNTrainingUpdateInputNum, prim_name);
  auto input_x_type = input_args[kInputIndex0]->GetType();
  auto sum_type = input_args[kInputIndex1]->GetType();
  auto square_sum_type = input_args[kInputIndex2]->GetType();
  auto scale_type = input_args[kInputIndex3]->GetType();
  auto offset_type = input_args[kInputIndex4]->GetType();
  auto mean_type = input_args[kInputIndex5]->GetType();
  auto variance_type = input_args[kInputIndex6]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // input_x type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x type", input_x_type, valid_types, prim_name);
  // sum type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("sum type", sum_type, valid_types, prim_name);
  // square_sum type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("square_sum type", square_sum_type, valid_types, prim_name);
  // scale type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("scale_type", scale_type, valid_types, prim_name);
  // offset type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("offset_type", offset_type, valid_types, prim_name);
  // mean type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("mean_type", mean_type, valid_types, prim_name);
  // variance type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("variance_type", variance_type, valid_types, prim_name);
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{input_x_type, variance_type, variance_type, variance_type, variance_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(BNTrainingUpdate, BaseOperator);
AbstractBasePtr BNTrainingUpdateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(BNTrainingUpdateInferShape(primitive, input_args),
                                BNTrainingUpdateInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGBNTrainingUpdateInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BNTrainingUpdateInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BNTrainingUpdateInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BNTrainingUpdateInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BNTrainingUpdate, prim::kPrimBNTrainingUpdate, AGBNTrainingUpdateInfer, false);
}  // namespace ops
}  // namespace mindspore
