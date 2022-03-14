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

#include <algorithm>
#include <set>

#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kBNTrainingUpdateInputNum = 7;

int64_t BNTrainingUpdateGetAndCheckFormat(const PrimitivePtr &primitive, const ValuePtr &value) {
  int64_t data_format;
  bool result = CheckAndConvertUtils::GetDataFormatEnumValue(value, &data_format);
  if (!result || (data_format != Format::NHWC && data_format != Format::NCHW && data_format != Format::NCDHW)) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', data format is invalid, only support NCHW, NHWC and NCDHW.";
  }
  return data_format;
}
abstract::TupleShapePtr BNTrainingUpdateInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kBNTrainingUpdateInputNum, prim_name);
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto sum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto square_sum_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto scale_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto offset_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto mean_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto variance_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
  auto data_format_ptr = primitive->GetAttr("format");
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  int64_t data_format = BNTrainingUpdateGetAndCheckFormat(primitive, data_format_ptr);
  size_t c_axis = kInputIndex1;
  if (data_format == Format::NHWC) {
    c_axis = kInputIndex3;
  }
  // input_x rank should be equal with 4
  (void)CheckAndConvertUtils::CheckInteger("input_x rank", input_x_shape.size(), kEqual, 4, prim_name);
  // sum rank should be equal with 1
  (void)CheckAndConvertUtils::CheckInteger("sum rank", sum_shape.size(), kEqual, 1, prim_name);
  // square_sum rank should be equal with 1
  (void)CheckAndConvertUtils::CheckInteger("square_sum rank", square_sum_shape.size(), kEqual, 1, prim_name);
  // scale rank should be equal with 1
  (void)CheckAndConvertUtils::CheckInteger("scale rank", scale_shape.size(), kEqual, 1, prim_name);
  // offset rank should be equal with 1
  (void)CheckAndConvertUtils::CheckInteger("offset rank", offset_shape.size(), kEqual, 1, prim_name);
  // mean rank should be equal with 1
  (void)CheckAndConvertUtils::CheckInteger("mean rank", mean_shape.size(), kEqual, 1, prim_name);
  // variance rank should be equal with 1
  (void)CheckAndConvertUtils::CheckInteger("variance rank", variance_shape.size(), kEqual, 1, prim_name);
  // sum shape should be equal with input_x_shape[1]
  CheckAndConvertUtils::Check("sum shape", sum_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  // square_sum shape should be equal with input_x_shape[1]
  CheckAndConvertUtils::Check("square_sum shape", square_sum_shape[0], kEqual, input_x_shape[c_axis], prim_name,
                              TypeError);
  // scale shape should be equal with input_x_shape[1]
  CheckAndConvertUtils::Check("scale shape", scale_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  // offset shape should be equal with input_x_shape[1]
  CheckAndConvertUtils::Check("offset shape", offset_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  // mean shape should be equal with input_x_shape[1]
  CheckAndConvertUtils::Check("mean shape", mean_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  // variance shape should be equal with input_x_shape[1]
  CheckAndConvertUtils::Check("variance shape", variance_shape[0], kEqual, input_x_shape[c_axis], prim_name, TypeError);
  auto input_x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  auto variance_shape_ptr = input_args[kInputIndex6]->BuildShape();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    input_x_shape_ptr, variance_shape_ptr, variance_shape_ptr, variance_shape_ptr, variance_shape_ptr});
}

TuplePtr BNTrainingUpdateInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kBNTrainingUpdateInputNum, prim_name);
  auto input_x_type = input_args[kInputIndex0]->BuildType();
  auto sum_type = input_args[kInputIndex1]->BuildType();
  auto square_sum_type = input_args[kInputIndex2]->BuildType();
  auto scale_type = input_args[kInputIndex3]->BuildType();
  auto offset_type = input_args[kInputIndex4]->BuildType();
  auto mean_type = input_args[kInputIndex5]->BuildType();
  auto variance_type = input_args[kInputIndex6]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // input_x type must be valid
  CheckAndConvertUtils::CheckTensorTypeValid("input_x type", input_x_type, valid_types, prim_name);
  // sum type must be valid
  CheckAndConvertUtils::CheckTensorTypeValid("sum type", sum_type, valid_types, prim_name);
  // square_sum type must be valid
  CheckAndConvertUtils::CheckTensorTypeValid("square_sum type", square_sum_type, valid_types, prim_name);
  // scale type must be valid
  CheckAndConvertUtils::CheckTensorTypeValid("scale_type", scale_type, valid_types, prim_name);
  // offset type must be valid
  CheckAndConvertUtils::CheckTensorTypeValid("offset_type", offset_type, valid_types, prim_name);
  // mean type must be valid
  CheckAndConvertUtils::CheckTensorTypeValid("mean_type", mean_type, valid_types, prim_name);
  // variance type must be valid
  CheckAndConvertUtils::CheckTensorTypeValid("variance_type", variance_type, valid_types, prim_name);
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{input_x_type, variance_type, variance_type, variance_type, variance_type});
}
}  // namespace

AbstractBasePtr BNTrainingUpdateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return abstract::MakeAbstract(BNTrainingUpdateInferShape(primitive, input_args),
                                BNTrainingUpdateInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(BNTrainingUpdate, prim::kPrimBNTrainingUpdate, BNTrainingUpdateInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
