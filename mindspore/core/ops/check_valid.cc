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

#include "ops/check_valid.h"
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kCheckValidInputsNum = 2;
constexpr int64_t kBboxesLastDim = 4;
constexpr int64_t kImgMetasFirstDim = 3;
constexpr int64_t kNumber1 = 1;
constexpr int64_t kNumber2 = 2;

TypePtr CheckValidInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  (void)CheckAndConvertUtils::CheckInteger("input args size", SizeToLong(input_args.size()), kEqual,
                                           kCheckValidInputsNum, prim_name);
  const std::set<TypePtr> valid_types = {kInt16, kUInt8, kFloat16, kFloat32};
  auto bboxes_dtype = input_args[0]->BuildType();
  auto metas_dtype = input_args[1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("bboxes", bboxes_dtype, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("metas", metas_dtype, valid_types, prim_name);
  return std::make_shared<TensorType>(kBool);
}

abstract::ShapePtr CheckValidInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto bboxes_shape_ptr = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(bboxes_shape_ptr);
  auto metas_shape_ptr = input_args[kInputIndex1]->BuildShape();
  MS_EXCEPTION_IF_NULL(metas_shape_ptr);

  if (bboxes_shape_ptr->IsDynamic() || metas_shape_ptr->IsDynamic()) {
    return bboxes_shape_ptr->cast<abstract::ShapePtr>();
  }

  auto bboxes_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  CheckAndConvertUtils::CheckInteger("bboxes rank", SizeToLong(bboxes_shape.size()), kEqual, kNumber2, prim_name);
  int64_t bboxes_last_dim = bboxes_shape[bboxes_shape.size() - 1];
  if (bboxes_last_dim != kBboxesLastDim) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the last dimension of 'bboxes' must equal to 4, but got: " << bboxes_last_dim
                             << ".";
  }

  auto img_metas_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  CheckAndConvertUtils::CheckInteger("img_metas rank", SizeToLong(img_metas_shape.size()), kEqual, kNumber1, prim_name);
  int64_t img_metas_first_dim = img_metas_shape[0];
  if (img_metas_first_dim != kImgMetasFirstDim) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the first dimension of 'img_metas' must equal to 3, but got: "
                             << img_metas_first_dim << ".";
  }
  bboxes_shape.pop_back();
  return std::make_shared<abstract::Shape>(bboxes_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(CheckValid, BaseOperator);
AbstractBasePtr CheckValidInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = CheckValidInferType(primitive, input_args);
  auto infer_shape = CheckValidInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CheckValid, prim::kPrimCheckValid, CheckValidInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
