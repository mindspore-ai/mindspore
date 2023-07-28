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

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/image_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "ops/upsample_interpolating_3d.h"
#include "ops/upsample_nearest_3d.h"
#include "ops/upsample_trilinear_3d.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kUpsample3DMinInputNum = 2;
constexpr int64_t kUpsample3DMaxInputNum = 3;
constexpr int64_t kVALUE_1 = 1;
constexpr int64_t kVALUE_2 = 2;
constexpr int64_t kVALUE_3 = 3;
constexpr int64_t kVALUE_5 = 5;

void UpdateAttrNoneList(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                        size_t *const scales_idx, const std::string &prim_name) {
  if (input_args.size() == kVALUE_3) {
    std::vector<int64_t> none_list{};
    auto size_type = input_args[kInputIndex1]->BuildType();
    MS_EXCEPTION_IF_NULL(size_type);
    auto is_output_size_none = size_type->type_id() == kMetaTypeNone;
    auto scale_type = input_args[kInputIndex2]->BuildType();
    MS_EXCEPTION_IF_NULL(scale_type);
    auto is_scales_none = scale_type->type_id() == kMetaTypeNone;
    if (is_output_size_none && is_scales_none) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", only one of 'scales' and 'output_size' can be specified."
                               << " But get both empty or None.";
    } else if (!is_output_size_none && !is_scales_none) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", only one of 'scales' and 'output_size' can be specified."
                               << " But get both.";
    } else if (is_output_size_none) {
      none_list.push_back(static_cast<int64_t>(kInputIndex1));
    } else {
      none_list.push_back(static_cast<int64_t>(kInputIndex2));
    }
    (void)primitive->AddAttr(kAttrNoneList, MakeValue<std::vector<int64_t>>(none_list));
  } else {
    *scales_idx = kInputIndex1;
  }
}

void InferFromSize(const PrimitivePtr &primitive, const AbstractBasePtr &input_arg, const std::string &prim_name,
                   std::vector<int64_t> *const y_shape) {
  auto size_value_ptr = input_arg->BuildValue();
  MS_EXCEPTION_IF_NULL(size_value_ptr);
  auto output_size = GetShapeValue(primitive, input_arg);
  if (IsValueKnown(size_value_ptr)) {
    (void)CheckAndConvertUtils::CheckPositiveVector(kOutputSize, output_size, prim_name);
  }
  if (!IsDynamicRank(output_size)) {
    (void)CheckAndConvertUtils::CheckInteger("elements' number of output_size", SizeToLong(output_size.size()), kEqual,
                                             kVALUE_3, prim_name);
  } else {
    output_size = std::vector<int64_t>(kVALUE_3, abstract::Shape::kShapeDimAny);
  }
  (void)y_shape->insert(y_shape->end(), output_size.begin(), output_size.end());
}

void InferFromScales(const AbstractBasePtr &input_arg, const std::string &prim_name,
                     const std::vector<int64_t> &x_shape, std::vector<int64_t> *const y_shape) {
  auto scales_value_ptr = input_arg->BuildValue();
  MS_EXCEPTION_IF_NULL(scales_value_ptr);
  if (IsValueKnown(scales_value_ptr) && !IsDynamicRank(x_shape)) {
    std::vector<double> scales;
    if (scales_value_ptr->isa<tensor::Tensor>()) {
      scales = CheckAndConvertUtils::CheckTensorFloatValue("scales", scales_value_ptr, prim_name);
    } else if (scales_value_ptr->isa<ValueSequence>()) {
      scales = CheckAndConvertUtils::CheckListOrTupleFloat("scales", scales_value_ptr, prim_name);
    } else {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', scales should be 1D Tensor[Float] or Tuple[Float].";
    }
    (void)CheckAndConvertUtils::CheckPositiveVector(kScales, scales, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("elements' number of scales", SizeToLong(scales.size()), kEqual, kVALUE_3,
                                             prim_name);
    for (int64_t idx = 0; idx < kVALUE_3; ++idx) {
      (void)y_shape->emplace_back(
        x_shape[LongToSize(idx + kVALUE_2)] != abstract::Shape::kShapeDimAny
          ? static_cast<int64_t>(floor(x_shape[LongToSize(idx + kVALUE_2)] * scales[LongToSize(idx)]))
          : abstract::Shape::kShapeDimAny);
    }
  } else {
    for (int64_t idx = 0; idx < kVALUE_3; ++idx) {
      (void)y_shape->emplace_back(abstract::Shape::kShapeDimAny);
    }
  }
}

void GetOutputShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                    const std::vector<int64_t> &x_shape, std::vector<int64_t> *const y_shape) {
  auto prim_name = primitive->name();
  // none_list and idx
  size_t scales_idx(kInputIndex2);
  UpdateAttrNoneList(primitive, input_args, &scales_idx, prim_name);
  auto none_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrNoneList));
  (void)CheckAndConvertUtils::CheckInteger("the length of non_list", SizeToLong(none_list.size()), kEqual, kVALUE_1,
                                           prim_name);
  // infer output shape
  if (none_list[kInputIndex0] != kVALUE_1) {
    InferFromSize(primitive, input_args[kInputIndex1], prim_name, y_shape);
  } else if (none_list[kInputIndex0] != kVALUE_2) {
    InferFromScales(input_args[scales_idx], prim_name, x_shape, y_shape);
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', something unexpected happened.";
  }
}

abstract::ShapePtr UpsampleInterpolating3DInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kUpsample3DMinInputNum, prim_name);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("dimension of x", SizeToLong(x_shape.size()), kEqual, kVALUE_5, prim_name);
  }

  ShapeVector y_shape;
  if (IsDynamicRank(x_shape)) {
    (void)y_shape.emplace_back(abstract::Shape::kShapeDimAny);
    (void)y_shape.emplace_back(abstract::Shape::kShapeDimAny);
  } else {
    (void)y_shape.emplace_back(x_shape[kInputIndex0]);
    (void)y_shape.emplace_back(x_shape[kInputIndex1]);
  }
  GetOutputShape(primitive, input_args, x_shape, &y_shape);

  if (!IsDynamic(y_shape)) {
    for (size_t i = 0; i < y_shape.size(); i++) {
      (void)CheckAndConvertUtils::CheckInteger("output shape", y_shape[i], kGreaterThan, 0, prim_name);
    }
  }

  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr UpsampleInterpolatingInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  std::set<TypePtr> valid_types(common_float_types);
  if (prim_name == "UpsampleNearest3D") {
    (void)valid_types.insert(kUInt8);
  }
  auto x_arg = input_args.at(kInputIndex0);
  MS_EXCEPTION_IF_NULL(x_arg);
  auto x_type = x_arg->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
}
}  // namespace

abstract::AbstractBasePtr UpsampleInterpolating3DInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kUpsample3DMaxInputNum, prim_name);
  auto type = UpsampleInterpolatingInferType(primitive, input_args);
  auto shape = UpsampleInterpolating3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

bool UpsampleTrilinear3D::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(UpsampleNearest3D, BaseOperator);
MIND_API_OPERATOR_IMPL(UpsampleTrilinear3D, BaseOperator);

// AG means auto generated
class MIND_API AGUpsampleInterpolating3DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolating3DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolatingInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolating3DInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleTrilinear3D, prim::kPrimUpsampleTrilinear3D, AGUpsampleInterpolating3DInfer,
                                 false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleNearest3D, prim::kPrimUpsampleNearest3D, AGUpsampleInterpolating3DInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
