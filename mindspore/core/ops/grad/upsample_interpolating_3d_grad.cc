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

#include <cmath>
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
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/image_ops.h"
#include "ops/grad/upsample_nearest_3d_grad.h"
#include "ops/grad/upsample_trilinear_3d_grad.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kUpsample3DGradMinInputNum = 3;
const int64_t kVALUE_1 = 1;
const int64_t kVALUE_2 = 2;
const int64_t kVALUE_3 = 3;
const int64_t kVALUE_4 = 4;
const int64_t kVALUE_5 = 5;

void UpdateAttrNoneList(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                        size_t *const scales_idx, const std::string &prim_name) {
  if (input_args.size() == kVALUE_4) {
    std::vector<int64_t> none_list{};
    auto size_type = input_args[kInputIndex2]->BuildType();
    MS_EXCEPTION_IF_NULL(size_type);
    auto is_output_size_none = size_type->type_id() == kMetaTypeNone;
    auto scale_type = input_args[kInputIndex3]->BuildType();
    MS_EXCEPTION_IF_NULL(scale_type);
    auto is_scales_none = scale_type->type_id() == kMetaTypeNone;
    if (is_output_size_none && is_scales_none) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', either output_size or scales should be defined.";
    } else if (!is_output_size_none && !is_scales_none) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', only one of output_size or scales should be defined.";
    } else if (is_output_size_none) {
      none_list.push_back(kVALUE_2);
    } else {
      none_list.push_back(kVALUE_3);
    }
    (void)primitive->AddAttr(kAttrNoneList, MakeValue<std::vector<int64_t>>(none_list));
  } else {
    *scales_idx = kInputIndex2;
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
    (void)CheckAndConvertUtils::CheckInteger("elements number of output_size", SizeToLong(output_size.size()), kEqual,
                                             kVALUE_3, prim_name);
  } else {
    output_size = std::vector<int64_t>(kVALUE_3, abstract::Shape::kShapeDimAny);
  }
  (void)y_shape->insert(y_shape->end(), output_size.begin(), output_size.end());
}

void InferFromScales(const AbstractBasePtr &input_arg, const std::string &prim_name,
                     const std::vector<int64_t> &input_size, std::vector<int64_t> *const y_shape) {
  auto scales_value_ptr = input_arg->BuildValue();
  MS_EXCEPTION_IF_NULL(scales_value_ptr);
  if (IsValueKnown(scales_value_ptr)) {
    std::vector<double> scales;
    if (scales_value_ptr->isa<tensor::Tensor>()) {
      scales = CheckAndConvertUtils::CheckTensorFloatValue("scales", scales_value_ptr, prim_name);
    } else if (scales_value_ptr->isa<ValueSequence>()) {
      scales = CheckAndConvertUtils::CheckListOrTupleFloat("scales", scales_value_ptr, prim_name);
    } else {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', scales should be 1D Tensor[Float] or Tuple[Float].";
    }
    (void)CheckAndConvertUtils::CheckPositiveVector(kScales, scales, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("elements number of scales", SizeToLong(scales.size()), kEqual, kVALUE_3,
                                             prim_name);
    for (int64_t idx = 0; idx < kVALUE_3; ++idx) {
      (void)y_shape->emplace_back(
        input_size[LongToSize(idx + kVALUE_2)] != abstract::Shape::kShapeDimAny
          ? static_cast<int64_t>(std::floor(input_size[LongToSize(idx + kVALUE_2)] * scales[LongToSize(idx)]))
          : abstract::Shape::kShapeDimAny);
    }
  } else {
    for (int64_t idx = 0; idx < kVALUE_3; ++idx) {
      (void)y_shape->emplace_back(abstract::Shape::kShapeDimAny);
    }
  }
}

void UpsampleInterpolating3DGradCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                      const ShapeVector &grad_shape, const ShapeVector &input_size) {
  auto prim_name = primitive->name();
  size_t scales_idx(kInputIndex3);
  UpdateAttrNoneList(primitive, input_args, &scales_idx, prim_name);

  if (IsDynamicRank(input_size)) {
    return;
  }

  auto none_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrNoneList));
  (void)CheckAndConvertUtils::CheckInteger("the length of non_list", SizeToLong(none_list.size()), kEqual, kVALUE_1,
                                           prim_name);

  std::vector<int64_t> y_shape{};
  y_shape.push_back(input_size[kInputIndex0]);
  y_shape.push_back(input_size[kInputIndex1]);

  if (none_list[kInputIndex0] != kVALUE_2) {
    InferFromSize(primitive, input_args[kInputIndex2], prim_name, &y_shape);
  } else if (none_list[kInputIndex0] != kVALUE_3) {
    InferFromScales(input_args[scales_idx], prim_name, input_size, &y_shape);
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', something unexpected happened.";
  }

  bool shape_error = false;
  const size_t shape_dims = 5;
  for (size_t i = 0; i < shape_dims; i++) {
    if (y_shape[i] != abstract::Shape::kShapeDimAny && grad_shape[i] != abstract::Shape::kShapeDimAny &&
        y_shape[i] != grad_shape[i]) {
      shape_error = true;
      break;
    }
  }
  if (shape_error) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', The shape of grad, which is the same as that of output, is "
                             << input_args[kInputIndex0]->BuildShape()->ToString() << ", but the shape of output is ("
                             << std::to_string(y_shape[kInputIndex0]) << ", " << std::to_string(y_shape[kInputIndex1])
                             << ", " << std::to_string(y_shape[kInputIndex2]) << ", "
                             << std::to_string(y_shape[kInputIndex3]) << ", " << std::to_string(y_shape[kInputIndex4])
                             << ").";
  }
}

abstract::ShapePtr UpsampleInterpolating3DGradInferShape(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kUpsample3DGradMinInputNum, prim_name);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto input_size_ptr = input_args[kInputIndex1]->BuildValue();
  MS_EXCEPTION_IF_NULL(input_size_ptr);
  auto input_size = GetShapeValue(primitive, input_args[kInputIndex1]);
  if (IsValueKnown(input_size_ptr)) {
    (void)CheckAndConvertUtils::CheckPositiveVector("input_size", input_size, prim_name);
  }
  if (!IsDynamicRank(input_size)) {
    (void)CheckAndConvertUtils::CheckInteger("the elements number of input_size", SizeToLong(input_size.size()), kEqual,
                                             kVALUE_5, prim_name);
  }
  if (!IsDynamicRank(grad_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("the rank of grad_output", SizeToLong(grad_shape.size()), kEqual, kVALUE_5,
                                             prim_name);
  }
  UpsampleInterpolating3DGradCheck(primitive, input_args, grad_shape, input_size);
  // Return the dinput shape
  return std::make_shared<abstract::Shape>(input_size);
}

TypePtr UpsampleInterpolating3DGradInferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  std::set<TypePtr> valid_types(common_float_types);
  if (prim_name == "UpsampleNearest3DGrad") {
    (void)valid_types.insert(kUInt8);
  }
  auto grad_arg = input_args.at(kInputIndex0);
  MS_EXCEPTION_IF_NULL(grad_arg);
  TypePtr grad_type = grad_arg->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_type, valid_types, prim_name);
}
}  // namespace

AbstractBasePtr UpsampleInterpolating3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_types = UpsampleInterpolating3DGradInferType(primitive, input_args);
  auto infer_shapes = UpsampleInterpolating3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shapes, infer_types);
}

bool UpsampleTrilinear3DGrad::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(UpsampleTrilinear3DGrad, BaseOperator);
MIND_API_OPERATOR_IMPL(UpsampleNearest3DGrad, BaseOperator);
// AG means auto generated
class MIND_API AGUpsampleInterpolating3DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolating3DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolating3DGradInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleInterpolating3DGradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1, 2, 3}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleTrilinear3DGrad, prim::kPrimUpsampleTrilinear3DGrad,
                                 AGUpsampleInterpolating3DGradInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleNearest3DGrad, prim::kPrimUpsampleNearest3DGrad,
                                 AGUpsampleInterpolating3DGradInfer, false);
}  // namespace ops
}  // namespace mindspore
