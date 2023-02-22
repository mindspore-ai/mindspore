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
#include "ops/resize_area.h"

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
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
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
abstract::ShapePtr ResizeAreaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int64_t> output_shape(4, -1);
  auto images_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (!IsDynamicRank(images_shape)) {
    constexpr int64_t image_shape_size = 4;
    (void)CheckAndConvertUtils::CheckInteger("images dimension", SizeToLong(images_shape.size()), kEqual,
                                             image_shape_size, primitive->name());
    output_shape[0] = images_shape[0];
    output_shape[kInputIndex3] = images_shape[kInputIndex3];
  }
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("size dimension", SizeToLong(size_shape.size()), kEqual, 1,
                                           primitive->name());
  if (!IsDynamic(size_shape)) {
    constexpr int64_t size_num = 2;
    (void)CheckAndConvertUtils::CheckInteger("input1 num", size_shape[0], kEqual, size_num, primitive->name());
  }

  if (!input_args[1]->BuildValue()->isa<AnyValue>() && !input_args[1]->BuildValue()->isa<None>()) {
    auto input1_value = input_args[1]->BuildValue();
    if (!input1_value->isa<tensor::Tensor>()) {
      MS_LOG(EXCEPTION) << "For ResizeArea, the inputs[1] must be a tensor, but got: " << input1_value->ToString()
                        << ".";
    }
    auto input1_shape_ptr = static_cast<int32_t *>(input1_value->cast<tensor::TensorPtr>()->data_c());
    if (input1_shape_ptr[0] <= 0 || input1_shape_ptr[1] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size must be positive "
                               << ", but got " << input1_shape_ptr[0] << " , " << input1_shape_ptr[1];
    }
    output_shape[kInputIndex1] = input1_shape_ptr[kInputIndex0];
    output_shape[kInputIndex2] = input1_shape_ptr[kInputIndex1];
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ResizeAreaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return kFloat32;
}
}  // namespace
MIND_API_OPERATOR_IMPL(ResizeArea, BaseOperator);
void ResizeArea::Init(const bool align_corners) { this->set_align_corners(align_corners); }
void ResizeArea::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}
bool ResizeArea::get_align_corners() const { return GetValue<bool>(GetAttr(kAlignCorners)); }
AbstractBasePtr ResizeAreaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  const std::set<TypePtr> valid_types = {kInt8, kUInt8, kInt16, kUInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid_types_1 = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", input_args[0]->BuildType(), valid_types,
                                                   primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", input_args[1]->BuildType(), valid_types_1,
                                                   primitive->name());
  auto infer_shape = ResizeAreaInferShape(primitive, input_args);
  auto infer_type = ResizeAreaInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGResizeAreaInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeAreaInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeAreaInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeAreaInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeArea, prim::kPrimResizeArea, AGResizeAreaInfer, false);
}  // namespace ops
}  // namespace mindspore
