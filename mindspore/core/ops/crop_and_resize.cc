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

#include "ops/crop_and_resize.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(CropAndResize, BaseOperator);
class CropAndResizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    MS_EXCEPTION_IF_CHECK_FAIL(
      input_args.size() == kCropAndResizeInputSize,
      "For primitive[" + prim_name + "], [input number] must be 4 but got " + std::to_string(input_args.size()));
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }

    MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex0]->BuildShape()->isa<abstract::Shape>(),
                               "For primitive[" + prim_name + "], the [x] has no abstract:Shape.");
    auto x_shape = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>()->shape();
    auto box_shape = input_args[kInputIndex1]->BuildShape()->cast<abstract::ShapePtr>()->shape();
    auto box_index_shape = input_args[kInputIndex2]->BuildShape()->cast<abstract::ShapePtr>()->shape();
    if (IsDynamicRank(x_shape) || IsDynamicRank(box_shape) || IsDynamicRank(box_index_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{UNKNOWN_RANK});
    }

    MS_EXCEPTION_IF_CHECK_FAIL(x_shape.size() == kShapeRank4, "For primitive[" + prim_name +
                                                                "], the [x shape-length] should be 4, bug got " +
                                                                std::to_string(x_shape.size()) + ".");

    if (IsDynamic(x_shape) || IsDynamic(box_shape) || IsDynamic(box_index_shape)) {
      return std::make_shared<abstract::Shape>(
        std::vector<int64_t>{UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM});
    }

    int64_t out_channel = -1;
    if (x_shape.size() == kShapeRank4 && x_shape.back() > 0) {
      out_channel = x_shape.back();
    }

    auto num_boxes = ParseNumBoxes(input_args, prim_name);
    auto crop_size_type = input_args[kInputIndex3]->BuildType();
    MS_EXCEPTION_IF_CHECK_FAIL(crop_size_type != nullptr,
                               "For primitive[" + prim_name + "], the [crop_size TypeId] is a nullptr.");
    auto value_ptr = input_args[kInputIndex3]->BuildValue();
    std::vector<int64_t> crop_size;
    if (crop_size_type->isa<TensorType>()) {
      crop_size = CheckAndConvertUtils::CheckTensorIntValue("crop_size", value_ptr, prim_name);
    } else if (IsIdentidityOrSubclass(crop_size_type, kTuple)) {
      crop_size = CheckAndConvertUtils::CheckIntOrTupleInt("crop_size", value_ptr, prim_name);
    } else {
      MS_LOG(EXCEPTION) << "For primitive[" + prim_name +
                             "], the [crop_size type] is invalid, which must be a Tensor or Tuple, but now is "
                        << crop_size_type->ToString();
    }
    CheckAndConvertUtils::Check("crop_size length", crop_size.size(), kEqual, kLimitValue2, prim_name);
    CheckAndConvertUtils::Check("crop_size weight", crop_size.front(), kGreaterThan, 0, prim_name);
    CheckAndConvertUtils::Check("box_index height", crop_size.back(), kGreaterThan, 0, prim_name);
    ShapeVector out_shape = {num_boxes, crop_size.front(), crop_size.back(), out_channel};
    return std::make_shared<abstract::Shape>(out_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    MS_EXCEPTION_IF_CHECK_FAIL(input_args.size() == kCropAndResizeInputSize,
                               "For primitive[" + prim_name + "], the [x shape-length] should be 4, bug got " +
                                 std::to_string(input_args.size()) + ".");
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    (void)CheckAndConvertUtils::CheckTensorTypeValid(
      "x", input_args[kInputIndex0]->BuildType(),
      {kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64, kUInt8, kUInt16}, prim_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("boxes", input_args[kInputIndex1]->BuildType(), {kFloat32},
                                                     prim_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("box_index", input_args[kInputIndex2]->BuildType(), {kInt32},
                                                     prim_name);
    return kFloat32;
  }

 protected:
  int64_t ParseNumBoxes(const std::vector<AbstractBasePtr> &input_args, const std::string &prim_name) const {
    MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex1]->BuildShape()->isa<abstract::Shape>(),
                               "For primitive[" + prim_name + "], the [boxes] has no abstract::Shape.");
    auto boxes_shape_element = input_args[kInputIndex1]->BuildShape()->cast<abstract::ShapePtr>();
    auto boxes_shape = boxes_shape_element->shape();
    MS_EXCEPTION_IF_CHECK_FAIL(
      boxes_shape.size() == kShapeRank2 || (boxes_shape.size() == 1 && boxes_shape[0] == kUnknownDims),
      "For primitive[" + prim_name + "], the [boxes shape-length] should be 2, bug got " +
        std::to_string(boxes_shape.size()) + ".");

    MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex2]->BuildShape()->isa<abstract::Shape>(),
                               "For primitive[" + prim_name + "], the [box_index] has no abstract::Shape.");
    auto box_index_shape_element = input_args[kInputIndex2]->BuildShape()->cast<abstract::ShapePtr>();
    auto box_index_shape = box_index_shape_element->shape();
    if (box_index_shape.size() != 1) {
      std::string log = "For primitive[" + prim_name + "], the [box_index shape-length] must be 1, but got " +
                        std::to_string(box_index_shape.size()) + ".";
      MS_EXCEPTION(ArgumentError) << log;
    }
    int64_t num_boxes = -1;
    if (boxes_shape[0] >= 0 || box_index_shape[0] >= 0) {
      if (boxes_shape[0] >= 0 && box_index_shape[0] >= 0) {
        MS_EXCEPTION_IF_CHECK_FAIL(
          boxes_shape[0] == box_index_shape[0],
          "For primitive[" + prim_name + "], the [boxes first-dim] must be equal to [box_index first-dim], but got " +
            std::to_string(boxes_shape[0]) + " vs " + std::to_string(box_index_shape[0]) + ".");
      }
      if (boxes_shape.size() == kShapeRank2) {
        MS_EXCEPTION_IF_CHECK_FAIL(boxes_shape[1] == kLimitValue4 || boxes_shape[1] == -1,
                                   "For primitive[" + prim_name + "], the [boxes second-dim] must be 4, but got " +
                                     std::to_string(boxes_shape[1]) + ".");
      }
      num_boxes = std::max(boxes_shape[0], box_index_shape[0]);
    }
    return num_boxes;
  }

 private:
  const int64_t kUnknownDims = -2;
  const int64_t kLimitValue2 = 2;
  const int64_t kLimitValue4 = 4;
  const size_t kCropAndResizeInputSize = 4;
  const size_t kShapeRank2 = 2;
  const size_t kShapeRank4 = 4;
};

void CropAndResize::Init(ResizeMethod method, float extrapolation_value) {
  this->set_method(method);
  this->set_extrapolation_value(extrapolation_value);
}

void CropAndResize::set_method(ResizeMethod method) {
  auto swi = static_cast<int64_t>(method);
  (void)this->AddAttr(kMethod, api::MakeValue(swi));
}

void CropAndResize::set_extrapolation_value(float extrapolation_value) {
  (void)this->AddAttr(kExtrapolationValue, api::MakeValue(extrapolation_value));
}

ResizeMethod CropAndResize::get_method() const {
  auto value_ptr = GetAttr(kMethod);
  return ResizeMethod(GetValue<int64_t>(value_ptr));
}

float CropAndResize::get_extrapolation_value() const {
  auto value_ptr = GetAttr(kExtrapolationValue);
  return GetValue<float>(value_ptr);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(CropAndResize, prim::kPrimCropAndResize, CropAndResizeInfer, false);
}  // namespace ops
}  // namespace mindspore
