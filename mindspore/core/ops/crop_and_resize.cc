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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(CropAndResize, BaseOperator);
class CropAndResizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("[input] number", static_cast<int64_t>(input_args.size()), kEqual,
                                             kCropAndResizeInputSize, prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }

    MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex0]->BuildShape()->isa<abstract::Shape>(),
                               "For primitive[" + prim_name + "], the [x] has no abstract:Shape.");
    auto x_shape = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>()->shape();
    MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex1]->BuildShape()->isa<abstract::Shape>(),
                               "For primitive[" + prim_name + "], the [boxes] has no abstract:Shape.");
    auto box_shape = input_args[kInputIndex1]->BuildShape()->cast<abstract::ShapePtr>()->shape();
    MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex2]->BuildShape()->isa<abstract::Shape>(),
                               "For primitive[" + prim_name + "], the [box_index] has no abstract:Shape.");
    auto box_index_shape = input_args[kInputIndex2]->BuildShape()->cast<abstract::ShapePtr>()->shape();
    if (IsDynamicRank(x_shape) || IsDynamicRank(box_shape) || IsDynamicRank(box_index_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    }
    if (IsDynamic(x_shape) || IsDynamic(box_shape) || IsDynamic(box_index_shape)) {
      return std::make_shared<abstract::Shape>(
        std::vector<int64_t>{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                             abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny});
    }

    auto x_dims = static_cast<int64_t>(x_shape.size());
    (void)CheckAndConvertUtils::CheckInteger("[x] rank", x_dims, kEqual, kShapeRank4, prim_name);
    int64_t out_channel = x_shape.back();

    auto num_boxes = ParseNumBoxes(box_shape, box_index_shape, prim_name);
    auto crop_size_type = input_args[kInputIndex3]->BuildType();
    MS_EXCEPTION_IF_CHECK_FAIL(crop_size_type != nullptr,
                               "For primitive[" + prim_name + "], the [crop_size] typeid is a nullptr.");
    auto value_ptr = input_args[kInputIndex3]->BuildValue();
    MS_EXCEPTION_IF_NULL(value_ptr);
    if (!IsValueKnown(value_ptr)) {
      return std::make_shared<abstract::Shape>(
        ShapeVector{num_boxes, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, out_channel});
    }

    std::vector<int64_t> crop_size;
    if (crop_size_type->isa<TensorType>()) {
      crop_size = CheckAndConvertUtils::CheckTensorIntValue("crop_size", value_ptr, prim_name);
    } else if (IsIdentidityOrSubclass(crop_size_type, kTuple)) {
      auto value_tuple = value_ptr->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple);
      const auto &elements = value_tuple->value();
      for (const auto &element : elements) {
        if (element->isa<Int64Imm>()) {
          crop_size.push_back(GetValue<int64_t>(element));
        } else {
          auto type = element->type();
          std::string real_type_str = type == nullptr ? "Unknown." : type->ToString() + ".";
          MS_EXCEPTION(TypeError) << "For primitive[" << prim_name
                                  << "], the [crop_size] must be a tuple with two Int elements, but got "
                                  << real_type_str;
        }
      }
    } else {
      MS_EXCEPTION(TypeError) << "For primitive[" + prim_name
                              << "], the [crop_size] is must be a Tuple with two Int elements, but got "
                              << crop_size_type->ToString();
    }
    (void)CheckAndConvertUtils::CheckInteger("[crop_size] length", static_cast<int64_t>(crop_size.size()), kEqual,
                                             kShapeRank2, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("[crop] height", crop_size[0], kGreaterThan, 0, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("[crop] width", crop_size.back(), kGreaterThan, 0, prim_name);
    ShapeVector out_shape = {num_boxes, crop_size[0], crop_size.back(), out_channel};
    return std::make_shared<abstract::Shape>(out_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("[input] number", static_cast<int64_t>(input_args.size()), kEqual,
                                             kCropAndResizeInputSize, prim_name);
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

  std::set<int64_t> GetValueDependArgIndices() const override { return {3}; }

 protected:
  int64_t ParseNumBoxes(const ShapeVector &box_shape, const ShapeVector &box_index_shape,
                        const std::string &prim_name) const {
    int64_t box_dims = static_cast<int64_t>(box_shape.size());
    (void)CheckAndConvertUtils::CheckInteger("[boxes] rank", box_dims, kEqual, kShapeRank2, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("[boxes] dim_1", box_shape.back(), kEqual, kLimitValue4, prim_name);

    int64_t box_index_dims = static_cast<int64_t>(box_index_shape.size());
    (void)CheckAndConvertUtils::CheckInteger("[box_index] rank", box_index_dims, kEqual, 1, prim_name);
    if (box_shape[0] != box_index_shape[0]) {
      MS_EXCEPTION(ValueError) << "For primitive[" + prim_name +
                                    "], the [boxes] dim_0 must be equal to [box_index] dim_0, but got " +
                                    std::to_string(box_shape[0]) + " vs " + std::to_string(box_index_shape[0]) + ".";
    }
    return box_shape[0];
  }

 private:
  const int64_t kLimitValue4 = 4;
  const int64_t kCropAndResizeInputSize = 4;
  const int64_t kShapeRank2 = 2;
  const int64_t kShapeRank4 = 4;
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
