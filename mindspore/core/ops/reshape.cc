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
#include "ops/reshape.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kShapeFromTensor = "shape_from_tensor";
constexpr size_t kSize2 = 2;

ShapeVector UpdateOutputShape(const ShapeVector &x_shape, const ShapeVector &y_shape) {
  if (IsDynamicShape(y_shape)) {
    ShapeVector ret = ShapeVector{std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>())};
    if (y_shape.size() == kSize2) {
      ret = ShapeVector{std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>()) / y_shape[1],
                        y_shape[1]};
    }
    return ret;
  }
  return y_shape;
}
}  // namespace

ShapeVector update_shape(const std::vector<int> &padding_axis_value, const ShapeVector &x_shape, ShapeVector shape) {
  // padding_axis_value is for the condition that the number of -1 in shape > 1, , but just paddingshape
  if (std::any_of(x_shape.begin(), x_shape.end(), [](const int &shape_i) { return shape_i < 0; })) {
    return shape;
  }
  int64_t x_num = 1;
  for (int64_t value : x_shape) {
    x_num = LongMulWithOverflowCheck(value, x_num);
  }

  auto it_first = find(shape.begin(), shape.end(), -1);
  if (it_first != shape.end()) {
    if (!padding_axis_value.empty()) {
      // the condition that the number of -1 in shape is > 1, but just paddingshape
      for (size_t index = 0; index < padding_axis_value.size(); ++index) {
        shape[IntToSize(padding_axis_value[index])] = x_shape[index];
      }
    } else {
      auto it_second = find(it_first + 1, shape.end(), -1);
      if (it_second != shape.end()) {
        MS_EXCEPTION(ValueError) << "At most one component of input shape can be -1, but got " << shape;
      }
      auto index = LongToSize(std::distance(shape.begin(), it_first));
      int64_t infer_value = x_num;
      for (size_t i = 0; i < shape.size(); ++i) {
        int64_t value = shape[i];
        if (value != -1 && value != 0) {
          infer_value = infer_value / value;
        }
      }
      shape[index] = infer_value;
    }
  }

  int64_t shape_num = 1;
  for (int64_t value : shape) {
    shape_num = LongMulWithOverflowCheck(value, shape_num);
  }
  if (shape_num == abstract::Shape::kShapeRankAny) {
    return {abstract::Shape::kShapeRankAny};
  }
  if (shape_num != x_num) {
    MS_EXCEPTION(ValueError) << "The accumulate of x_shape must be equal to out_shape, but got x_shape: " << x_shape
                             << ", and out_shape: " << shape;
  }
  return shape;
}

MIND_API_OPERATOR_IMPL(Reshape, BaseOperator);
class ReshapeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr int64_t empty_tensor_num = 0;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterThan, empty_tensor_num, prim_name);
    constexpr size_t max_size = 2;
    (void)CheckAndConvertUtils::CheckValue<size_t>("input size", input_args.size(), kLessEqual, max_size, prim_name);
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
    std::vector<int64_t> output_shape;
    if (input_args.size() == max_size) {
      auto input_y = input_args[1];
      MS_EXCEPTION_IF_NULL(input_y);
      auto value = input_y->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      output_shape = GetShapeValue(primitive, input_y);
      if (primitive->HasAttr(kShapeFromTensor) && GetValue<bool>(primitive->GetAttr(kShapeFromTensor))) {
        if (input_y->isa<abstract::AbstractTensor>() && value->isa<tensor::Tensor>()) {
          auto y_shape = CheckAndConvertUtils::CheckTensorIntValue("y_shape", value, prim_name);
          output_shape = UpdateOutputShape(x_shape, y_shape);
        }
      }

      const int64_t kSelfComputedDim = -1;
      const int64_t kMaxSelfComputedDimCount = 1;

      auto self_computed_dim_count = std::count(output_shape.begin(), output_shape.end(), kSelfComputedDim);
      if (!IsValueKnown(value) && self_computed_dim_count > kMaxSelfComputedDimCount) {
        return std::make_shared<abstract::Shape>(output_shape);
      }
    } else {
      // When the shape is passed as an attribute, shape should be constant.
      ValuePtr sh = primitive->GetAttr("shape");
      MS_EXCEPTION_IF_NULL(sh);
      if (sh->isa<ValueTuple>()) {
        auto reshape_value_tuple = sh->cast<ValueTuplePtr>();
        MS_EXCEPTION_IF_NULL(reshape_value_tuple);
        auto reshape_tuple = reshape_value_tuple->value();
        (void)std::transform(reshape_tuple.begin(), reshape_tuple.end(), std::back_inserter(output_shape),
                             [=](const ValuePtr &e) -> int64_t {
                               if (!e->isa<Int64Imm>()) {
                                 MS_EXCEPTION(TypeError)
                                   << "For primitive[" << prim_name << "], the 'shape'"
                                   << " must be a tuple with all Int elements, but got " << sh->ToString();
                               }
                               return GetValue<int64_t>(e);
                             });
      } else if (sh->isa<tensor::Tensor>()) {
        output_shape = CheckAndConvertUtils::CheckTensorIntValue("shape", sh, "Reshape");
      } else {
        MS_EXCEPTION(ValueError)
          << "In stage of execution， the primitive[Reshape]'s input['shape'] must be a tuple or "
          << "constant Tensor.";
      }
    }

    std::vector<int> padding_axis_value;
    ValuePtr padding_axis = primitive->GetAttr("reshape_padding_axis");
    if (padding_axis != nullptr) {
      padding_axis_value = GetValue<std::vector<int>>(padding_axis);
    }
    ShapeVector updated_shape = update_shape(padding_axis_value, x_shape, output_shape);
    return std::make_shared<abstract::Shape>(updated_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    constexpr int64_t empty_tensor_num = 0;
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterThan, empty_tensor_num, prim->name());
    auto x_dtype = input_args[0]->BuildType();
    std::set<TypePtr> template_types = {kTensorType};
    (void)CheckAndConvertUtils::CheckSubClass("x_dtype", x_dtype, template_types, prim->name());
    return x_dtype;
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(Reshape, prim::kPrimReshape, ReshapeInfer, false);
}  // namespace ops
}  // namespace mindspore
