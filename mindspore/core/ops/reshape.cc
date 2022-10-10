/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <string>
#include <memory>
#include <functional>
#include <set>
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
ShapeVector update_shape(std::vector<int> padding_axis_value, ShapeVector x_shape, ShapeVector shape) {
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
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];

    std::vector<int64_t> output_shape;
    size_t normal_inputs_number = 2;
    if (input_args.size() == normal_inputs_number) {
      auto input_y = input_args[1];
      MS_EXCEPTION_IF_NULL(input_y);
      auto y_value = input_y->BuildValue();
      MS_EXCEPTION_IF_NULL(y_value);
      if (input_y->isa<abstract::AbstractTensor>()) {
        if (y_value->isa<tensor::Tensor>()) {
          output_shape = CheckAndConvertUtils::CheckTensorIntValue("shape", y_value, prim_name);
        } else {
          abstract::ShapePtr y_shape = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 1);
          auto shape_value = y_shape->shape();
          if (shape_value.size() != 1) {
            MS_EXCEPTION(TypeError) << "For '" << prim_name
                                    << "', the shape size must be 1, but got: " << shape_value.size() << ".";
          }
          if (y_shape->IsDynamic()) {
            output_shape.push_back(abstract::Shape::kShapeRankAny);
            return std::make_shared<abstract::Shape>(output_shape);
          } else if (x_shape.size() == 0) {
            MS_LOG(DEBUG) << "x is a scalar.";
            MS_LOG(DEBUG) << "the size of 'shape' is: " << shape_value[0];
            for (int i = 0; i < SizeToInt(shape_value[0]); i++) {
              output_shape.push_back(1);
            }
            return std::make_shared<abstract::Shape>(output_shape);
          } else {
            auto y_tensor = input_y->cast<abstract::AbstractTensorPtr>();
            auto tensor_shape_value = y_tensor->get_shape_value();
            if (tensor_shape_value == nullptr) {
              output_shape = ShapeVector(shape_value[0], abstract::Shape::kShapeDimAny);
              return std::make_shared<abstract::Shape>(output_shape);
            } else {
              auto shape_vector = GetValue<ShapeVector>(tensor_shape_value);
              MS_EXCEPTION_IF_CHECK_FAIL(LongToSize(shape_value[0]) == shape_vector.size(),
                                         "Illegal shape of shape value");
              output_shape = shape_vector;
              if (std::count_if(output_shape.begin(), output_shape.end(), [](int64_t s) { return s < 0; }) > 1) {
                return std::make_shared<abstract::Shape>(output_shape);
              }
            }
          }
        }
      } else if (input_y->isa<abstract::AbstractTuple>()) {
        output_shape = CheckAndConvertUtils::CheckTupleInt("input[shape]", y_value, primitive->name());
      } else {
        MS_EXCEPTION(TypeError) << "input_y must be AbstractTensor or AbstractTuple, but got: " << input_y;
      }
    } else {
      ValuePtr sh = primitive->GetAttr("shape");
      MS_EXCEPTION_IF_NULL(sh);
      if (sh->isa<ValueTuple>()) {
        auto reshape_value_tuple = sh->cast<ValueTuplePtr>();
        MS_EXCEPTION_IF_NULL(reshape_value_tuple);
        auto reshape_tuple = reshape_value_tuple->value();
        (void)std::transform(std::begin(reshape_tuple), std::end(reshape_tuple), std::back_inserter(output_shape),
                             [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
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
    auto x_dtype = input_args[0]->BuildType();
    std::set<TypePtr> template_types = {kTensorType};
    (void)CheckAndConvertUtils::CheckSubClass("x_dtype", x_dtype, template_types, prim->name());
    return x_dtype;
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(Reshape, prim::kPrimReshape, ReshapeInfer, false);
}  // namespace ops
}  // namespace mindspore
