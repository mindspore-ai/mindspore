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

#include "ops/expand.h"

#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
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
template <typename T>
std::vector<int64_t> ExpandInferOutShape(std::vector<int64_t> output_shape, std::vector<int64_t> x_shape,
                                         const int64_t x_shape_size, const int64_t shape_size,
                                         const tensor::TensorPtr shape_tensor, int64_t max_length,
                                         const string prim_name) {
  auto input_shape_ptr = reinterpret_cast<T *>(shape_tensor->data_c());
  int64_t shape_m = 1;
  if (shape_size >= x_shape_size) {
    int64_t sub = shape_size - x_shape_size;
    for (int i = 0; i < shape_size; i = i + 1) {
      if (i >= sub) {
        if (x_shape[LongToSize(i - sub)] != input_shape_ptr[i]) {
          if (input_shape_ptr[i] != -1) {
            if (x_shape[LongToSize(i - sub)] != 1) {
              MS_EXCEPTION(ValueError) << "For " << prim_name << ", the expanded size of the tensor ("
                                       << std::to_string(input_shape_ptr[i]) << ") must be equal to the existing size ("
                                       << std::to_string(x_shape[LongToSize(i - sub)]) << ") which is not 1 at dim ("
                                       << std::to_string(i) << ").";
            }
          } else {
            output_shape.push_back(x_shape[LongToSize(i - sub)]);
            shape_m *= static_cast<int64_t>(input_shape_ptr[i]);
            continue;
          }
        }
      } else if (i < sub && input_shape_ptr[i] == -1) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", the expanded size of the tensor (" << std::to_string(-1)
                                 << ") isn't allowed in a leading, non-existing dimension " << std::to_string(i) << ".";
      }
      output_shape.push_back(input_shape_ptr[i]);
      shape_m *= static_cast<int64_t>(input_shape_ptr[i]);
    }
  } else {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the size of shape provided (" << std::to_string(shape_size)
                             << ") must be greater than or equal to the size of tensor x's shape ("
                             << std::to_string(x_shape_size) << ").";
  }
  if (shape_m > max_length) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", the number of elements of output must be less than max length: " << max_length
                             << ", but got " << shape_m
                             << "! The shape of  output should be reduced or max_length should be increased.";
  }
  return output_shape;
}

abstract::ShapePtr ExpandInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(shape_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto shape = input_args[1]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_value_ptr = shape->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_value_ptr);
  auto shape_tensor = shape_value_ptr->cast<tensor::TensorPtr>();
  auto shape_ptr = std::make_shared<abstract::Shape>(shape_shape);
  auto shape_v = shape_ptr->shape();

  const int64_t shape_dim = 1;
  if (shape_v.size() != shape_dim) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the input tensor 'shape' must be a 1-D tensor.";
  }
  const int64_t x_shape_size = SizeToLong(x_shape.size());
  const int64_t shape_size = shape_v[0];
  auto shape_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(shape_type);
  auto shape_type_id = shape_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(shape_type_id);
  auto shape_type_element = shape_type_id->element();
  MS_EXCEPTION_IF_NULL(shape_type_element);
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  if (!input_args[1]->BuildValue()->isa<ValueAny>() && !input_args[1]->BuildValue()->isa<None>()) {
    std::vector<int64_t> output_shape;
    if (shape_type_element->type_id() == kNumberTypeInt16) {
      output_shape = ExpandInferOutShape<int16_t>(output_shape, x_shape, x_shape_size, shape_size, shape_tensor,
                                                  max_length, prim_name);
    } else if (shape_type_element->type_id() == kNumberTypeInt32) {
      output_shape = ExpandInferOutShape<int32_t>(output_shape, x_shape, x_shape_size, shape_size, shape_tensor,
                                                  max_length, prim_name);
    } else if (shape_type_element->type_id() == kNumberTypeInt64) {
      output_shape = ExpandInferOutShape<int64_t>(output_shape, x_shape, x_shape_size, shape_size, shape_tensor,
                                                  max_length, prim_name);
    }
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    std::vector<int64_t> output_shape;
    for (int i = 0; i < shape_size; i++) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr ExpandInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  TypePtr x_type = input_args[0]->BuildType();
  TypePtr shape_type = input_args[1]->BuildType();
  std::set<TypePtr> x_valid_types = {kFloat16, kFloat32, kInt32, kInt8, kUInt8};
  std::set<TypePtr> shape_valid_types = {kInt16, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, x_valid_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", shape_type, shape_valid_types, primitive->name());
  return x_type;
}
}  // namespace

AbstractBasePtr ExpandInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = ExpandInferType(primitive, input_args);
  auto infer_shape = ExpandInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Expand, BaseOperator);

// AG means auto generated
class MIND_API AGExpandInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ExpandInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ExpandInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ExpandInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Expand, prim::kPrimExpand, AGExpandInfer, false);
}  // namespace ops
}  // namespace mindspore
