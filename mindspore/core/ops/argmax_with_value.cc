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

#include "ops/argmax_with_value.h"

#include <set>
#include <map>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
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
int64_t ArgMaxWithValue::axis() const {
  auto value_ptr = GetAttr("axis");
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

bool ArgMaxWithValue::keep_dims() const {
  auto value_ptr = GetAttr("keep_dims");
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

namespace {
abstract::TupleShapePtr ArgMaxWithValueInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_shape_ptr = input_args[0]->BuildShape();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr);
  auto x_shape = x_shape_map[kShape];
  auto axis = GetValue<int64_t>(primitive->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(primitive->GetAttr("keep_dims"));
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape_ptr, x_shape_ptr});
  }
  auto x_rank = static_cast<int64_t>(x_shape.size());
  if (x_rank == 0) {
    if (axis != -1 && axis != 0) {
      MS_EXCEPTION(ValueError) << "For ArgMaxWithValue with 0d input tensor, axis must be one of 0 or -1, but got "
                               << axis << ".";
    }
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape_ptr, x_shape_ptr});
  }

  if (axis < -x_rank || axis >= x_rank) {
    MS_EXCEPTION(ValueError) << "For ArgMaxWithValue, axis must be in range [" << -x_rank << ", " << x_rank
                             << "), but got " << axis << ".";
  }
  if (axis < 0) {
    axis += x_rank;
  }
  (void)primitive->AddAttr("dimension", MakeValue(axis));
  // Calculate all the shapes.
  auto cal_shape = [axis, keep_dims](ShapeVector &shape, const ShapeVector &x_shape) -> void {
    (void)shape.insert(shape.end(), x_shape.begin(), x_shape.end());
    if (keep_dims) {
      shape[LongToSize(axis)] = 1;
    } else {
      (void)shape.erase(shape.begin() + axis);
    }
  };
  ShapeVector output_shape;
  cal_shape(output_shape, x_shape);

  auto index_and_value_shape = std::make_shared<abstract::Shape>(output_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{index_and_value_shape, index_and_value_shape});
}

TuplePtr ArgMaxWithValueInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt8,   kInt16, kInt32,
                                         kInt64,   kUInt8,   kUInt16,  kUInt32, kUInt64};
  TypePtr input_x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, valid_types, prim->name());
  auto index_type = std::make_shared<TensorType>(kInt32);
  return std::make_shared<Tuple>(std::vector<TypePtr>{index_type, input_x_type});
}
}  // namespace
AbstractBasePtr ArgMaxWithValueInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto shapes = ArgMaxWithValueInferShape(primitive, input_args);
  auto types = ArgMaxWithValueInferType(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
MIND_API_OPERATOR_IMPL(ArgMaxWithValue, BaseOperator);

// AG means auto generated
class MIND_API AGArgMaxWithValueInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ArgMaxWithValueInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ArgMaxWithValueInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ArgMaxWithValueInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ArgMaxWithValue, prim::kPrimArgMaxWithValue, AGArgMaxWithValueInfer, false);
}  // namespace ops
}  // namespace mindspore
