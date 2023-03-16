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
#include "ops/range_v2.h"

#include <memory>
#include <set>
#include <vector>
#include <type_traits>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
#define IsSameType(source_type, cmp_type) (cmp_type->equal(source_type))
#define IsNoneOrAnyValue(value_ptr) ((value_ptr->isa<None>()) || (value_ptr->isa<ValueAny>()))
constexpr auto op_name = "RangeV2";

template <typename T>
int64_t RangeV2CalculateShape(const tensor::TensorPtr start_ptr, const tensor::TensorPtr limit_ptr,
                              const tensor::TensorPtr delta_ptr) {
  // DataSize == 1
  if (start_ptr->DataSize() != 1) {
    MS_EXCEPTION(TypeError) << "For RangeV2, start must a scalar but element number more than 1.";
  }
  if (limit_ptr->DataSize() != 1) {
    MS_EXCEPTION(TypeError) << "For RangeV2, limit must a scalar but element number more than 1.";
  }
  if (delta_ptr->DataSize() != 1) {
    MS_EXCEPTION(TypeError) << "For RangeV2, delta must a scalar but element number more than 1.";
  }
  T start = *(reinterpret_cast<T *>(start_ptr->data_c()));
  T limit = *(reinterpret_cast<T *>(limit_ptr->data_c()));
  T delta = *(reinterpret_cast<T *>(delta_ptr->data_c()));
  bool valid_value = (delta == T(0) || (delta > 0 && start > limit) || (delta < 0 && start < limit));
  if (valid_value) {
    if (delta == T(0)) {
      MS_EXCEPTION(ValueError) << "For RangeV2, delta cannot be equal to zero.";
    }
    if (delta > 0 && start > limit) {
      MS_EXCEPTION(ValueError) << "For RangeV2, delta cannot be positive when limit < start.";
    }
    if (delta < 0 && start < limit) {
      MS_EXCEPTION(ValueError) << "For RangeV2, delta cannot be negative when limit > start.";
    }
  }
  int64_t shape_size = 0;
  if (std::is_integral<T>::value) {
    shape_size = static_cast<int64_t>((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
  } else {
    shape_size = static_cast<int64_t>(std::ceil(std::abs((limit - start) / delta)));
  }
  return shape_size;
}

abstract::ShapePtr RangeV2CheckAndInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive->GetAttr(kMaxLen));
  // support dynamic rank
  auto start_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto limit_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto delta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamicRank(start_shape) || IsDynamicRank(limit_shape) || IsDynamicRank(delta_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  int64_t shape_size = abstract::Shape::kShapeDimAny;
  auto start_value = input_args[kInputIndex0]->BuildValue();
  auto limit_value = input_args[kInputIndex1]->BuildValue();
  auto delta_value = input_args[kInputIndex2]->BuildValue();
  MS_EXCEPTION_IF_NULL(start_value);
  MS_EXCEPTION_IF_NULL(limit_value);
  MS_EXCEPTION_IF_NULL(delta_value);

  bool is_compile = (IsNoneOrAnyValue(start_value) && IsNoneOrAnyValue(limit_value) && IsNoneOrAnyValue(delta_value));
  // not in compile, need inferShape
  if (!is_compile) {
    auto dtype = CheckAndConvertUtils::GetTensorInputType(op_name, input_args, kInputIndex0);
    auto start_tensor = start_value->cast<tensor::TensorPtr>();
    auto limit_tensor = limit_value->cast<tensor::TensorPtr>();
    auto delta_tensor = delta_value->cast<tensor::TensorPtr>();
    if (IsSameType(dtype, kInt) || IsSameType(dtype, kInt32)) {
      shape_size = RangeV2CalculateShape<int32_t>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kInt64)) {
      shape_size = RangeV2CalculateShape<int64_t>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kFloat) || IsSameType(dtype, kFloat32)) {
      shape_size = RangeV2CalculateShape<float>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kFloat64)) {
      shape_size = RangeV2CalculateShape<double>(start_tensor, limit_tensor, delta_tensor);
    } else {
      MS_EXCEPTION(TypeError) << "For RangeV2, the dtype of input must be int32, int64, float32, float64, but got "
                              << dtype->meta_type() << ".";
    }
    if (shape_size < 0) {
      MS_EXCEPTION(ValueError) << "For RangeV2, infer shape error, shape_size [" << shape_size << "] is negative.";
    }
  }

  ShapeVector out_shape = {};
  if (is_compile) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  (void)out_shape.emplace_back(shape_size);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr RangeV2CheckAndInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::set<TypePtr> support_types = {kInt32, kInt64, kFloat32, kFloat64};
  auto start_type = CheckAndConvertUtils::CheckTensorTypeValid("start", input_args[kInputIndex0]->BuildType(),
                                                               support_types, prim->name());
  auto limit_type = CheckAndConvertUtils::CheckTensorTypeValid("limit", input_args[kInputIndex1]->BuildType(),
                                                               support_types, prim->name());
  auto delta_type = CheckAndConvertUtils::CheckTensorTypeValid("delta", input_args[kInputIndex1]->BuildType(),
                                                               support_types, prim->name());
  MS_EXCEPTION_IF_NULL(start_type);
  MS_EXCEPTION_IF_NULL(limit_type);
  MS_EXCEPTION_IF_NULL(delta_type);
  bool same_type = IsSameType(start_type, limit_type) && IsSameType(limit_type, delta_type);
  if (!same_type) {
    MS_EXCEPTION(TypeError) << "For RangeV2, start, limit delta should have same type, but get start["
                            << start_type->meta_type() << "], limit[" << limit_type->meta_type() << "], delta["
                            << delta_type->meta_type() << "].";
  }
  return start_type;
}
}  // namespace

AbstractBasePtr RangeV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex1);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex2);
  // infer type must in before
  auto infer_type = RangeV2CheckAndInferType(primitive, input_args);
  auto infer_shape = RangeV2CheckAndInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(RangeV2, BaseOperator);

// AG means auto generated
class MIND_API AGRangeV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RangeV2CheckAndInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RangeV2CheckAndInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RangeV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RangeV2, prim::kPrimRangeV2, AGRangeV2Infer, false);
}  // namespace ops
}  // namespace mindspore
