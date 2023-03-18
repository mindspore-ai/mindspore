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
#include "ops/range.h"

#include <memory>
#include <set>
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
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void Range::set_d_type(const int64_t d_type) { (void)this->AddAttr(kDType, api::MakeValue(d_type)); }

int64_t Range::get_d_type() const {
  auto value_ptr = GetAttr(kDType);
  return GetValue<int64_t>(value_ptr);
}

void Range::set_start(const int64_t start) { (void)this->AddAttr(kStart, api::MakeValue(start)); }

int64_t Range::get_start() const { return GetValue<int64_t>(GetAttr(kStart)); }

int64_t Range::get_maxlen() const { return GetValue<int64_t>(GetAttr(kMaxLen)); }

void Range::set_limit(const int64_t limit) { (void)this->AddAttr(kLimit, api::MakeValue(limit)); }

int64_t Range::get_limit() const {
  auto value_ptr = GetAttr(kLimit);
  return GetValue<int64_t>(value_ptr);
}

void Range::set_delta(const int64_t delta) { (void)this->AddAttr(kDelta, api::MakeValue(delta)); }

int64_t Range::get_delta() const {
  auto value_ptr = GetAttr(kDelta);
  return GetValue<int64_t>(value_ptr);
}

void Range::Init(const int64_t d_type, const int64_t start, const int64_t limit, const int64_t delta) {
  this->set_d_type(d_type);
  this->set_start(start);
  this->set_limit(limit);
  this->set_delta(delta);
}

namespace {
#define IsSameType(source_type, cmp_type) (cmp_type->equal(source_type))
#define IsNoneOrAnyValue(value_ptr) ((value_ptr->isa<None>()) || (value_ptr->isa<ValueAny>()))
template <typename T>
int64_t RangeCalculateShape(const tensor::TensorPtr start_ptr, const tensor::TensorPtr limit_ptr,
                            const tensor::TensorPtr delta_ptr) {
  T start = *(reinterpret_cast<T *>(start_ptr->data_c()));
  T limit = *(reinterpret_cast<T *>(limit_ptr->data_c()));
  T delta = *(reinterpret_cast<T *>(delta_ptr->data_c()));
  bool valid_value = (delta == T(0) || (delta > 0 && start > limit) || (delta < 0 && start < limit));
  if (valid_value) {
    if (delta == T(0)) {
      MS_EXCEPTION(ValueError) << "For Range, delta cannot be equal to zero.";
    }
    if (delta > 0 && start > limit) {
      MS_EXCEPTION(ValueError) << "For Range, delta cannot be positive when limit < start.";
    }
    if (delta < 0 && start < limit) {
      MS_EXCEPTION(ValueError) << "For Range, delta cannot be negative when limit > start.";
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

abstract::ShapePtr RangeCheckAndInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  int64_t shape_size = abstract::Shape::kShapeDimAny;
  auto start_value = input_args[kInputIndex0]->BuildValue();
  auto limit_value = input_args[kInputIndex1]->BuildValue();
  auto delta_value = input_args[kInputIndex2]->BuildValue();
  MS_EXCEPTION_IF_NULL(start_value);
  MS_EXCEPTION_IF_NULL(limit_value);
  MS_EXCEPTION_IF_NULL(delta_value);

  bool is_compile = (IsNoneOrAnyValue(start_value) || IsNoneOrAnyValue(limit_value) || IsNoneOrAnyValue(delta_value));
  // not in compile, need inferShape
  if (!is_compile) {
    auto op_name = "Range";
    auto dtype = CheckAndConvertUtils::GetTensorInputType(op_name, input_args, kInputIndex0);
    auto start_tensor = start_value->cast<tensor::TensorPtr>();
    auto limit_tensor = limit_value->cast<tensor::TensorPtr>();
    auto delta_tensor = delta_value->cast<tensor::TensorPtr>();
    if (IsSameType(dtype, kInt) || IsSameType(dtype, kInt32)) {
      shape_size = RangeCalculateShape<int32_t>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kInt64)) {
      shape_size = RangeCalculateShape<int64_t>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kFloat) || IsSameType(dtype, kFloat32)) {
      shape_size = RangeCalculateShape<float>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kFloat64)) {
      shape_size = RangeCalculateShape<double>(start_tensor, limit_tensor, delta_tensor);
    } else {
      MS_EXCEPTION(TypeError) << "For Range, the dtype of input must be int32, int64, float32, float64, but got "
                              << dtype->meta_type() << ".";
    }
    if (shape_size < 0) {
      MS_EXCEPTION(ValueError) << "For Range, infer shape error, shape_size [" << shape_size << "] is negative.";
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

TypePtr RangeCheckAndInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::set<TypePtr> support_types = {kInt32, kInt64, kFloat32, kFloat64};
  auto start_type = CheckAndConvertUtils::CheckTensorTypeValid("start", input_args[kInputIndex0]->BuildType(),
                                                               support_types, prim->name());
  auto limit_type = CheckAndConvertUtils::CheckTensorTypeValid("limit", input_args[kInputIndex1]->BuildType(),
                                                               support_types, prim->name());
  auto delta_type = CheckAndConvertUtils::CheckTensorTypeValid("delta", input_args[kInputIndex2]->BuildType(),
                                                               support_types, prim->name());
  MS_EXCEPTION_IF_NULL(start_type);
  MS_EXCEPTION_IF_NULL(limit_type);
  MS_EXCEPTION_IF_NULL(delta_type);
  bool same_type = IsSameType(start_type, limit_type) && IsSameType(limit_type, delta_type);
  if (!same_type) {
    MS_EXCEPTION(TypeError) << "For Range, start, limit delta should have same type, but get start["
                            << start_type->meta_type() << "], limit[" << limit_type->meta_type() << "], delta["
                            << delta_type->meta_type() << "].";
  }
  return start_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Range, BaseOperator);
AbstractBasePtr RangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  const int kInputIndex0 = 0;
  const int kInputIndex1 = 1;
  const int kInputIndex2 = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex1);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex2);
  // infer type must in before
  auto infer_type = RangeCheckAndInferType(primitive, input_args);
  auto infer_shape = RangeCheckAndInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGRangeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RangeCheckAndInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RangeCheckAndInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RangeInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0, 1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Range, prim::kPrimRange, AGRangeInfer, false);
}  // namespace ops
}  // namespace mindspore
