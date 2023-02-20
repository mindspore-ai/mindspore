/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/ragged_range.h"

#include <memory>
#include <string>
#include <vector>
#include <set>

#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
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
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
int64_t CalculateShape(const tensor::TensorPtr starts_ptr, const tensor::TensorPtr limits_ptr,
                       const tensor::TensorPtr deltas_ptr, int64_t nrows) {
  T *starts_val = reinterpret_cast<T *>(starts_ptr->data_c());
  T *limits_val = reinterpret_cast<T *>(limits_ptr->data_c());
  T *deltas_val = reinterpret_cast<T *>(deltas_ptr->data_c());
  int64_t shape_size = 0;
  for (int64_t row = 0; row < nrows; ++row) {
    auto start = starts_val[row];
    auto limit = limits_val[row];
    auto delta = deltas_val[row];
    if (delta == static_cast<T>(0)) {
      MS_EXCEPTION(ValueError) << "For RaggedRange, "
                               << "requires input delta != 0.";
    }
    if (((delta < 0) && (limit < start)) || ((delta > 0) && (limit > start))) {
      if (std::is_integral<T>::value) {
        shape_size += ((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
      } else {
        shape_size += static_cast<int64_t>(std::ceil(std::abs((limit - start) / delta)));
      }
    }
  }
  return shape_size;
}
abstract::TupleShapePtr RaggedRangeInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t max_dim = 2;
  auto in_shape_starts = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto in_shape_limits = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto in_shape_deltas =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  // support dynamic rank
  if (IsDynamicRank(in_shape_starts) || IsDynamicRank(in_shape_limits) || IsDynamicRank(in_shape_deltas)) {
    auto unknow_rank_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{unknow_rank_ptr});
  }

  (void)CheckAndConvertUtils::CheckInteger("dimension of RaggedRange input starts", SizeToLong(in_shape_starts.size()),
                                           kLessThan, max_dim, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of RaggedRange input limits", SizeToLong(in_shape_limits.size()),
                                           kLessThan, max_dim, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dimension of RaggedRange input deltas", SizeToLong(in_shape_deltas.size()),
                                           kLessThan, max_dim, prim_name);
  int64_t starts_size = in_shape_starts.size() == 0 ? 1 : in_shape_starts[0];
  int64_t limits_size = in_shape_limits.size() == 0 ? 1 : in_shape_limits[0];
  int64_t deltas_size = in_shape_deltas.size() == 0 ? 1 : in_shape_deltas[0];
  if (!(starts_size == limits_size && starts_size == deltas_size && limits_size == deltas_size)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", starts, limits, and deltas must have the same shape"
                             << ", but got starts (" << starts_size << ",)"
                             << ", limits (" << limits_size << ",)"
                             << ", deltas (" << deltas_size << ",).";
  }
  int64_t nrows = starts_size;
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  const int64_t max_length = GetValue<int64_t>(max_length_ptr);
  if (input_args[0]->isa<abstract::AbstractTensor>() && !input_args[0]->BuildValue()->isa<AnyValue>() &&
      !input_args[0]->BuildValue()->isa<None>()) {
    auto starts = input_args[0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(starts);
    auto starts_value_ptr = starts->BuildValue();
    MS_EXCEPTION_IF_NULL(starts_value_ptr);
    auto starts_tensor = starts_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(starts_tensor);
    auto limits = input_args[1]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(limits);
    auto limits_value_ptr = limits->BuildValue();
    MS_EXCEPTION_IF_NULL(limits_value_ptr);
    auto limits_tensor = limits_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(limits_tensor);
    auto deltas = input_args[kInputIndex2]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(deltas);
    auto deltas_value_ptr = deltas->BuildValue();
    MS_EXCEPTION_IF_NULL(deltas_value_ptr);
    auto deltas_tensor = deltas_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(deltas_tensor);
    auto dtype = starts_tensor->data_type();
    int64_t shape_size = 0;
    if (dtype == kNumberTypeInt32) {
      shape_size = CalculateShape<int32_t>(starts_tensor, limits_tensor, deltas_tensor, nrows);
    } else if (dtype == kNumberTypeInt64) {
      shape_size = CalculateShape<int64_t>(starts_tensor, limits_tensor, deltas_tensor, nrows);
    } else if (dtype == kNumberTypeFloat32) {
      shape_size = CalculateShape<float>(starts_tensor, limits_tensor, deltas_tensor, nrows);
    } else if (dtype == kNumberTypeFloat64) {
      shape_size = CalculateShape<double>(starts_tensor, limits_tensor, deltas_tensor, nrows);
    } else {
      MS_LOG(EXCEPTION) << "RaggedRange has unsupported dataType: " << dtype << ".";
    }
    ShapeVector rt_dense_values_shape_vec = {};
    ShapeVector rt_nested_splits_shape_vec = {};
    rt_dense_values_shape_vec.push_back(shape_size);
    rt_nested_splits_shape_vec.push_back(nrows + 1);
    if (rt_dense_values_shape_vec[0] > max_length) {
      MS_EXCEPTION(ValueError) << "For RaggedRange"
                               << ", the number of elements of output must be less than max length: " << max_length
                               << ", but got " << rt_dense_values_shape_vec[0]
                               << "! The shape of output must be reduced or max_length must be increased.";
    }
    if (rt_nested_splits_shape_vec[0] > max_length) {
      MS_EXCEPTION(ValueError) << "For RaggedRange"
                               << ", the number of elements of output must be less than max length: " << max_length
                               << ", but got " << rt_nested_splits_shape_vec[0]
                               << "! The shape of output must be reduced or max_length must be increased.";
    }
    abstract::ShapePtr rt_dense_values_shape = std::make_shared<abstract::Shape>(rt_dense_values_shape_vec);
    abstract::ShapePtr rt_nested_splits_shape = std::make_shared<abstract::Shape>(rt_nested_splits_shape_vec);
    return (std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{rt_nested_splits_shape, rt_dense_values_shape}));
  }
  std::vector<int64_t> rt_nested_splits_shape_vec = {abstract::Shape::kShapeDimAny};
  std::vector<int64_t> rt_dense_values_shape_vec = {abstract::Shape::kShapeDimAny};
  abstract::ShapePtr rt_nested_splits_shape = std::make_shared<abstract::Shape>(rt_nested_splits_shape_vec);
  abstract::ShapePtr rt_dense_values_shape = std::make_shared<abstract::Shape>(rt_dense_values_shape_vec);
  return (std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{rt_nested_splits_shape, rt_dense_values_shape}));
}
TuplePtr RaggedRangeInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto starts_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(starts_type);
  auto limits_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(limits_type);
  auto deltas_type = input_args[kInputIndex2]->BuildType();
  MS_EXCEPTION_IF_NULL(deltas_type);
  if (!(starts_type->isa<TensorType>())) {
    MS_EXCEPTION(TypeError) << "For " << prim_name << ", the input starts must be a Tensor, but got "
                            << starts_type->ToString() << ".";
  }
  if (!(limits_type->isa<TensorType>())) {
    MS_EXCEPTION(TypeError) << "For " << prim_name << ", the input limits must be a Tensor, but got "
                            << limits_type->ToString() << ".";
  }
  if (!(deltas_type->isa<TensorType>())) {
    MS_EXCEPTION(TypeError) << "For " << prim_name << ", the input deltas must be a Tensor, but got "
                            << deltas_type->ToString() << ".";
  }
  if (!((starts_type->ToString() == limits_type->ToString()) && (starts_type->ToString() == deltas_type->ToString()) &&
        (limits_type->ToString() == deltas_type->ToString()))) {
    MS_EXCEPTION(TypeError) << "For " << prim_name << ", starts, limits, and deltas must have the same type, "
                            << "but got starts " << starts_type->ToString() << ", limits " << limits_type->ToString()
                            << ", deltas " << deltas_type->ToString() << ".";
  }
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("starts", starts_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("limits", deltas_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("deltas", limits_type, valid_types, prim_name);
  auto Tsplits = primitive->GetAttr("Tsplits");
  MS_EXCEPTION_IF_NULL(Tsplits);
  auto infer_type = Tsplits->cast<TypePtr>();
  auto rt_nested_splits_type = infer_type;
  auto rt_dense_values_type = starts_type;
  return std::make_shared<Tuple>(std::vector<TypePtr>{rt_nested_splits_type, rt_dense_values_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(RaggedRange, BaseOperator);
AbstractBasePtr RaggedRangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto shape = RaggedRangeInferShape(primitive, input_args);
  auto type = RaggedRangeInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGRaggedRangeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RaggedRangeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RaggedRangeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RaggedRangeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RaggedRange, prim::kPrimRaggedRange, AGRaggedRangeInfer, false);
}  // namespace ops
}  // namespace mindspore
