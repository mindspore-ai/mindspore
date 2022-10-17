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
#include <memory>
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "ops/segment_arithmetic.h"
#include "ops/segment_mean.h"
#include "ops/segment_max.h"
#include "ops/segment_min.h"
#include "ops/segment_prod.h"
#include "ops/segment_sum.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void CheckSegmentIDDataMean(const T *segment_ids_data, const size_t data_size) {
  if (segment_ids_data[0] < 0) {
    MS_EXCEPTION(ValueError) << "For 'SegmentMean', the values of segment_ids must be nonnegative. but got "
                             << segment_ids_data[0] << ".";
  }
  for (size_t i = 0; i < data_size - 1; ++i) {
    if (segment_ids_data[i] > segment_ids_data[i + 1]) {
      MS_EXCEPTION(ValueError)
        << "For 'SegmentMean', segment_ids must be a tensor with element values sorted in ascending order.";
    }
  }
}
}  // namespace

namespace {
abstract::ShapePtr SegmentArithmeticInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto segment_ids_shape_ptr = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(segment_ids_shape_ptr);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  auto segment_ids_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(segment_ids_shape_ptr)[kShape];
  (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", SizeToLong(x_shape.size()), kGreaterEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of 'segment_ids'", SizeToLong(segment_ids_shape.size()), kEqual, 1,
                                           prim_name);
  if (segment_ids_shape[0] != x_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the amount of data for segment_ids must be equal to the first dimension of the "
                                "shape of input_x, but got "
                             << segment_ids_shape[0] << " and " << x_shape[0] << ".";
  }
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  ShapeVector out_shape(x_shape);
  auto segment_ids_ptr = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(segment_ids_ptr);
  if (!segment_ids_ptr->isa<AnyValue>() && !segment_ids_ptr->isa<None>()) {
    auto segment_ids_tensor = segment_ids_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(segment_ids_tensor);
    auto data_size = segment_ids_tensor->DataSize();
    auto segment_ids_type_id = segment_ids_tensor->data_type();
    if (segment_ids_type_id == kNumberTypeInt64) {
      int64_t *segment_ids_data = static_cast<int64_t *>(segment_ids_tensor->data_c());
      CheckSegmentIDDataMean<int64_t>(segment_ids_data, data_size);
      out_shape[0] = static_cast<int64_t>(segment_ids_data[data_size - 1] + 1);
    } else if (segment_ids_type_id == kNumberTypeInt32) {
      int32_t *segment_ids_data = static_cast<int32_t *>(segment_ids_tensor->data_c());
      CheckSegmentIDDataMean<int32_t>(segment_ids_data, data_size);
      out_shape[0] = static_cast<int64_t>(segment_ids_data[data_size - 1] + 1);
    }
    uint32_t length = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      length *= static_cast<uint32_t>(out_shape[i]);
    }
    if (length > max_length) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', The number of elements of output must be less than max length: " << max_length
                               << ", but got " << length
                               << "! The shape of output should be reduced or max_length should be increased";
    }
  } else {
    out_shape[0] = abstract::Shape::kShapeDimAny;
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr SegmentArithmeticInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  TypePtr x_type = input_args[0]->BuildType();
  TypePtr segment_ids_type = input_args[1]->BuildType();
  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32, kFloat64, kInt8,   kInt16,  kComplex128, kInt32,
                                           kInt64,   kUInt8,   kUInt16,  kUInt32, kUInt64, kComplex64};
  const std::set<TypePtr> segment_ids_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x_type", x_type, x_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("segment_ids_type", segment_ids_type, segment_ids_valid_types,
                                                   prim_name);
  return x_type;
}
}  // namespace

AbstractBasePtr SegmentArithmeticInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto type = SegmentArithmeticInferType(primitive, input_args);
  auto shape = SegmentArithmeticInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

MIND_API_OPERATOR_IMPL(SegmentMax, BaseOperator);
MIND_API_OPERATOR_IMPL(SegmentMin, BaseOperator);
MIND_API_OPERATOR_IMPL(SegmentMean, BaseOperator);
MIND_API_OPERATOR_IMPL(SegmentProd, BaseOperator);
MIND_API_OPERATOR_IMPL(SegmentSum, BaseOperator);

REGISTER_PRIMITIVE_EVAL_IMPL(SegmentMax, prim::kPrimSegmentMax, SegmentArithmeticInfer, nullptr, true);
REGISTER_PRIMITIVE_EVAL_IMPL(SegmentMin, prim::kPrimSegmentMin, SegmentArithmeticInfer, nullptr, true);
REGISTER_PRIMITIVE_EVAL_IMPL(SegmentMean, prim::kPrimSegmentMean, SegmentArithmeticInfer, nullptr, true);
REGISTER_PRIMITIVE_EVAL_IMPL(SegmentProd, prim::kPrimSegmentProd, SegmentArithmeticInfer, nullptr, true);
REGISTER_PRIMITIVE_EVAL_IMPL(SegmentSum, prim::kPrimSegmentSum, SegmentArithmeticInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
