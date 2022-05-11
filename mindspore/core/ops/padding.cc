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

#include "ops/padding.h"
#include <vector>
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr PaddingInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,   kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBool};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, name);
}

abstract::ShapePtr PaddingInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t kInputSize = 1;
  constexpr int64_t kNumber1 = 1;
  constexpr int64_t kNumber2 = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInteger("x rank", x_rank, kGreaterEqual, kNumber2, prim_name);
  int64_t x_last_dim = x_shape[x_shape.size() - 1];
  if (x_last_dim != kNumber1) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', the last dimension of 'x' must be 1, but got: " << x_last_dim
                      << ".";
  }

  auto value_ptr = primitive->GetAttr(kPadDimSize);
  auto pad_dim_size = GetValue<int64_t>(value_ptr);
  CheckAndConvertUtils::CheckInteger("pad_dim_size", pad_dim_size, kGreaterEqual, kNumber1, prim_name);
  std::vector<int64_t> out_dim_shape;
  (void)std::transform(x_shape.begin(), x_shape.end(), std::back_inserter(out_dim_shape), SizeToLong);
  // Extends the last dimension of the input tensor from 1 to pad_dim_size.
  out_dim_shape[out_dim_shape.size() - 1] += pad_dim_size - 1;
  return std::make_shared<abstract::Shape>(out_dim_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Padding, BaseOperator);

void Padding::Init(int64_t pad_dim_size) { set_pad_dim_size(pad_dim_size); }

void Padding::set_pad_dim_size(int64_t pad_dim_size) { (void)AddAttr(kPadDimSize, api::MakeValue(pad_dim_size)); }

int64_t Padding::get_pad_dim_size() const {
  auto value_ptr = GetAttr(kPadDimSize);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr PaddingInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  TypePtr output_type = PaddingInferType(primitive, input_args);
  abstract::ShapePtr output_shape = PaddingInferShape(primitive, input_args);
  return abstract::MakeAbstract(output_shape, output_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Padding, prim::kPrimPadding, PaddingInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
