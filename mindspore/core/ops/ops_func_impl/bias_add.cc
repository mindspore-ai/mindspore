/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/bias_add.h"

#include <unordered_map>
#include <utility>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t x_min_rank = 2;
constexpr int64_t x_max_rank = 5;
static std::unordered_map<mindspore::Format, int64_t> ChannelDimMap{
  {Format::NHWC, -1}, {Format::NCHW, 1}, {Format::NCDHW, 1}};

void CheckShapeValid(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_shape_base = input_args[kInputIndex0]->GetShape();
  auto bias_shape_base = input_args[kInputIndex1]->GetShape();
  const auto &input_shape = input_shape_base->GetShapeVector();
  const auto &bias_shape = bias_shape_base->GetShapeVector();
  MS_CHECK_VALUE(bias_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "rank of bias", SizeToLong(bias_shape.size()), kEqual, 1, primitive));
  auto input_rank = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(input_rank >= x_min_rank && input_rank <= x_max_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input_x", input_rank, kIncludeBoth,
                                                             {x_min_rank, x_max_rank}, primitive));

  if (IsDynamic(bias_shape)) {
    return;
  }

  auto data_format_ptr = input_args[kInputIndex2]->GetValue();
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  auto data_format_opt = GetScalarValue<int64_t>(data_format_ptr);
  if (!data_format_opt.has_value()) {
    return;
  }

  auto data_format = static_cast<mindspore::Format>(data_format_opt.value());
  auto it = ChannelDimMap.find(data_format);
  if (it == ChannelDimMap.end()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the data_format must be NCHW, NHWC, or NCDHW, but got: " << data_format << ".";
  }

  auto channel_dim = (input_rank + it->second) % input_rank;
  if (input_shape[channel_dim] != abstract::Shape::kShapeDimAny && bias_shape[0] != input_shape[channel_dim]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', bias[0] shape should be equal to input_x[" << channel_dim
                             << "] shape when data_format is " << data_format << ", but got bias shape: ." << bias_shape
                             << ", input_shape: " << input_shape << ".";
  }
}
}  // namespace

BaseShapePtr BiasAddFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_base = input_args[kInputIndex0]->GetShape();
  auto bias_shape_base = input_args[kInputIndex1]->GetShape();
  const auto &input_shape = input_shape_base->GetShapeVector();
  const auto &bias_shape = bias_shape_base->GetShapeVector();
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  CheckShapeValid(primitive, input_args);
  auto data_format_ptr = input_args[kInputIndex2]->GetValue();
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  auto data_format_opt = GetScalarValue<int64_t>(data_format_ptr);
  if (!data_format_opt.has_value()) {
    return input_shape_base->Clone();
  }
  auto data_format = data_format_opt.value();
  // infer channel dim by bias dim
  ShapeVector output_shape(input_shape);
  if (!IsDynamic(bias_shape)) {
    if ((data_format == Format::NCHW || data_format == Format::NCDHW) &&
        input_shape[1] == abstract::Shape::kShapeDimAny) {
      output_shape[1] = bias_shape[0];
    }
    if (data_format == Format::NHWC && input_shape.back() == abstract::Shape::kShapeDimAny) {
      output_shape.back() = bias_shape[0];
    }
  }
  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}
TypePtr BiasAddFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[0]->GetType();
  return input_type->Clone();
}

int32_t BiasAddFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto data_format_ptr = input_args[kInputIndex2]->GetValue();
  MS_EXCEPTION_IF_NULL(data_format_ptr);
  auto data_format_opt = GetScalarValue<int64_t>(data_format_ptr);
  if (!data_format_opt.has_value()) {
    return OP_CHECK_RETRY;
  }
  auto data_format = static_cast<mindspore::Format>(data_format_opt.value());
  if (data_format != Format::NCHW && data_format != Format::NHWC && data_format != Format::NCDHW) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the data_format must be NCHW, NHWC, or NCDHW, but got: " << data_format << ".";
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
