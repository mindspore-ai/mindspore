/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/conv2d_padding.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "primitive/op_name.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConv2dPaddingInputArgsSize = 7;
}  // namespace

ShapeArray Conv2DPaddingFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  if (input_infos.size() != kConv2dPaddingInputArgsSize) {
    MS_LOG(EXCEPTION) << "input args size should be  " << kConv2dPaddingInputArgsSize << ", but got "
                      << input_infos.size();
  }

  auto &input_tensor = input_infos[idxes_.input_idx];
  auto &weight_tensor = input_infos[idxes_.weight_idx];
  auto input_shape = input_tensor->GetShape();
  auto weight_shape = weight_tensor->GetShape();
  auto prim_name = primitive->name();
  auto padding_opt = input_infos[idxes_.padding_idx]->GetScalarValue<int64_t>();

  const auto rank_2d_isbatch = 4;
  const auto rank_2d_notbatch = 3;
  if (input_tensor->IsDynamicRank() || weight_tensor->IsDynamicRank()) {
    if (MS_LIKELY(!input_tensor->IsDynamicRank())) {
      auto input_rank = SizeToLong(input_shape.size());
      MS_CHECK_VALUE(input_rank == rank_2d_isbatch || input_rank == rank_2d_notbatch,
                     CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("rank of input", input_rank, kIncludeBoth,
                                                                          {3, 4}, primitive));
      auto [batched_input_shape, is_batched] = Batchify(input_shape, 2, primitive->name());
      batched_input_shape[kIndex1] = abstract::Shape::kShapeDimAny;
      if (padding_opt.has_value()) {
        mindspore::PadMode padding_enum_value = static_cast<mindspore::PadMode>(padding_opt.value());
        if (padding_enum_value != PadMode::SAME) {
          batched_input_shape[kIndex2] = abstract::Shape::kShapeDimAny;
          batched_input_shape[kIndex3] = abstract::Shape::kShapeDimAny;
        }
      }
      if (!is_batched) {
        batched_input_shape.erase(batched_input_shape.begin());
      }
      return {batched_input_shape};
    }
    std::vector<int64_t> output_shape = {abstract::Shape::kShapeRankAny};
    return {output_shape};
  }

  auto input_rank = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(
    input_rank == rank_2d_isbatch || input_rank == rank_2d_notbatch,
    CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("rank of input", input_rank, kIncludeBoth, {3, 4}, primitive));
  int64_t weight_rank = SizeToLong(weight_shape.size());
  MS_CHECK_VALUE(weight_rank == rank_2d_isbatch,
                 CheckAndConvertUtils::CheckInteger("weight rank", weight_rank, kEqual, rank_2d_isbatch, prim_name));
  return ConvNdInferShape(primitive, input_infos, input_shape, weight_shape);
}
}  // namespace ops
}  // namespace mindspore
