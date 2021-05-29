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
#include "ops/space_to_batch.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/infer_base.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto spacetobatch_prim = primitive->cast<PrimSpaceToBatchPtr>();
  MS_EXCEPTION_IF_NULL(spacetobatch_prim);
  auto prim_name = spacetobatch_prim->name();
  auto input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("input shape", SizeToLong(input_shape.size()), kEqual, 4, prim_name);
  std::vector<int64_t> output_shape(input_shape.size());
  auto block_shape_vector = spacetobatch_prim->get_block_size();
  auto paddings = spacetobatch_prim->get_paddings();
  for (size_t i = 0; i < 2; i++) {
    auto padded = LongToSize(output_shape[i + 2] + paddings[i][0] + paddings[i][1]);
    CheckAndConvertUtils::CheckInteger("padded shape", padded % block_shape_vector.size(), kEqual, 0, prim_name);
    output_shape[i + 2] = padded / block_shape_vector.size();
  }
  output_shape[0] *= block_shape_vector.size() * block_shape_vector.size();
  return std::make_shared<abstract::Shape>(output_shape);
}
}  // namespace
void SpaceToBatch::set_paddings(const std::vector<std::vector<int64_t>> &paddings) {
  this->AddAttr(kPaddings, MakeValue(paddings));
  int64_t h = SizeToLong(paddings.size());
  int64_t w = SizeToLong(paddings[0].size());
  std::vector<int64_t> temp_w = {2, 2};
  CheckAndConvertUtils::Check(kPaddings, {h, w}, kEqual, "paddings_shape(2,2)", temp_w, this->name());
  for (size_t i = 0; i < LongToSize(h); i++) {
    for (size_t j = 0; j < LongToSize(w); j++) {
      CheckAndConvertUtils::CheckInteger(kPadding, paddings[i][j], kGreaterEqual, 0, this->name());
    }
  }
}

std::vector<std::vector<int64_t>> SpaceToBatch::get_paddings() const {
  auto value_ptr = GetAttr(kPaddings);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}
void SpaceToBatch::set_block_size(const std::vector<int64_t> block_size) {
  this->AddAttr(kBlockSize, MakeValue(block_size));
}

std::vector<int64_t> SpaceToBatch::get_block_size() const {
  auto value_ptr = GetAttr(kBlockSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void SpaceToBatch::Init(const std::vector<int64_t> block_size, const std::vector<std::vector<int64_t>> &paddings) {
  this->set_paddings(paddings);
  this->set_block_size(block_size);
}
AbstractBasePtr SpaceToBatchInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  size_t input_num = 1;
  auto type = InferBase::CheckSameInferType(primitive, input_args, common_valid_types, input_num);
  return std::make_shared<abstract::AbstractTensor>(type, InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameSpaceToBatch, SpaceToBatch);
}  // namespace ops
}  // namespace mindspore
