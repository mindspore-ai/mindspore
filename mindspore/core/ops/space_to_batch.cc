/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  const int64_t x_rank = 4;
  (void)CheckAndConvertUtils::CheckInteger("x rank", SizeToLong(input_shape.size()), kEqual, x_rank, prim_name);
  std::vector<int64_t> output_shape(input_shape.size());
  auto block_shape_vector = GetValue<std::vector<int64_t>>(primitive->GetAttr(kBlockSize));
  auto paddings = GetValue<std::vector<std::vector<int64_t>>>(primitive->GetAttr(kPaddings));
  const size_t kDimsOffset = 2;
  for (size_t i = 0; i < kDimsOffset; i++) {
    auto padded = output_shape[i + kDimsOffset] + paddings[i][0] + paddings[i][1];
    const int64_t input_num = 0;
    (void)CheckAndConvertUtils::CheckInteger("padded shape", padded % SizeToLong(block_shape_vector.size()), kEqual,
                                             input_num, prim_name);
    output_shape[i + kDimsOffset] = padded / SizeToLong(block_shape_vector.size());
  }
  output_shape[0] *= SizeToLong(block_shape_vector.size() * block_shape_vector.size());
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
}
}  // namespace
void SpaceToBatch::set_paddings(const std::vector<std::vector<int64_t>> &paddings) {
  (void)this->AddAttr(kPaddings, MakeValue(paddings));
  int64_t h = SizeToLong(paddings.size());
  int64_t w = SizeToLong(paddings[0].size());
  std::vector<int64_t> temp_w = {2, 2};
  CheckAndConvertUtils::Check(kPaddings, {h, w}, kEqual, "paddings_shape(2,2)", temp_w, this->name());
  for (size_t i = 0; i < LongToSize(h); i++) {
    for (size_t j = 0; j < LongToSize(w); j++) {
      (void)CheckAndConvertUtils::CheckInteger(kPadding, paddings[i][j], kGreaterEqual, 0, this->name());
    }
  }
}

std::vector<std::vector<int64_t>> SpaceToBatch::get_paddings() const {
  auto value_ptr = GetAttr(kPaddings);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}
void SpaceToBatch::set_block_size(const std::vector<int64_t> block_size) {
  (void)this->AddAttr(kBlockSize, MakeValue(block_size));
}

std::vector<int64_t> SpaceToBatch::get_block_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kBlockSize));
}

void SpaceToBatch::Init(const std::vector<int64_t> block_size, const std::vector<std::vector<int64_t>> &paddings) {
  this->set_paddings(paddings);
  this->set_block_size(block_size);
}
AbstractBasePtr SpaceToBatchInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameSpaceToBatch, SpaceToBatch);
}  // namespace ops
}  // namespace mindspore
