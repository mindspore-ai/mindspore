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
#include "ops/space_to_batch_nd.h"
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
  auto space_prim = primitive->cast<PrimSpaceToBatchNDPtr>();
  MS_EXCEPTION_IF_NULL(space_prim);
  auto prim_name = space_prim->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("input_x rank", x_shape.size(), kEqual, 4, prim_name);
  auto out_shape = x_shape;
  int64_t block_shape_prod = 1;
  const int64_t offset = 2;
  auto block_shape = space_prim->get_block_shape();
  auto padding = space_prim->get_paddings();
  int64_t size = block_shape.size();
  for (int64_t i = 0; i < size; i++) {
    int64_t padded = out_shape[i + offset] + padding[i][0] + padding[i][1];
    if (padded % block_shape[i] != 0) {
      MS_EXCEPTION(ValueError) << prim_name << " padded[" << i << "]" << padded << "should be divisible by block_shape["
                               << i << "]" << block_shape[i];
    }
    out_shape[i + offset] = int64_t(floor(padded / block_shape[i]));
    block_shape_prod = block_shape_prod * block_shape[i];
  }
  out_shape[0] = out_shape[0] * block_shape_prod;
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  return infer_type;
}
}  // namespace

void SpaceToBatchND::set_paddings(std::vector<std::vector<int64_t>> paddings) {
  CheckAndConvertUtils::CheckInteger(kPaddings, paddings.size(), kEqual, 2, this->name());
  int64_t h = paddings.size();
  int64_t w = paddings[0].size();
  std::vector<int64_t> temp_w = {2, 2};
  CheckAndConvertUtils::Check(kPaddings, {h, w}, kEqual, "paddings_shape(2,2)", temp_w, this->name());
  for (int64_t i = 0; i < h; i++) {
    for (int64_t j = 0; j < w; j++) {
      CheckAndConvertUtils::CheckInteger(kPaddings, paddings[i][j], kGreaterEqual, 0, this->name());
    }
  }
  this->AddAttr(kPaddings, MakeValue(paddings));
}

std::vector<std::vector<int64_t>> SpaceToBatchND::get_paddings() const {
  auto value_ptr = GetAttr(kPaddings);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}
void SpaceToBatchND::set_block_shape(std::vector<int64_t> block_shape) {
  CheckAndConvertUtils::CheckInteger(kBlockShape, block_shape.size(), kEqual, 2, this->name());
  for (int64_t i = 0; i < (int64_t)block_shape.size(); i++) {
    CheckAndConvertUtils::CheckInteger(kBlockShape, block_shape[i], kGreaterEqual, 1, this->name());
  }
  this->AddAttr(kBlockShape, MakeValue(block_shape));
}

std::vector<int64_t> SpaceToBatchND::get_block_shape() const {
  auto value_ptr = GetAttr(kBlockShape);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void SpaceToBatchND::Init(std::vector<int64_t> block_shape, std::vector<std::vector<int64_t>> paddings) {
  this->set_paddings(paddings);
  this->set_block_shape(block_shape);
}

AbstractBasePtr SpaceToBatchNDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameSpaceToBatchND, SpaceToBatchND);
}  // namespace ops
}  // namespace mindspore
