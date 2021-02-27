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

#include "ops/batch_to_space.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void BatchToSpace::Init(const std::vector<int64_t> &block_size, const std::vector<std::vector<int64_t>> &crops) {
  this->set_block_size(block_size);
  this->set_crops(crops);
}

void BatchToSpace::set_block_size(const std::vector<int64_t> &block_size) {
  this->AddAttr(kBlockSize, MakeValue(block_size));
}

std::vector<int64_t> BatchToSpace::get_block_size() const {
  auto value_ptr = this->GetAttr(kBlockSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void BatchToSpace::set_crops(const std::vector<std::vector<int64_t>> &crops) {
  this->AddAttr(kCrops, MakeValue(crops));
}

std::vector<std::vector<int64_t>> BatchToSpace::get_crops() const {
  auto value_ptr = this->GetAttr(kCrops);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}

AbstractBasePtr BatchToSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim = primitive->cast<PrimBatchToSpacePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), common_valid_types, prim_name);

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  CheckAndConvertUtils::CheckInteger("x rank", x_shape.size(), kEqual, 4, prim_name);
  auto block_size = prim->get_block_size();
  auto crops = prim->get_crops();
  auto out_shape = x_shape;
  for (size_t i = 0; i < 2; ++i) {
    auto x_block_prod = out_shape[i + 2] * block_size[i];
    auto crops_sum = crops[i][0] + crops[i][1];
    CheckAndConvertUtils::Check("x block shape prod", x_block_prod, kGreaterThan, "crops sum", 4, prim_name);
    out_shape[i + 2] = x_block_prod - crops_sum;
  }
  CheckAndConvertUtils::CheckInteger("x_shape[0] % (block_size[0]*block_size[1])",
                                     out_shape[0] % (block_size[0] * block_size[1]), kEqual, 0, prim_name);
  out_shape[0] /= block_size[0] * block_size[1];

  auto ret = input_args[0]->Broaden();
  ret->set_shape(std::make_shared<abstract::Shape>(out_shape));
  return ret;
}
REGISTER_PRIMITIVE_C(kNameBatchToSpace, BatchToSpace);
}  // namespace ops
}  // namespace mindspore
