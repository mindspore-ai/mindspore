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
void SpaceToBatchND::set_paddings(std::vector<std::vector<int64_t>> paddings) {
  const int64_t pad_size = 2;
  (void)CheckAndConvertUtils::CheckInteger(kPaddings, SizeToLong(paddings.size()), kEqual, pad_size, this->name());
  size_t h = paddings.size();
  size_t w = paddings[0].size();
  std::vector<size_t> temp_w = {2, 2};
  CheckAndConvertUtils::Check(kPaddings, {h, w}, kEqual, temp_w, this->name());
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      (void)CheckAndConvertUtils::CheckInteger(kPaddings, paddings[i][j], kGreaterEqual, 0LL, this->name());
    }
  }
  (void)this->AddAttr(kPaddings, MakeValue(paddings));
}

std::vector<std::vector<int64_t>> SpaceToBatchND::get_paddings() const {
  auto value_ptr = GetAttr(kPaddings);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}
void SpaceToBatchND::set_block_shape(std::vector<int64_t> block_shape) {
  const int64_t block_size = 2;
  (void)CheckAndConvertUtils::CheckInteger(kBlockShape, SizeToLong(block_shape.size()), kEqual, block_size,
                                           this->name());
  for (size_t i = 0; i < block_shape.size(); i++) {
    (void)CheckAndConvertUtils::CheckInteger(kBlockShape, block_shape[i], kGreaterEqual, 1LL, this->name());
  }
  (void)this->AddAttr(kBlockShape, MakeValue(block_shape));
}

std::vector<int64_t> SpaceToBatchND::get_block_shape() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kBlockShape));
}

void SpaceToBatchND::Init(const std::vector<int64_t> block_shape, const std::vector<std::vector<int64_t>> paddings) {
  this->set_paddings(paddings);
  this->set_block_shape(block_shape);
}

REGISTER_PRIMITIVE_C(kNameSpaceToBatchND, SpaceToBatchND);
}  // namespace ops
}  // namespace mindspore
