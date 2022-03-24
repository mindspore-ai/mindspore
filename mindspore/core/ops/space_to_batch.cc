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
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void SpaceToBatch::set_paddings(const std::vector<std::vector<int64_t>> &paddings) {
  (void)this->AddAttr(kPaddings, api::MakeValue(paddings));
  int64_t h = SizeToLong(paddings.size());
  int64_t w = SizeToLong(paddings[0].size());
  std::vector<int64_t> temp_w = {2, 2};
  CheckAndConvertUtils::Check(kPaddings, {h, w}, kEqual, temp_w, this->name());
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
  (void)this->AddAttr(kBlockSize, api::MakeValue(block_size));
}

std::vector<int64_t> SpaceToBatch::get_block_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kBlockSize));
}

void SpaceToBatch::Init(const std::vector<int64_t> block_size, const std::vector<std::vector<int64_t>> &paddings) {
  this->set_paddings(paddings);
  this->set_block_size(block_size);
}

MIND_API_BASE_IMPL(SpaceToBatch, PrimitiveC, BaseOperator);
REGISTER_PRIMITIVE_C(kNameSpaceToBatch, SpaceToBatch);
}  // namespace ops
}  // namespace mindspore
