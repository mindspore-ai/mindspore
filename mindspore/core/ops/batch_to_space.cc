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
  (void)this->AddAttr(kBlockSize, MakeValue(block_size));
}

std::vector<int64_t> BatchToSpace::get_block_size() const {
  auto value_ptr = this->GetAttr(kBlockSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void BatchToSpace::set_crops(const std::vector<std::vector<int64_t>> &crops) {
  (void)this->AddAttr(kCrops, MakeValue(crops));
}

std::vector<std::vector<int64_t>> BatchToSpace::get_crops() const {
  auto value_ptr = this->GetAttr(kCrops);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameBatchToSpace, BatchToSpace);
}  // namespace ops
}  // namespace mindspore
