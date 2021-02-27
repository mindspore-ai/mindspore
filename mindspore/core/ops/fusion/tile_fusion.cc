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

#include "ops/fusion/tile_fusion.h"
#include <string>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void TileFusion::Init(const std::vector<int64_t> &dims) { this->set_dims(dims); }

void TileFusion::set_dims(const std::vector<int64_t> &dims) { this->AddAttr(kDims, MakeValue(dims)); }

std::vector<int64_t> TileFusion::get_dims() const {
  auto value_ptr = GetAttr(kDims);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameTileFusion, TileFusion);
}  // namespace ops
}  // namespace mindspore
