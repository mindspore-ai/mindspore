/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "infer/cxx_api/tile_fusion.h"

#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(TileFusion, BaseOperator);
void TileFusion::Init(const std::vector<int64_t> &dims) { this->set_dims(dims); }

void TileFusion::set_dims(const std::vector<int64_t> &dims) { (void)this->AddAttr(kDims, api::MakeValue(dims)); }

std::vector<int64_t> TileFusion::get_dims() const {
  auto value_ptr = GetAttr(kDims);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameTileFusion, TileFusion);
}  // namespace ops
}  // namespace mindspore
