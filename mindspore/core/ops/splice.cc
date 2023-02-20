/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/splice.h"

#include <vector>

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Splice, BaseOperator);
void Splice::Init(const std::vector<int64_t> &contexts, const std::vector<int64_t> &forward_indexes,
                  int64_t output_dims) {
  this->set_context(contexts);
  this->set_forward_indexes(forward_indexes);
  this->set_output_dim(output_dims);
}

void Splice::set_context(const std::vector<int64_t> &contexts) {
  (void)this->AddAttr(kSpliceContext, api::MakeValue(contexts));
}

void Splice::set_forward_indexes(const std::vector<int64_t> &forward_indexes) {
  (void)this->AddAttr(kSpliceForwardIndexes, api::MakeValue(forward_indexes));
}

void Splice::set_output_dim(int64_t output_dim) { (void)this->AddAttr(kSpliceOutputDims, api::MakeValue(output_dim)); }

std::vector<int64_t> Splice::get_context() const {
  auto value_ptr = GetAttr(kSpliceContext);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Splice::get_forward_indexes() const {
  auto value_ptr = GetAttr(kSpliceForwardIndexes);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Splice::get_output_dim() const {
  auto value_ptr = GetAttr(kSpliceOutputDims);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameSplice, Splice);
}  // namespace ops
}  // namespace mindspore
