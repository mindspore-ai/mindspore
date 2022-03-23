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

#include <set>
#include <vector>
#include <memory>
#include "ops/reverse_sequence.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(ReverseSequence, PrimitiveC, BaseOperator);
void ReverseSequence::Init(const int64_t seq_dim, const int64_t batch_dim) {
  this->set_seq_dim(seq_dim);
  this->set_batch_dim(batch_dim);
}
void ReverseSequence::set_seq_dim(const int64_t seq_dim) { (void)this->AddAttr(kSeqDim, api::MakeValue(seq_dim)); }
void ReverseSequence::set_batch_dim(const int64_t batch_dim) {
  (void)this->AddAttr(kBatchDim, api::MakeValue(batch_dim));
}

int64_t ReverseSequence::get_seq_dim() const { return GetValue<int64_t>(GetAttr(kSeqDim)); }
int64_t ReverseSequence::get_batch_dim() const {
  auto value_ptr = this->GetAttr(kBatchDim);
  return GetValue<int64_t>(value_ptr);
}

REGISTER_PRIMITIVE_C(kNameReverseSequence, ReverseSequence);
}  // namespace ops
}  // namespace mindspore
