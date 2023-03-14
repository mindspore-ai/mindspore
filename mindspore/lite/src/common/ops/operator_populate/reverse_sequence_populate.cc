/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/reverse_sequence_parameter.h"
#include "ops/reverse_sequence.h"
using mindspore::ops::kNameReverseSequence;
using mindspore::schema::PrimitiveType_ReverseSequence;

namespace mindspore {
namespace lite {
OpParameter *PopulateReverseSequenceOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ReverseSequenceParameter *>(PopulateOpParameter<ReverseSequenceParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ReverseSequenceParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::ReverseSequence *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not ReverseSequence.";
    return nullptr;
  }

  param->seq_axis_ = static_cast<int>(op->get_seq_dim());
  param->batch_axis_ = static_cast<int>(op->get_batch_dim());
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameReverseSequence, PrimitiveType_ReverseSequence, PopulateReverseSequenceOpParameter);
}  // namespace lite
}  // namespace mindspore
