/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/reverse_sequence.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "mindspore/lite/nnacl/fp32/reverse_sequence_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateReverseSequenceParameter(const mindspore::lite::PrimitiveC *primitive) {
  ReverseSequenceParameter *reverse_sequence_param =
    reinterpret_cast<ReverseSequenceParameter *>(malloc(sizeof(ReverseSequenceParameter)));
  if (reverse_sequence_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseSequenceParameter failed.";
    return nullptr;
  }
  memset(reverse_sequence_param, 0, sizeof(ReverseSequenceParameter));
  auto param =
    reinterpret_cast<mindspore::lite::ReverseSequence *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  reverse_sequence_param->op_parameter_.type_ = primitive->Type();
  reverse_sequence_param->seq_axis_ = param->GetSeqAxis();
  reverse_sequence_param->batch_axis_ = param->GetBatchAxis();
  return reinterpret_cast<OpParameter *>(reverse_sequence_param);
}
Registry ReverseSequenceParameterRegistry(schema::PrimitiveType_ReverseSequence, PopulateReverseSequenceParameter);

}  // namespace lite
}  // namespace mindspore
