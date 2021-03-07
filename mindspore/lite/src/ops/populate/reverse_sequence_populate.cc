/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/reverse_sequence_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateReverseSequenceParameter(const void *prim) {
  ReverseSequenceParameter *reverse_sequence_param =
    reinterpret_cast<ReverseSequenceParameter *>(malloc(sizeof(ReverseSequenceParameter)));
  if (reverse_sequence_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseSequenceParameter failed.";
    return nullptr;
  }
  memset(reverse_sequence_param, 0, sizeof(ReverseSequenceParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto param = primitive->value_as_ReverseSequence();
  reverse_sequence_param->op_parameter_.type_ = primitive->value_type();
  reverse_sequence_param->seq_axis_ = static_cast<int>(param->seq_dim());
  reverse_sequence_param->batch_axis_ = static_cast<int>(param->batch_dim());
  return reinterpret_cast<OpParameter *>(reverse_sequence_param);
}
}  // namespace

Registry ReverseSequenceParameterRegistry(schema::PrimitiveType_ReverseSequence, PopulateReverseSequenceParameter,
                                          SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
