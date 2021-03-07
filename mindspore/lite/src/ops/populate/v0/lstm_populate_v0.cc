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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/lstm_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateLstmParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto lstm_prim = primitive->value_as_Lstm();
  LstmParameter *lstm_param = reinterpret_cast<LstmParameter *>(malloc(sizeof(LstmParameter)));
  if (lstm_param == nullptr) {
    MS_LOG(ERROR) << "malloc LstmParameter failed.";
    return nullptr;
  }
  memset(lstm_param, 0, sizeof(LstmParameter));
  lstm_param->op_parameter_.type_ = schema::PrimitiveType_LSTM;

  if (lstm_prim == nullptr) {
    free(lstm_param);
    MS_LOG(ERROR) << "get Lstm param nullptr.";
    return nullptr;
  }
  lstm_param->bidirectional_ = lstm_prim->bidirection();
  return reinterpret_cast<OpParameter *>(lstm_param);
}
}  // namespace

Registry g_lstmV0ParameterRegistry(schema::v0::PrimitiveType_Lstm, PopulateLstmParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
