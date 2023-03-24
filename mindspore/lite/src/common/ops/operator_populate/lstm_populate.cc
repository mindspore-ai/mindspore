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
#include "nnacl/fp32/lstm_fp32.h"
#include "ops/lstm.h"
using mindspore::ops::kNameLSTM;
using mindspore::schema::PrimitiveType_LSTM;
namespace mindspore {
namespace lite {
OpParameter *PopulateLstmOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<LstmParameter *>(PopulateOpParameter<LstmParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new LstmParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::LSTM *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not LSTM.";
    free(param);
    return nullptr;
  }
  param->bidirectional_ = op->get_bidirectional();
  param->zoneout_cell_ = op->get_zoneout_cell();
  param->zoneout_hidden_ = op->get_zoneout_hidden();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameLSTM, PrimitiveType_LSTM, PopulateLstmOpParameter)
}  // namespace lite
}  // namespace mindspore
