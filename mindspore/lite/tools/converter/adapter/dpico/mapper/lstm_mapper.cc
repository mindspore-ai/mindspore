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

#include "mapper/lstm_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/lstm_operator.h"

namespace mindspore {
namespace dpico {
constexpr int kNums1 = 1;
constexpr int kNums2 = 2;
STATUS LstmMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                       const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto lstm_operator = std::make_unique<mapper::LstmOperator>();
  if (lstm_operator == nullptr) {
    MS_LOG(ERROR) << "lstm_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, lstm_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  // Caffe
  if (api::GetValue<string>(prim->GetAttr("type")) == "Lstm") {
    if (prim->GetAttr(kNumOutput) != nullptr) {
      lstm_operator->SetRecurrentNumOutput(static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kNumOutput))));
    }
    if (prim->GetAttr(kExposeHidden) != nullptr) {
      lstm_operator->SetRecurrentExposeHidden(api::GetValue<bool>(prim->GetAttr(kExposeHidden)));
    }
    if (SetRecurrentDataInfo(cnode, lstm_operator.get()) != RET_OK) {
      MS_LOG(ERROR) << "set lstm data info failed.";
      return RET_ERROR;
    }
  }

  // ONNX
  if (api::GetValue<string>(prim->GetAttr("type")) == "LSTM") {
    lstm_operator->SetParserMode(mapper::PARSER_ONNX);
    if (prim->GetAttr(kDirection) != nullptr) {
      if (api::GetValue<string>(prim->GetAttr(kDirection)) == "forward") {
        lstm_operator->SetRecurrentDirection(mapper::RECURRENT_FORWARD);
        lstm_operator->SetNumOfDirection(kNums1);
      }
      if (api::GetValue<string>(prim->GetAttr(kDirection)) == "reverse") {
        lstm_operator->SetRecurrentDirection(mapper::RECURRENT_REVERSE);
        lstm_operator->SetNumOfDirection(kNums1);
      }
      if (api::GetValue<string>(prim->GetAttr(kDirection)) == "bidirectional") {
        lstm_operator->SetRecurrentDirection(mapper::RECURRENT_BIDIRECTIONAL);
        lstm_operator->SetNumOfDirection(kNums2);
      }
    }
    if (prim->GetAttr(kClip) != nullptr) {
      lstm_operator->SetAfClip(api::GetValue<float>(prim->GetAttr(kClip)));
    }
    if (prim->GetAttr(kHiddenSize) != nullptr) {
      lstm_operator->SetRecurrentNumOutput(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(kHiddenSize))));
    }
    if (SetRecurrentOnnxInfo(cnode, lstm_operator.get()) != RET_OK) {
      MS_LOG(ERROR) << "set lstm data info failed.";
      return RET_ERROR;
    }
    if (PushOfflineArgs(cnode, lstm_operator.get(), 1) != RET_OK) {
      MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }

  base_operators->push_back(std::move(lstm_operator));
  return RET_OK;
}
REG_MAPPER(Lstm, LstmMapper)
REG_MAPPER(LSTM, LstmMapper)
}  // namespace dpico
}  // namespace mindspore
