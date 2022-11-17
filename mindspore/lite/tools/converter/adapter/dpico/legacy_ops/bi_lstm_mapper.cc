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

#include "legacy_ops/bi_lstm_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/bi_lstm_operator.h"

namespace mindspore {
namespace dpico {
STATUS BiLstmMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto bi_lstm_operator = std::make_unique<mapper::BiLstmOperator>();
  if (bi_lstm_operator == nullptr) {
    MS_LOG(ERROR) << "bi_lstm_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, bi_lstm_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (prim->GetAttr(kNumOutput) != nullptr) {
    bi_lstm_operator->SetRecurrentNumOutput(static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kNumOutput))));
  }
  if (prim->GetAttr(kExposeHidden) != nullptr) {
    bi_lstm_operator->SetRecurrentExposeHidden(api::GetValue<bool>(prim->GetAttr(kExposeHidden)));
  }
  if (prim->GetAttr(kOutputChannel) != nullptr) {
    bi_lstm_operator->SetOutputChannel(static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kOutputChannel))));
  }
  bi_lstm_operator->SetRecurrentContFlag(true);
  if (SetRecurrentDataInfo(cnode, bi_lstm_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set bi_lstm data info failed.";
    return RET_ERROR;
  }

  base_operators->push_back(std::move(bi_lstm_operator));
  return RET_OK;
}
REG_MAPPER(BiLstm, BiLstmMapper)
}  // namespace dpico
}  // namespace mindspore
