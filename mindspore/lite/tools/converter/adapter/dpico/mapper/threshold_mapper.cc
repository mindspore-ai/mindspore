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

#include "mapper/threshold_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/threshold_operator.h"

namespace mindspore {
namespace dpico {
STATUS ThresholdMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                            const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto threshold_operator = std::make_unique<mapper::ThresholdOperator>();
  if (threshold_operator == nullptr) {
    MS_LOG(ERROR) << "threshold_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, threshold_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  threshold_operator->SetOpType(mapper::OpType::THRESHOLD);
  if (prim->GetAttr(dpico::kThreshold) != nullptr) {
    threshold_operator->SetThreshold(api::GetValue<float>(prim->GetAttr(dpico::kThreshold)));
  }

  base_operators->push_back(std::move(threshold_operator));
  return RET_OK;
}
REG_MAPPER(Threshold, ThresholdMapper)
}  // namespace dpico
}  // namespace mindspore
