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

#include "mapper/gather_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include "common/fetch_content.h"
#include "common/op_enum.h"
#include "ops/gather.h"
#include "op/gather_operator.h"

namespace mindspore {
namespace dpico {
namespace {
const size_t kOfflineArgSize2 = 2;
}  // namespace
STATUS GatherMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto gather_prim = api::utils::cast<api::SharedPtr<ops::Gather>>(prim);
  MS_ASSERT(gather_prim != nullptr);

  auto gather_operator = std::make_unique<mapper::GatherOperator>();
  if (gather_operator == nullptr) {
    MS_LOG(ERROR) << "gather_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, gather_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  gather_operator->SetOpType(mapper::OpType::GATHER);
  DataInfo data_info;
  if (cnode->inputs().size() > kInputIndex3 &&
      FetchDataFromParameterNode(cnode, kInputIndex3, &data_info) == lite::RET_OK) {
    if (data_info.data_type_ != static_cast<int>(kNumberTypeInt32)) {
      MS_LOG(ERROR) << "data_type not correct";
      return RET_ERROR;
    }
    auto data = reinterpret_cast<int32_t *>(data_info.data_.data());
    gather_operator->SetAxis(*data);
  } else if (gather_prim->GetAttr(ops::kAxis) != nullptr) {
    gather_operator->SetAxis(static_cast<int32_t>(api::GetValue<int64_t>(gather_prim->GetAttr(ops::kAxis))));
  } else {
    MS_LOG(ERROR) << "null param";
    return RET_ERROR;
  }

  if (PushOfflineArgs(cnode, gather_operator.get(), kOfflineArgSize2) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed.";
    return RET_ERROR;
  }

  base_operators->push_back(std::move(gather_operator));
  return RET_OK;
}
REG_MAPPER(Gather, GatherMapper)
}  // namespace dpico
}  // namespace mindspore
