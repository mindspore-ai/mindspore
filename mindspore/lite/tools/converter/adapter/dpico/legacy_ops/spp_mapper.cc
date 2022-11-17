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

#include "legacy_ops/spp_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/spp_operator.h"

namespace mindspore {
namespace dpico {
STATUS SppMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                      const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto spp_operator = std::make_unique<mapper::SppOperator>();
  if (spp_operator == nullptr) {
    MS_LOG(ERROR) << "spp_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, spp_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  spp_operator->SetOpType(mapper::OpType::SPPPOOLING);

  if (prim->GetAttr(dpico::kPyramidHeight) != nullptr) {
    spp_operator->SetPyramidHeight(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kPyramidHeight))));
  }
  if (prim->GetAttr(dpico::kPoolMethod) != nullptr) {
    auto pool_method = api::GetValue<int64_t>(prim->GetAttr(dpico::kPoolMethod));
    if (pool_method == 0) {
      spp_operator->SetSppType(mapper::OpType::POOLINGMAX);
    } else if (pool_method == 1) {
      spp_operator->SetSppType(mapper::OpType::POOLINGAVE);
    } else {
      MS_LOG(ERROR) << "only supports max && ave pool by dpico. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }

  base_operators->push_back(std::move(spp_operator));
  return RET_OK;
}
REG_MAPPER(Spp, SppMapper)
}  // namespace dpico
}  // namespace mindspore
