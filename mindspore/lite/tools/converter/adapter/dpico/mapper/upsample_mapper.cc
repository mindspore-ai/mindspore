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

#include "mapper/upsample_mapper.h"
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include "common/op_attr.h"
#include "op/upsample_operator.h"

namespace mindspore {
namespace dpico {
STATUS UpsampleMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                           const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto upsample_operator = std::make_unique<mapper::UpsampleOperator>();
  if (upsample_operator == nullptr) {
    MS_LOG(ERROR) << "upsample_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, upsample_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  upsample_operator->SetOpType(mapper::OpType::UPSAMPLE);
  if (prim->GetAttr(ops::kScale) != nullptr) {
    upsample_operator->SetUpsampleScale(api::GetValue<float>(prim->GetAttr(ops::kScale)));
  }
  if (prim->GetAttr(kInterpolationMode) != nullptr) {
    auto mode = api::GetValue<std::string>(prim->GetAttr(kInterpolationMode));
    if (mode == kNearest) {
      upsample_operator->SetInterpolationMode(mapper::InterpolationMode::NEAREST);
    } else if (mode == kBilinear) {
      upsample_operator->SetInterpolationMode(mapper::InterpolationMode::BILINEAR);
    } else {
      MS_LOG(ERROR) << "current interpolation mode is not supported. " << mode;
      return RET_ERROR;
    }
  }

  base_operators->push_back(std::move(upsample_operator));
  return RET_OK;
}
REG_MAPPER(Upsample, UpsampleMapper)
}  // namespace dpico
}  // namespace mindspore
