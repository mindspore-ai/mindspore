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

#include "mapper/argmax_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "ops/fusion/arg_max_fusion.h"
#include "op/argmax_operator.h"

namespace mindspore {
namespace dpico {
STATUS ArgMaxMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto argmax_prim = api::utils::cast<api::SharedPtr<ops::ArgMaxFusion>>(prim);
  MS_ASSERT(argmax_prim != nullptr);

  auto argmax_operator = std::make_unique<mapper::ArgmaxOperator>();
  if (argmax_operator == nullptr) {
    MS_LOG(ERROR) << "argmax_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, argmax_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  argmax_operator->SetOpType(mapper::OpType::ARGMAX);
  if (prim->GetAttr(ops::kAxis) != nullptr) {
    argmax_operator->SetAxis(static_cast<int>(argmax_prim->get_axis()));
    argmax_operator->SetArgMaxHasAxis(true);
  }
  if (prim->GetAttr(ops::kTopK) != nullptr) {
    argmax_operator->SetArgMaxTopK(static_cast<uint32_t>(argmax_prim->get_top_k()));
  }
  if (prim->GetAttr(ops::kOutMaxValue) != nullptr) {
    argmax_operator->SetArgMaxOutMaxVal(argmax_prim->get_out_max_value());
  }
  if (prim->GetAttr(ops::kKeepDims) != nullptr) {
    argmax_operator->SetKeepDims(argmax_prim->get_keep_dims());
  }
  if (PushOfflineArgs(cnode, argmax_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(argmax_operator));
  return RET_OK;
}
REG_MAPPER(ArgMaxFusion, ArgMaxMapper)
}  // namespace dpico
}  // namespace mindspore
