/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "mapper/mod_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "op/mod_operator.h"
#include "parser/onnx/onnx_mod_parser.h"

namespace mindspore {
namespace dpico {
STATUS ModMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                      const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto mod_operator = std::make_unique<mapper::ModOperator>();
  if (SetCommonAttr(cnode, mod_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  mod_operator->SetOpType(mapper::OpType::MOD_TYPE);
  mod_operator->SetIsFmod(false);
  if (prim->GetAttr(kFmod) != nullptr) {
    if (api::GetValue<int64_t>(prim->GetAttr(kFmod)) == 1) {
      mod_operator->SetIsFmod(true);
    }
  }

  base_operators->push_back(std::move(mod_operator));
  return RET_OK;
}
REG_MAPPER(Mod, ModMapper)
}  // namespace dpico
}  // namespace mindspore
