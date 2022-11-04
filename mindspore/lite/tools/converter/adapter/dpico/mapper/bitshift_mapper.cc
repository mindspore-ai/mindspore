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

#include "mapper/bitshift_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "op/bit_shift_operator.h"
#include "parser/onnx/onnx_bitshift_parser.h"

namespace mindspore {
namespace dpico {
STATUS BitShiftMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                           const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto bitshift_operator = std::make_unique<mapper::BitShiftOperator>();
  if (SetCommonAttr(cnode, bitshift_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  bitshift_operator->SetOpType(mapper::OpType::BIT_SHIFT);
  if (prim->GetAttr("direction") != nullptr) {
    if (api::GetValue<string>(prim->GetAttr("direction")) == "LEFT") {
      bitshift_operator->SetBitShiftDirection(mapper::BIT_SHIFT_LEFT);
    } else if (api::GetValue<string>(prim->GetAttr("direction")) == "RIGHT") {
      bitshift_operator->SetBitShiftDirection(mapper::BIT_SHIFT_RIGHT);
    } else {
      MS_LOG(ERROR) << "set bitshift direction failed";
      return RET_ERROR;
    }
  }

  base_operators->push_back(std::move(bitshift_operator));
  return RET_OK;
}
REG_MAPPER(BitShift, BitShiftMapper)
}  // namespace dpico
}  // namespace mindspore
