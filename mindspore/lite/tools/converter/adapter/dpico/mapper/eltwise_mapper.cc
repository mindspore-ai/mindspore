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

#include "mapper/eltwise_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/eltwise_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace dpico {
STATUS EltwiseMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto eltwise_operator = std::make_unique<mapper::EltwiseOperator>();
  if (eltwise_operator == nullptr) {
    MS_LOG(ERROR) << "eltwise_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, eltwise_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  eltwise_operator->SetOpType(mapper::OpType::ELTWISE);
  if (prim->GetAttr(ops::kMode) == nullptr) {
    MS_LOG(ERROR) << "eltwise op should have mode attr. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto eltwise_mode = static_cast<EltwiseMode>(api::GetValue<int64_t>(prim->GetAttr(ops::kMode)));
  switch (eltwise_mode) {
    case mindspore::EltwiseMode::PROD:
      eltwise_operator->SetEltwiseOp(mapper::BinaryMathOp::MUL_OP);
      break;
    case mindspore::EltwiseMode::SUM:
      eltwise_operator->SetEltwiseOp(mapper::BinaryMathOp::SUM_OP);
      break;
    case mindspore::EltwiseMode::MAXIMUM:
      eltwise_operator->SetEltwiseOp(mapper::BinaryMathOp::MAX_OP);
      break;
    default:
      MS_LOG(ERROR) << "Eltwise parse params fail, unsupported operation: " << eltwise_mode;
      return RET_ERROR;
  }

  if (prim->GetAttr(dpico::kCoeffs) != nullptr) {
    eltwise_operator->SetEltCoeffVec(api::GetValue<std::vector<float>>(prim->GetAttr(dpico::kCoeffs)));
  }

  base_operators->push_back(std::move(eltwise_operator));
  return RET_OK;
}
REG_MAPPER(Eltwise, EltwiseMapper)
}  // namespace dpico
}  // namespace mindspore
