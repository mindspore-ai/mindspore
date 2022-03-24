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

#include "mapper/arithmetic_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <string>
#include "ops/neg.h"
#include "common/anf_util.h"
#include "op/binary_math_operator.h"

namespace mindspore {
namespace dpico {
namespace {
const size_t kOfflineArgSize2 = 2;
const std::unordered_map<std::string, mapper::BinaryMathOp> kArithmeticOpMap = {
  {"AddFusion", mapper::BinaryMathOp::ADD_OP},
  {"BiasAdd", mapper::BinaryMathOp::ADD_OP},
  {"SubFusion", mapper::BinaryMathOp::SUB_OP},
  {"DivFusion", mapper::BinaryMathOp::DIV_OP},
  {"MulFusion", mapper::BinaryMathOp::MUL_OP},
  {"Neg", mapper::BinaryMathOp::MUL_OP},
  {"Maximum", mapper::BinaryMathOp::MAX_OP},
  {"Minimum", mapper::BinaryMathOp::MIN_OP},
  {"SquaredDifference", mapper::BinaryMathOp::SQUARE_DIFF_OP},
  {"X_DIV_Y", mapper::BinaryMathOp::X_DIV_Y_OP},
  {"X_LOG_Y", mapper::BinaryMathOp::X_LOG_Y_OP}};
}  // namespace
STATUS ArithmeticMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                             const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto arithmetic_operator = std::make_unique<mapper::BinaryMathOperator>();
  if (arithmetic_operator == nullptr) {
    MS_LOG(ERROR) << "arithmetic_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, arithmetic_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  std::string op_type_name;
  if (GetPrimitiveType(cnode, &op_type_name) != RET_OK) {
    MS_LOG(ERROR) << "get cnode primitive type failed:" << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (kArithmeticOpMap.find(op_type_name) == kArithmeticOpMap.end()) {
    MS_LOG(ERROR) << op_type_name << " there isn't corresponding mapper of " << op_type_name << " for now.";
    return RET_ERROR;
  }
  auto binary_math_op = kArithmeticOpMap.at(op_type_name);
  arithmetic_operator->SetBinaryMathOp(binary_math_op);

  if (CheckPrimitiveType(cnode, api::MakeShared<ops::Neg>())) {
    arithmetic_operator->PushOfflineArgs(std::make_pair(std::vector<float>{-1.0}, std::vector<int32_t>{}));
    arithmetic_operator->PushOfflineArgs(std::make_pair(std::vector<float>{}, std::vector<int32_t>{}));
  } else {
    if (PushOfflineArgs(cnode, arithmetic_operator.get(), kOfflineArgSize2) != RET_OK) {
      MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }

  base_operators->push_back(std::move(arithmetic_operator));
  return RET_OK;
}
REG_MAPPER(AddFusion, ArithmeticMapper)
REG_MAPPER(BiasAdd, ArithmeticMapper)
REG_MAPPER(SubFusion, ArithmeticMapper)
REG_MAPPER(MulFusion, ArithmeticMapper)
REG_MAPPER(DivFusion, ArithmeticMapper)
REG_MAPPER(Maximum, ArithmeticMapper)
REG_MAPPER(Minimum, ArithmeticMapper)
REG_MAPPER(SquaredDifference, ArithmeticMapper)
REG_MAPPER(X_DIV_Y, ArithmeticMapper)
REG_MAPPER(X_LOG_Y, ArithmeticMapper)
REG_MAPPER(Neg, ArithmeticMapper)
}  // namespace dpico
}  // namespace mindspore
