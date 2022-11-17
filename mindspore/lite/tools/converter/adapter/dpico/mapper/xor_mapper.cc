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

#include "mapper/xor_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "op/logical_operator.h"
#include "parser/onnx/onnx_xor_parser.h"

namespace mindspore {
namespace dpico {
const size_t kOfflineArgSize2 = 2;

STATUS XorMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                      const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto xor_operator = std::make_unique<mapper::LogicalOperator>();
  if (xor_operator == nullptr) {
    MS_LOG(ERROR) << "xor_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, xor_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  xor_operator->SetOpType(mapper::OpType::LOGICAL);
  xor_operator->SetLogicalOperator(mapper::LOGICAL_XOR);

  if (PushOfflineArgs(cnode, xor_operator.get(), kOfflineArgSize2) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(xor_operator));
  return RET_OK;
}
REG_MAPPER(Xor, XorMapper)
}  // namespace dpico
}  // namespace mindspore
