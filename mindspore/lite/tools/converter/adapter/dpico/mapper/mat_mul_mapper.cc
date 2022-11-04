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

#include "mapper/mat_mul_mapper.h"
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/matrix_operator.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS DoMaxtixOperatorMap(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                           const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  auto matrix_operator = std::make_unique<mapper::MatrixOperator>();
  if (matrix_operator == nullptr) {
    MS_LOG(ERROR) << "matrix_operator is nullptr.";
    return RET_ERROR;
  }
  if (SetCommonAttr(cnode, matrix_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (prim->GetAttr(kDim1) != nullptr) {
    matrix_operator->SetMatMulDim1(static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kDim1))));
  } else {
    MS_LOG(ERROR) << kDim1 << " attr is missed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (prim->GetAttr(kDim2) != nullptr) {
    matrix_operator->SetMatMulDim2(static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kDim2))));
  } else {
    MS_LOG(ERROR) << kDim2 << " attr is missed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (prim->GetAttr(kDim3) != nullptr) {
    matrix_operator->SetMatMulDim3(static_cast<uint32_t>(api::GetValue<int64_t>(prim->GetAttr(kDim3))));
  } else {
    MS_LOG(ERROR) << kDim3 << " attr is missed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(matrix_operator));
  return RET_OK;
}
}  // namespace
STATUS MatMulMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  if (prim->GetAttr(kOperatorType) != nullptr) {
    auto op_type = api::GetValue<std::string>(prim->GetAttr(kOperatorType));
    if (op_type == "FullConnection") {
      if (OpMapperRegistry::GetInstance()->GetOpMapper(op_type) == nullptr) {
        MS_LOG(ERROR) << "FullConnection get op mapper failed. ";
        return RET_ERROR;
      }
      int ret = OpMapperRegistry::GetInstance()->GetOpMapper(op_type)->Map(cnode, base_operators, prim, output_cnodes);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "FullConnection mapper failed. ";
        return RET_ERROR;
      }
    } else if (op_type == "Matrix") {
      if (DoMaxtixOperatorMap(cnode, base_operators, prim, output_cnodes) != RET_OK) {
        MS_LOG(ERROR) << "map to matrix operator failed. " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << op_type << " of MatMul op is invalid. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << kOperatorType << " attr is missed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  return RET_OK;
}
REG_MAPPER(MatMulFusion, MatMulMapper)
}  // namespace dpico
}  // namespace mindspore
