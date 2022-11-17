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

#include "mapper/fc_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_attr.h"
#include "common/anf_util.h"
#include "op/fc_operator.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS SetNumOutput(const api::CNodePtr &cnode, const api::PrimitivePtr &prim, mapper::FcOperator *fc_operator) {
  if (fc_operator == nullptr) {
    MS_LOG(ERROR) << "fc_operator is nullptr.";
    return RET_ERROR;
  }
  std::vector<ShapeVector> output_shapes;
  if (GetOutputShapesFromCNode(cnode, &output_shapes) != RET_OK) {
    MS_LOG(ERROR) << "get node shape failed";
    return RET_ERROR;
  }
  if (output_shapes.size() != 1) {
    MS_LOG(ERROR) << "fc should have single output, but in fact it has " << output_shapes.size() << " outputs.";
    return RET_ERROR;
  }
  auto output_shape = output_shapes.at(0);
  if (prim->GetAttr(kNumOutput) != nullptr) {
    uint32_t num_output = static_cast<size_t>(api::GetValue<int64_t>(prim->GetAttr(kNumOutput)));
    if (output_shape.back() != num_output) {
      MS_LOG(ERROR) << "num output attr isn't matched with fc output shape.";
      return RET_ERROR;
    }
    fc_operator->SetNumOutput(num_output);
  } else {
    fc_operator->SetNumOutput(static_cast<uint32_t>(output_shape.back()));
  }
  return RET_OK;
}
}  // namespace
STATUS FCMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                     const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto fc_operator = std::make_unique<mapper::FcOperator>();
  if (fc_operator == nullptr) {
    MS_LOG(ERROR) << "fc_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, fc_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  fc_operator->SetOpType(mapper::OpType::INNERPRODUCT);
  if (prim->GetAttr(ops::kAxis) != nullptr) {
    fc_operator->SetAxis(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(ops::kAxis))));
  }
  if (prim->GetAttr(ops::kTransposeB) != nullptr) {
    //  note that this value of fc operator is opposite to kTransposeB
    fc_operator->SetFcTransposeFlag(!api::GetValue<bool>(prim->GetAttr(ops::kTransposeB)));
  }
  if (SetNumOutput(cnode, prim, fc_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set num output failed.";
    return RET_ERROR;
  }
  if (SetConvFcDataInfo(cnode, fc_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set fc data info failed.";
    return RET_ERROR;
  }

  base_operators->push_back(std::move(fc_operator));
  return RET_OK;
}
REG_MAPPER(FullConnection, FCMapper)
}  // namespace dpico
}  // namespace mindspore
