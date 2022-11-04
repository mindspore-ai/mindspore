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
#include "mapper/pool_mapper.h"
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "common/op_enum.h"
#include "common/anf_util.h"
#include "op/pool_operator.h"
#include "ops/fusion/max_pool_fusion.h"
#include "ops/fusion/avg_pool_fusion.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS SetPadAttr(const api::PrimitivePtr &prim, mapper::PoolOperator *pool_operator) {
  if (pool_operator == nullptr) {
    MS_LOG(ERROR) << "pool_operator is nullptr. ";
    return RET_ERROR;
  }
  if (prim->GetAttr(ops::kPadMode) != nullptr) {
    auto pad_mode = PadMode(api::GetValue<int64_t>(prim->GetAttr(ops::kPadMode)));
    if (pad_mode == PadMode::PAD) {
      auto pad_list = api::GetValue<std::vector<int64_t>>(prim->GetAttr(ops::kPad));
      if (pad_list.size() != kDims4) {
        MS_LOG(ERROR) << "pad_list size is invalid. " << pad_list.size();
        return RET_ERROR;
      }
      pool_operator->SetPadUp(static_cast<int>(pad_list[0]));
      pool_operator->SetPadDown(static_cast<int>(pad_list[1]));
      pool_operator->SetPadLeft(static_cast<int>(pad_list[kAxis2]));
      pool_operator->SetPadRight(static_cast<int>(pad_list[kAxis3]));
      if (pad_list[0] == pad_list[1]) {
        pool_operator->SetPadHeight(static_cast<uint32_t>(pad_list[0]));
      }
      if (pad_list[kAxis2] == pad_list[kAxis3]) {
        pool_operator->SetPadWidth(static_cast<uint32_t>(pad_list[kAxis3]));
      }
    } else if (pad_mode == PadMode::SAME) {
      pool_operator->SetAutoPadType(mapper::AutoPadType::PAD_SAME_UPPER);
    } else if (pad_mode == PadMode::VALID) {
      pool_operator->SetAutoPadType(mapper::AutoPadType::PAD_VALID);
    } else {
      MS_LOG(ERROR) << "Non supported pad mode. " << pad_mode;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace
STATUS PoolMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                       const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto pool_operator = std::make_unique<mapper::PoolOperator>();
  if (pool_operator == nullptr) {
    MS_LOG(ERROR) << "pool_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, pool_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (CheckPrimitiveType(cnode, api::MakeShared<ops::AvgPoolFusion>())) {
    pool_operator->SetOpType(mapper::OpType::POOLINGAVE);
  } else if (CheckPrimitiveType(cnode, api::MakeShared<ops::MaxPoolFusion>())) {
    pool_operator->SetOpType(mapper::OpType::POOLINGMAX);
  } else {
    auto primitive = api::GetValueNode<api::PrimitivePtr>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "primitive is nullptr. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    MS_LOG(ERROR) << "pool method is not supported. " << primitive->name();
    return RET_ERROR;
  }

  if (prim->GetAttr(ops::kKernelSize)) {
    auto kernel_size = api::GetValue<std::vector<int64_t>>(prim->GetAttr(ops::kKernelSize));
    if (kernel_size.size() != kDims2) {
      MS_LOG(ERROR) << "kernel_size should be 2 dims, which is " << kernel_size.size();
      return RET_ERROR;
    }
    pool_operator->SetKernelHeight(static_cast<uint32_t>(kernel_size[0]));
    pool_operator->SetKernelWidth(static_cast<uint32_t>(kernel_size[1]));
  }

  if (prim->GetAttr(ops::kStrides) != nullptr) {
    auto stride = api::GetValue<std::vector<int64_t>>(prim->GetAttr(ops::kStrides));
    if (stride.size() != kDims2) {
      MS_LOG(ERROR) << "stride should be 2 dims, which is " << stride.size();
      return RET_ERROR;
    }
    pool_operator->SetStrideHeight(static_cast<uint32_t>(stride[0]));
    pool_operator->SetStrideWidth(static_cast<uint32_t>(stride[1]));
  }

  if (prim->GetAttr(ops::kGlobal) != nullptr) {
    auto global_flag = api::GetValue<bool>(prim->GetAttr(ops::kGlobal));
    pool_operator->SetGlobalPoolingFlag(global_flag);
  }

  if (prim->GetAttr(ops::kRoundMode) != nullptr) {
    auto round_mode = api::GetValue<int64_t>(prim->GetAttr(ops::kRoundMode));
    if (round_mode == static_cast<int64_t>(RoundMode::CEIL)) {
      pool_operator->SetRoundMode(mapper::POOLING_ROUND_MODE_CEIL);
    } else if (round_mode == static_cast<int64_t>(RoundMode::FLOOR)) {
      pool_operator->SetRoundMode(mapper::POOLING_ROUND_MODE_FLOOR);
    }
  }
  if (SetPadAttr(prim, pool_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set pad attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  base_operators->push_back(std::move(pool_operator));
  return RET_OK;
}
REG_MAPPER(AvgPoolFusion, PoolMapper)
REG_MAPPER(MaxPoolFusion, PoolMapper)
}  // namespace dpico
}  // namespace mindspore
