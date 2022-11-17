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

#include "mapper/batch_norm_mapper.h"
#include <memory>
#include <utility>
#include <limits>
#include <vector>
#include "ops/batch_norm.h"
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "op/batch_norm_operator.h"

namespace mindspore {
namespace dpico {
namespace {
// BatchNorm: {BNMeanIndex:2, BNVarIndex:3, ScaleFactorIndex:4}
STATUS SetBnDataInfo(const api::CNodePtr &cnode, mapper::BatchNormOperator *batch_norm_operator) {
  if (batch_norm_operator == nullptr) {
    MS_LOG(ERROR) << "batch_norm_operator is nullptr.";
    return RET_ERROR;
  }
  for (size_t i = 2; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    MS_ASSERT(input_node != nullptr);
    auto param_node = input_node->cast<api::ParameterPtr>();
    if (param_node == nullptr || !param_node->has_default()) {
      continue;
    }
    auto tensor_info = param_node->default_param()->cast<api::TensorPtr>();
    if (tensor_info != nullptr && tensor_info->DataSize() != 0) {
      auto data = reinterpret_cast<float *>(tensor_info->data());
      MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
      if (i == kInputIndex2) {
        batch_norm_operator->SetBnMeanDataPtr(data);
        batch_norm_operator->SetBnMeanSize(tensor_info->DataSize());
      } else if (i == kInputIndex3) {
        batch_norm_operator->SetBnVarDataPtr(data);
        batch_norm_operator->SetBnVarSize(tensor_info->DataSize());
      } else if (i == kInputIndex4) {
        batch_norm_operator->SetBnScalePtr(data);
        batch_norm_operator->SetBnScaleSize(tensor_info->DataSize());
      } else {
        MS_LOG(ERROR) << "batch_norm only support 3 offline inputs at most, but " << cnode->fullname_with_scope()
                      << " has " << i << " offline inputs.";
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << "param node's tensor info is invalid. " << input_node->fullname_with_scope();
      return RET_ERROR;
    }
  }

  return RET_OK;
}
// FusedBatchNorm: {ScaleIndex:2, BiasIndex:3, MeanIndex:4, VarIndex:5}
STATUS SetFusedBnDataInfo(const api::CNodePtr &cnode, mapper::BatchNormOperator *batch_norm_operator) {
  if (batch_norm_operator == nullptr) {
    MS_LOG(ERROR) << "batch_norm_operator is nullptr.";
    return RET_ERROR;
  }
  for (size_t i = 2; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    MS_ASSERT(input_node != nullptr);
    auto param_node = input_node->cast<api::ParameterPtr>();
    if (param_node == nullptr || !param_node->has_default()) {
      continue;
    }
    auto tensor_info = param_node->default_param()->cast<api::TensorPtr>();
    if (tensor_info != nullptr && tensor_info->DataSize() != 0) {
      auto data = reinterpret_cast<float *>(tensor_info->data());
      MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
      if (i == kInputIndex2) {
        batch_norm_operator->SetBnScalePtr(data);
        batch_norm_operator->SetBnScaleSize(tensor_info->DataSize());
      } else if (i == kInputIndex3) {
        batch_norm_operator->SetBnBiasPtr(data);
        batch_norm_operator->SetBnBiasSize(tensor_info->DataSize());
      } else if (i == kInputIndex4) {
        batch_norm_operator->SetBnMeanDataPtr(data);
        batch_norm_operator->SetBnMeanSize(tensor_info->DataSize());
      } else if (i == kInputIndex5) {
        batch_norm_operator->SetBnVarDataPtr(data);
        batch_norm_operator->SetBnVarSize(tensor_info->DataSize());
      } else {
        MS_LOG(ERROR) << "fused batch_norm only support 4 offline inputs at most, but " << cnode->fullname_with_scope()
                      << " has " << i << " offline inputs.";
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << "param node's tensor info is invalid. " << input_node->fullname_with_scope();
      return RET_ERROR;
    }
  }

  return RET_OK;
}
}  // namespace
STATUS BatchNormMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                            const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto batch_norm_operator = std::make_unique<mapper::BatchNormOperator>();
  if (batch_norm_operator == nullptr) {
    MS_LOG(ERROR) << "batch_norm_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, batch_norm_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  batch_norm_operator->SetOpType(mapper::OpType::BN);
  if (prim->GetAttr(ops::kEpsilon) != nullptr) {
    batch_norm_operator->SetBnEps(api::GetValue<float>(prim->GetAttr(ops::kEpsilon)));
  }

  if (prim->GetAttr(ops::kMomentum) != nullptr) {
    auto momentum = api::GetValue<float>(prim->GetAttr(ops::kMomentum));
    const float default_momentum_value = 0.9;
    if (std::fabs(momentum - default_momentum_value) > std::numeric_limits<float>::epsilon()) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << "'s momentum attr value " << momentum
                   << " is not equal to mapper default value 0.9. Note that mapper will ignore this value.";
    }
  }

  if (CheckPrimitiveType(cnode, api::MakeShared<ops::BatchNorm>())) {
    if (SetBnDataInfo(cnode, batch_norm_operator.get()) != RET_OK) {
      MS_LOG(ERROR) << "set bn data info failed.";
      return RET_ERROR;
    }
  } else {
    if (SetFusedBnDataInfo(cnode, batch_norm_operator.get()) != RET_OK) {
      MS_LOG(ERROR) << "set fused bn data info failed.";
      return RET_ERROR;
    }
  }
  if (PushOfflineArgs(cnode, batch_norm_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(batch_norm_operator));
  return RET_OK;
}
REG_MAPPER(BatchNorm, BatchNormMapper)
REG_MAPPER(FusedBatchNorm, BatchNormMapper)
}  // namespace dpico
}  // namespace mindspore
