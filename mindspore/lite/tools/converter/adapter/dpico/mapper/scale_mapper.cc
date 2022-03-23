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

#include "mapper/scale_mapper.h"
#include <memory>
#include <vector>
#include <utility>
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "common/anf_util.h"
#include "ops/scale.h"
#include "op/scale_operator.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS SetScaleDataInfo(const api::CNodePtr &cnode, mapper::ScaleOperator *scale_operator) {
  if (scale_operator == nullptr) {
    MS_LOG(ERROR) << "scale_operator is nullptr.";
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
        scale_operator->SetScaleWeightPtr(data);
        scale_operator->SetScaleWeightSize(tensor_info->DataSize());
        DataInfo data_info;
        if (FetchDataFromParameterNode(cnode, kInputIndex2, &data_info) != RET_OK) {
          MS_LOG(ERROR) << "fetch data from param node failed." << cnode->fullname_with_scope();
          return RET_ERROR;
        }
        scale_operator->SetScaleShapeVec(data_info.shape_);
      } else if (i == kInputIndex3) {
        scale_operator->SetScaleBiasPtr(data);
        scale_operator->SetScaleBiasSize(tensor_info->DataSize());
      } else {
        MS_LOG(ERROR) << "scale operator only support 2 offline inputs at most, but " << cnode->fullname_with_scope()
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
STATUS ScaleMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                        const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto scale_prim = api::utils::cast<api::SharedPtr<ops::Scale>>(prim);
  MS_ASSERT(scale_prim != nullptr);

  auto scale_operator = std::make_unique<mapper::ScaleOperator>();
  if (scale_operator == nullptr) {
    MS_LOG(ERROR) << "scale_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, scale_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  scale_operator->SetOpType(mapper::OpType::SCALE);
  scale_operator->SetAxis(static_cast<int32_t>(scale_prim->get_axis()));
  if (scale_prim->GetAttr(kBiasTerm) != nullptr) {
    scale_operator->SetScaleBiasFlag(api::GetValue<bool>(scale_prim->GetAttr(kBiasTerm)));
  }
  if (scale_prim->GetAttr(kNumAxes) != nullptr) {
    scale_operator->SetScaleNumAxes(static_cast<int32_t>(api::GetValue<int64_t>(scale_prim->GetAttr(kNumAxes))));
  }

  if (SetScaleDataInfo(cnode, scale_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set scale data info failed.";
    return RET_ERROR;
  }
  if (PushOfflineArgs(cnode, scale_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(scale_operator));
  return RET_OK;
}
REG_MAPPER(ScaleFusion, ScaleMapper)
}  // namespace dpico
}  // namespace mindspore
