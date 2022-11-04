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

#include "mapper/prelu_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_enum.h"
#include "common/fetch_content.h"
#include "common/anf_util.h"
#include "ops/fusion/prelu_fusion.h"
#include "op/prelu_operator.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS SetPReluDataInfo(const api::CNodePtr &cnode, const api::PrimitivePtr &prim,
                        mapper::PreluOperator *prelu_operator) {
  if (cnode->inputs().size() > kInputIndex2) {
    auto input_anode = cnode->input(kInputIndex2);
    if (api::utils::isa<api::ParameterPtr>(input_anode)) {
      auto input_param_node = input_anode->cast<api::ParameterPtr>();
      if (input_param_node == nullptr) {
        MS_LOG(ERROR) << "input_param_node is nullptr.";
        return RET_ERROR;
      }
      auto tensor_info = input_param_node->default_param()->cast<api::TensorPtr>();
      if (tensor_info != nullptr && tensor_info->DataSize() != 0) {
        auto raw_datas = static_cast<float *>(tensor_info->data());
        auto elem_count = tensor_info->DataSize();
        prelu_operator->SetAlphaNegVec(std::vector<float>(raw_datas, raw_datas + elem_count));
      } else {
        MS_LOG(ERROR) << "tensor_info is nullptr, or DataSize equals zero. " << input_param_node->fullname_with_scope();
        return RET_ERROR;
      }
    }
  } else if (prim->GetAttr(ops::kSlope) != nullptr) {
    prelu_operator->SetAlphaNegVec(api::GetValue<std::vector<float>>(prim->GetAttr(ops::kSlope)));
  }
  return RET_OK;
}
}  // namespace
STATUS PReluMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                        const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto prelu_prim = api::utils::cast<api::SharedPtr<ops::PReLUFusion>>(prim);
  MS_ASSERT(prelu_prim != nullptr);

  auto prelu_operator = std::make_unique<mapper::PreluOperator>();
  if (prelu_operator == nullptr) {
    MS_LOG(ERROR) << "prelu_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, prelu_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (prim->GetAttr(ops::kChannelShared) != nullptr) {
    prelu_operator->SetPreluIsChannelShared(prelu_prim->get_channel_shared());
  }

  if (SetPReluDataInfo(cnode, prim, prelu_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set prelu data info failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  base_operators->push_back(std::move(prelu_operator));
  return RET_OK;
}
REG_MAPPER(PReLUFusion, PReluMapper)
}  // namespace dpico
}  // namespace mindspore
