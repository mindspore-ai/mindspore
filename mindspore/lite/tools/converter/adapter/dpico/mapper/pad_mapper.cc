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
#include "mapper/pad_mapper.h"
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include "common/op_enum.h"
#include "common/anf_util.h"
#include "op/pad_operator.h"
#include "ops/fusion/pad_fusion.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS SetPadDataInfo(const api::CNodePtr &cnode, mapper::PadOperator *pad_operator) {
  if (pad_operator == nullptr) {
    MS_LOG(ERROR) << "pad_operator is nullptr.";
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
    const int size = tensor_info->DataSize();
    std::vector<int32_t> pad_vec;
    if (tensor_info != nullptr && size != 0) {
      if (i == kInputIndex2) {
        auto data = reinterpret_cast<int *>(tensor_info->data());
        MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
        pad_vec = std::vector<int32_t>(data, data + size);
        pad_operator->SetPadsLocVec(pad_vec);
      } else if (i == kInputIndex3) {
        auto pad_value = reinterpret_cast<int64_t *>(tensor_info->data());
        pad_operator->SetPadValue(static_cast<float>(*pad_value));
      } else {
        MS_LOG(ERROR) << "pad operator only support 2 offline inputs at most, but " << cnode->fullname_with_scope()
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

STATUS PadMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                      const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto pad_prim = api::utils::cast<api::SharedPtr<ops::PadFusion>>(prim);
  MS_ASSERT(pad_prim != nullptr);

  auto pad_operator = std::make_unique<mapper::PadOperator>();
  if (SetCommonAttr(cnode, pad_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  pad_operator->SetOpType(mapper::OpType::PAD);
  if (pad_prim->GetAttr(ops::kPaddingMode) != nullptr) {
    pad_operator->SetPadMode(static_cast<mapper::PadMode>(pad_prim->get_padding_mode()));
  }

  if (SetPadDataInfo(cnode, pad_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set pad data info failed.";
    return RET_ERROR;
  }

  base_operators->push_back(std::move(pad_operator));
  return RET_OK;
}
REG_MAPPER(PadFusion, PadMapper)
}  // namespace dpico
}  // namespace mindspore
