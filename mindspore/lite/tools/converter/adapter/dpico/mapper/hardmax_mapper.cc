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

#include "mapper/hardmax_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "op/hardmax_operator.h"
#include "parser/onnx/onnx_hardmax_parser.h"

namespace mindspore {
namespace dpico {
STATUS HardmaxMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto custom_prim = api::utils::cast<api::SharedPtr<ops::Custom>>(prim);
  MS_CHECK_TRUE_MSG(custom_prim != nullptr, RET_ERROR, "custom_prim is nullptr");
  auto hard_max_operator = std::make_unique<mapper::HardmaxOperator>();
  if (SetCommonAttr(cnode, hard_max_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  hard_max_operator->SetOpType(mapper::OpType::HARDMAX);
  DataInfo data_info;
  if (cnode->inputs().size() > kInputIndex3 &&
      FetchDataFromParameterNode(cnode, kInputIndex3, &data_info) == lite::RET_OK) {
    if (data_info.data_type_ != static_cast<int>(kNumberTypeInt32)) {
      MS_LOG(ERROR) << "data_type not correct";
      return RET_ERROR;
    }
    auto data = reinterpret_cast<int32_t *>(data_info.data_.data());
    hard_max_operator->SetHardmaxAxis(*data);
  } else if (custom_prim->GetAttr(ops::kAxis) != nullptr) {
    hard_max_operator->SetHardmaxAxis(static_cast<int32_t>(api::GetValue<int64_t>(custom_prim->GetAttr(ops::kAxis))));
  } else {
    MS_LOG(ERROR) << "null param";
    return RET_ERROR;
  }

  if (PushOfflineArgs(cnode, hard_max_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed.";
    return RET_ERROR;
  }
  base_operators->push_back(std::move(hard_max_operator));
  return RET_OK;
}
REG_MAPPER(Hardmax, HardmaxMapper)
}  // namespace dpico
}  // namespace mindspore
