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

#include "mapper/reshape_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "common/fetch_content.h"
#include "ops/reshape.h"
#include "op/reshape_operator.h"

namespace mindspore {
namespace dpico {
STATUS ReshapeMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto reshape_prim = api::utils::cast<api::SharedPtr<ops::Reshape>>(prim);
  MS_ASSERT(reshape_prim != nullptr);

  auto reshape_operator = std::make_unique<mapper::ReshapeOperator>();
  if (reshape_operator == nullptr) {
    MS_LOG(ERROR) << "reshape_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, reshape_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  reshape_operator->SetOpType(mapper::OpType::RESHAPE);

  DataInfo data_info;
  if (cnode->inputs().size() > kInputIndex2 &&
      FetchDataFromParameterNode(cnode, kInputIndex2, &data_info) == lite::RET_OK) {
    if (data_info.data_type_ != static_cast<int>(kNumberTypeInt32)) {
      MS_LOG(ERROR) << "data_type not correct";
      return RET_ERROR;
    }
    auto data = reinterpret_cast<int32_t *>(data_info.data_.data());
    if (data == nullptr) {
      MS_LOG(ERROR) << "data is nullptr. " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    std::vector<int> dims;
    int data_size;
    if (GetDataSizeFromTensor(&data_info, &data_size) != RET_OK) {
      MS_LOG(ERROR) << "get data size from tensor failed.";
      return RET_ERROR;
    }
    (void)std::transform(data, data + data_size, std::back_inserter(dims),
                         [](const int32_t &value) { return static_cast<int32_t>(value); });
    reshape_operator->SetReshapeDimVec(dims);
  } else if (reshape_prim->GetAttr(ops::kShape) != nullptr) {
    auto shape = api::GetValue<std::vector<int64_t>>(reshape_prim->GetAttr(ops::kShape));
    std::vector<int> dims;
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(dims),
                         [](const int64_t &value) { return static_cast<int32_t>(value); });
    reshape_operator->SetReshapeDimVec(dims);
  } else {
    MS_LOG(ERROR) << "shape attr doesn't exist." << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (prim->GetAttr(ops::kAxis) != nullptr) {
    auto axis = static_cast<int32_t>(api::GetValue<int64_t>(reshape_prim->GetAttr(ops::kAxis)));
    reshape_operator->SetAxis(axis);
  } else {
    reshape_operator->SetAxis(0);
  }
  if (prim->GetAttr(kNumAxes) != nullptr) {
    auto num_axes = static_cast<int32_t>(api::GetValue<int64_t>(reshape_prim->GetAttr(kNumAxes)));
    reshape_operator->SetReshapeNumAxes(num_axes);
  }
  if (PushOfflineArgs(cnode, reshape_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(reshape_operator));
  return RET_OK;
}
REG_MAPPER(Reshape, ReshapeMapper)
}  // namespace dpico
}  // namespace mindspore
