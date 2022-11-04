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

#include "mapper/permute_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include "common/op_enum.h"
#include "common/data_transpose_utils.h"
#include "common/fetch_content.h"
#include "common/op_attr.h"
#include "ops/transpose.h"
#include "op/permute_operator.h"

namespace mindspore {
namespace dpico {
STATUS PermuteMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto permute_prim = api::utils::cast<api::SharedPtr<ops::Transpose>>(prim);
  MS_ASSERT(permute_prim != nullptr);

  auto permute_operator = std::make_unique<mapper::PermuteOperator>();
  if (permute_operator == nullptr) {
    MS_LOG(ERROR) << "permute_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, permute_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  permute_operator->SetOpType(mapper::OpType::PERMUTE);
  std::vector<int32_t> perm_val;
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
    int data_size;
    if (GetDataSizeFromTensor(&data_info, &data_size) != RET_OK) {
      MS_LOG(ERROR) << "get data size from tensor failed.";
      return RET_ERROR;
    }
    if (static_cast<size_t>(data_size) != kDims4) {
      MS_LOG(ERROR) << "perm val size should be " << kDims4;
      return RET_ERROR;
    }
    perm_val = {data[0], data[1], data[kAxis2], data[kAxis3]};
  } else if (permute_prim->GetAttr(kPerm) != nullptr) {
    auto perm_vec = api::GetValue<std::vector<int64_t>>(permute_prim->GetAttr(kPerm));
    (void)std::transform(perm_vec.begin(), perm_vec.end(), std::back_inserter(perm_val),
                         [](int64_t p) { return static_cast<int32_t>(p); });
  } else {
    MS_LOG(ERROR) << "can't get perm value. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (perm_val == kNH2NC) {
    permute_operator->SetOutputDimOrderFormat(mapper::DimOrderFormat::NCHW_FORMAT);
  } else if (perm_val == kNC2NH) {
    permute_operator->SetOutputDimOrderFormat(mapper::DimOrderFormat::NHWC_FORMAT);
  } else {
    MS_LOG(ERROR) << "invalid perm vec. " << perm_val[0] << " " << perm_val[1] << " " << perm_val[kAxis2] << " "
                  << perm_val[kAxis3];
    return RET_ERROR;
  }

  base_operators->push_back(std::move(permute_operator));
  return RET_OK;
}
REG_MAPPER(Transpose, PermuteMapper)
}  // namespace dpico
}  // namespace mindspore
