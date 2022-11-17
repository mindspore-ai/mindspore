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

#include "mapper/strided_slice_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "ops/strided_slice.h"
#include "op/extract_slice_operator.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS SetExtractSliceDataInfo(const api::CNodePtr &cnode, mapper::ExtractSliceOperator *extract_slice_operator) {
  if (extract_slice_operator == nullptr) {
    MS_LOG(ERROR) << "extract_slice_operator is nullptr.";
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
      auto data = reinterpret_cast<int32_t *>(tensor_info->data());
      if (i == kInputIndex2) {
        extract_slice_operator->SetStartsVec(std::vector<int32_t>(data, data + tensor_info->DataSize()));
      } else if (i == kInputIndex3) {
        extract_slice_operator->SetEndsVec(std::vector<int32_t>(data, data + tensor_info->DataSize()));
      } else if (i == kInputIndex4) {
        extract_slice_operator->SetAxesVec(std::vector<int32_t>(data, data + tensor_info->DataSize()));
      } else if (i == kInputIndex5) {
        extract_slice_operator->SetStepsVec(std::vector<int32_t>(data, data + tensor_info->DataSize()));
      } else {
        MS_LOG(ERROR) << "extract slice operator only support 4 offline inputs at most, but "
                      << cnode->fullname_with_scope() << " has " << i << " offline inputs.";
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
STATUS StridedSliceMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                               const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto extract_slice_operator = std::make_unique<mapper::ExtractSliceOperator>();
  if (extract_slice_operator == nullptr) {
    MS_LOG(ERROR) << "extract_slice_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, extract_slice_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  extract_slice_operator->SetOpType(mapper::OpType::EXTRACT_SLICE);

  if (SetExtractSliceDataInfo(cnode, extract_slice_operator.get())) {
    MS_LOG(ERROR) << "set strided slice data info failed." << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(extract_slice_operator));
  return RET_OK;
}
REG_MAPPER(StridedSlice, StridedSliceMapper)
}  // namespace dpico
}  // namespace mindspore
