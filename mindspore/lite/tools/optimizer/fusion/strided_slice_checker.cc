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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/strided_slice_checker.h"
#include <functional>
#include <vector>
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/op_name.h"

namespace mindspore {
namespace opt {
bool StridedSliceChecker::CheckCommonInfo(const CNodePtr &strided_slice) {
  if (strided_slice == nullptr) {
    return false;
  }
  auto prim = GetCNodePrimitive(strided_slice);
  MS_CHECK_TRUE_RET(prim != nullptr, false);
  if (IsQuantParameterNode(prim)) {
    return false;
  }
  auto ellipsis_mask =
    prim->GetAttr(ops::kEllipsisMask) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kEllipsisMask)) : 0;
  auto new_axis_mask =
    prim->GetAttr(ops::kNewAxisMask) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kNewAxisMask)) : 0;
  auto shrink_axis_mask =
    prim->GetAttr(ops::kShrinkAxisMask) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kShrinkAxisMask)) : 0;
  if (ellipsis_mask != 0 || new_axis_mask != 0 || shrink_axis_mask != 0) {
    return false;
  }

  if (!CheckStepIsOne(strided_slice)) {
    return false;
  }
  return true;
}

int StridedSliceChecker::GetBegin(const CNodePtr &strided_slice, std::vector<int> *begin) {
  if (strided_slice == nullptr || begin == nullptr) {
    MS_LOG(ERROR) << "exist in-parameter is a nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto prim = GetCNodePrimitive(strided_slice);
  MS_CHECK_TRUE_MSG(prim != nullptr, false, "Strided_slice's prim is a nullptr.");
  auto begin_mask = prim->GetAttr(ops::kBeginMask) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kBeginMask)) : 0;
  lite::DataInfo data;
  auto ret = GetConstTensor(strided_slice, ops::kInputIndex2, &data);
  if (ret == lite::RET_NOT_SUPPORT) {
    return ret;
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Get Strided_slice's begin failed, node name is " << strided_slice->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto num = std::accumulate(data.shape_.begin(), data.shape_.end(), 1, std::multiplies<>());
  for (int i = 0; i < num; ++i) {
    bool begin_ineffective = (begin_mask & (1 << i));
    int cur_begin = begin_ineffective ? 0 : static_cast<int *>(data.data_ptr_)[i];
    begin->push_back(cur_begin);
  }
  return lite::RET_OK;
}

int StridedSliceChecker::GetEnd(const CNodePtr &strided_slice, std::vector<int> *end) {
  if (strided_slice == nullptr || end == nullptr) {
    MS_LOG(ERROR) << "exist in-parameter is a nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto prim = GetCNodePrimitive(strided_slice);
  MS_CHECK_TRUE_MSG(prim != nullptr, false, "Strided_slice's prim is a nullptr.");
  auto end_mask = prim->GetAttr(ops::kEndMask) != nullptr ? GetValue<int64_t>(prim->GetAttr(ops::kEndMask)) : 0;
  lite::DataInfo data;
  auto ret = GetConstTensor(strided_slice, ops::kInputIndex3, &data);
  if (ret == lite::RET_NOT_SUPPORT) {
    return ret;
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Get Strided_slice's end failed, node name is " << strided_slice->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto num = std::accumulate(data.shape_.begin(), data.shape_.end(), 1, std::multiplies<>());
  for (int i = 0; i < num; ++i) {
    bool end_ineffective = (end_mask & (1 << i));
    int cur_end = end_ineffective ? INT_MAX : static_cast<int *>(data.data_ptr_)[i];
    end->push_back(cur_end);
  }
  return lite::RET_OK;
}

bool StridedSliceChecker::CheckStepIsOne(const CNodePtr &strided_slice) {
  if (strided_slice == nullptr) {
    return false;
  }
  if (strided_slice->size() < kInputSizeFive) {
    return true;
  }
  lite::DataInfo data;
  auto status = GetConstTensor(strided_slice, ops::kInputIndex4, &data);
  if (status != lite::RET_OK) {
    return false;
  }
  auto num = std::accumulate(data.shape_.begin(), data.shape_.end(), 1, std::multiplies<>());
  std::vector<int> temp(num, 1);
  return memcmp(data.data_ptr_, temp.data(), temp.size() * sizeof(int)) == 0;
}

int StridedSliceChecker::GetConstTensor(const CNodePtr &strided_slice, size_t index, lite::DataInfo *data_info) {
  if (strided_slice == nullptr || data_info == nullptr) {
    MS_LOG(ERROR) << "exist in-parameter is a nullptr.";
    return lite::RET_NULL_PTR;
  }
  if (index >= strided_slice->size() || strided_slice->input(index) == nullptr) {
    MS_LOG(ERROR) << "Strided_slice input is invalid, node is " << strided_slice->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (utils::isa<CNode>(strided_slice->input(index))) {
    MS_LOG(DEBUG) << "Strided_slice " << index << " input is not a constant, node is "
                  << strided_slice->fullname_with_scope();
    return lite::RET_NOT_SUPPORT;
  }
  if (lite::FetchConstData(strided_slice, index, converter::kFmkTypeMs, data_info, false) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get Strided_slice " << index << "-input failed, node is " << strided_slice->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (data_info->data_ptr_ == nullptr ||
      (data_info->data_type_ != kNumberTypeInt && data_info->data_type_ != kNumberTypeInt32)) {
    MS_LOG(ERROR) << "Get Strided_slice's constant failed, node name is " << strided_slice->fullname_with_scope();
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
