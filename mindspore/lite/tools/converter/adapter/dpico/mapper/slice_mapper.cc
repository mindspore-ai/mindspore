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

#include "mapper/slice_mapper.h"
#include <memory>
#include <utility>
#include <algorithm>
#include <vector>
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "common/check_base.h"
#include "ops/split.h"
#include "op/slice_operator.h"

namespace mindspore {
namespace dpico {
STATUS SliceMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                        const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto split_prim = api::utils::cast<api::SharedPtr<ops::Split>>(prim);
  MS_ASSERT(split_prim != nullptr);

  auto slice_operator = std::make_unique<mapper::SliceOperator>();
  MS_CHECK_TRUE_MSG(slice_operator != nullptr, RET_ERROR, "slice_operator is nullptr.");

  if (SetCommonAttr(cnode, slice_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  slice_operator->SetOpType(mapper::OpType::SLICE);
  ShapeVector shape;
  if (GetInputShapeFromCNode(cnode, kInputIndex1, &shape) != RET_OK) {
    MS_LOG(ERROR) << "fetch input shape failed.";
    return RET_ERROR;
  }
  if (std::any_of(shape.begin(), shape.end(),
                  [](int64_t dim) { return dim <= 0 || dim > static_cast<int64_t>(UINT32_MAX); })) {
    MS_LOG(ERROR) << "shape is invalid, which is not larger than 0 and less than uint32_max";
    return RET_ERROR;
  }
  MS_ASSERT(shape.size() <= kDims4);
  if (split_prim->GetAttr(ops::kAxis) == nullptr) {
    MS_LOG(ERROR) << "axis attr is nullptr, please check split_checker.";
    return RET_ERROR;
  }
  auto split_axis = split_prim->get_axis();
  split_axis = split_axis < 0 ? split_axis + static_cast<int64_t>(shape.size()) : split_axis;
  if (split_axis > static_cast<int64_t>(kDims4)) {
    MS_LOG(ERROR) << "split axis is invalid.";
    return RET_ERROR;
  }
  slice_operator->SetAxis(static_cast<int32_t>(split_axis));
  if (split_prim->GetAttr(ops::kSizeSplits) != nullptr) {
    auto sizes = split_prim->get_size_splits();
    if (sizes.empty()) {
      MS_LOG(ERROR) << "sizes shouldn't be empty." << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (std::any_of(sizes.begin(), sizes.end(), [](int64_t size) { return size > static_cast<int64_t>(UINT32_MAX); })) {
      MS_LOG(ERROR) << "split sizes is invalid, which is not larger than 0 and less than uint32_max";
      return RET_ERROR;
    }
    std::vector<uint32_t> sizes_u;
    (void)std::transform(sizes.begin(), sizes.end(), std::back_inserter(sizes_u),
                         [](int64_t size) { return static_cast<uint32_t>(size); });
    uint32_t slice_point_cnt = 0;
    for (size_t i = 0; i < sizes_u.size() - 1; i++) {
      if (sizes_u.at(i) >= (static_cast<uint32_t>(shape[split_axis]) - slice_point_cnt)) {
        MS_LOG(ERROR) << "split sizes is invalid, which is larger than the related dim.";
        return RET_ERROR;
      }
      slice_operator->AddSlicePoint(sizes_u.at(i) + slice_point_cnt);
      slice_point_cnt += sizes_u.at(i);
    }
  }

  if (slice_operator->GetSlicePointVec().empty()) {
    if (split_prim->GetAttr(ops::kOutputNum) == nullptr) {
      MS_LOG(ERROR) << "cannot determine split points.";
      return RET_ERROR;
    }
    auto output_num = api::GetValue<int64_t>(split_prim->GetAttr(ops::kOutputNum));
    MS_CHECK_TRUE_MSG(output_num != 0, RET_ERROR, "output_num is 0.");
    if (shape[split_axis] % output_num != 0) {
      MS_LOG(ERROR) << "output_num is 0 or split op is invalid, which input shape cannot be splited.";
      return RET_ERROR;
    }
    uint32_t size_of_each_out = static_cast<uint32_t>(shape[split_axis] / output_num);
    for (uint32_t i = 1; i < static_cast<uint32_t>(output_num); ++i) {
      slice_operator->AddSlicePoint(i * size_of_each_out);
    }
  }
  base_operators->push_back(std::move(slice_operator));
  return RET_OK;
}
REG_MAPPER(Split, SliceMapper)
}  // namespace dpico
}  // namespace mindspore
