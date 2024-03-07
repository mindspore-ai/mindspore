/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/stridedslice.h"

#include <memory>
#include <algorithm>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalStridedSlice::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  if (GetValue<int64_t>(primitive_->GetAttr("begin_mask")) != kDim0 ||
      GetValue<int64_t>(primitive_->GetAttr("end_mask")) != kDim0 ||
      GetValue<int64_t>(primitive_->GetAttr("ellipsis_mask")) != kDim0 ||
      GetValue<int64_t>(primitive_->GetAttr("new_axis_mask")) != kDim0 ||
      GetValue<int64_t>(primitive_->GetAttr("shrink_axis_mask")) != kDim0) {
    MS_LOG(WARNING) << "internal op StridedSlice only support: begin_mask=0, end_mask=0, ellipsis_mask=0, "
                       "new_axis_mask=0, shrink_axis_mask=0";
  }
  auto strides = inputs[3]->GetValue<std::vector<size_t>>().value();
  if (std::any_of(strides.begin(), strides.end(), [](const size_t value) { return value != kDim1; })) {
    std::ostringstream stream;
    if (!strides.empty()) {
      copy(strides.begin(), strides.end() - 1, std::ostream_iterator<size_t>(stream, ", "));
    }
    MS_LOG(ERROR) << "internal op StridedSlice only support stride 1, but input strides is" << stream.str() << ")";
    return nullptr;
  }

  auto input_shape = inputs[kIndex0]->GetShape()->GetShapeVector();
  internal::SliceParam slice_param;
  std::vector<int64_t> begin = inputs[kIndex1]->GetValue<std::vector<int64_t>>().value();
  std::vector<int64_t> end = inputs[kIndex2]->GetValue<std::vector<int64_t>>().value();
  size_t i = 0;
  for (; i < begin.size(); ++i) {
    int64_t bg = begin[i] >= 0 ? begin[i] : input_shape[i] + begin[i];
    int64_t ed = end[i] >= 0 ? end[i] : input_shape[i] + end[i];
    slice_param.offsets.emplace_back(bg);
    slice_param.size.emplace_back(ed - bg);
  }
  if (begin.size() < input_shape.size()) {
    for (; i < input_shape.size(); ++i) {
      slice_param.offsets.emplace_back(kDim0);
      slice_param.size.emplace_back(input_shape[i]);
    }
  }

  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::Slice;
  param_ptr->specificParam = slice_param;
  return param_ptr;
}
void InternalStridedSlice::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  outputsIdxMap_[kIndex0] = kIndex0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(StridedSlice, InternalStridedSlice);
}  // namespace kernel
}  // namespace mindspore
