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
bool InternalStridedSlice::CheckMasks(const std::vector<KernelTensor *> &inputs) {
  bool mask_all_zero = true;
  constexpr size_t kInputNumber = 9;
  if (primitive_->HasAttr("begin_mask") && primitive_->HasAttr("end_mask") && primitive_->HasAttr("ellipsis_mask") &&
      primitive_->HasAttr("new_axis_mask") && primitive_->HasAttr("shrink_axis_mask")) {
    mask_all_zero = (GetValue<int64_t>(primitive_->GetAttr("begin_mask")) == kDim0 &&
                     GetValue<int64_t>(primitive_->GetAttr("end_mask")) == kDim0 &&
                     GetValue<int64_t>(primitive_->GetAttr("ellipsis_mask")) == kDim0 &&
                     GetValue<int64_t>(primitive_->GetAttr("new_axis_mask")) == kDim0 &&
                     GetValue<int64_t>(primitive_->GetAttr("shrink_axis_mask")) == kDim0);
  } else if (inputs.size() >= kInputNumber) {
    mask_all_zero = (inputs[kIndex4]->GetValueWithCheck<int64_t>() == kDim0 &&
                     inputs[kIndex5]->GetValueWithCheck<int64_t>() == kDim0 &&
                     inputs[kIndex6]->GetValueWithCheck<int64_t>() == kDim0 &&
                     inputs[kIndex7]->GetValueWithCheck<int64_t>() == kDim0 &&
                     inputs[kIndex8]->GetValueWithCheck<int64_t>() == kDim0);
  } else {
    mask_all_zero = false;
  }
  return mask_all_zero;
}

internal::OpParamPtr InternalStridedSlice::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  if (!CheckMasks(inputs)) {
    MS_LOG(WARNING) << "Internal op StridedSlice only support: begin_mask=0, end_mask=0, ellipsis_mask=0, "
                       "new_axis_mask=0, shrink_axis_mask=0";
  }

  auto strides = inputs[kIndex3]->GetValueWithCheck<std::vector<size_t>>();
  if (std::any_of(strides.begin(), strides.end(), [](const size_t value) { return value != kDim1; })) {
    std::ostringstream stream;
    if (!strides.empty()) {
      copy(strides.begin(), strides.end() - 1, std::ostream_iterator<size_t>(stream, ", "));
    }
    MS_LOG(ERROR) << "Internal op StridedSlice only support stride 1, but input strides is" << stream.str() << ")";
    return nullptr;
  }

  auto input_shape = inputs[kIndex0]->GetShape()->GetShapeVector();
  internal::SliceParam slice_param;
  std::vector<int64_t> begin = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  std::vector<int64_t> end = inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
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

std::vector<size_t> InternalStridedSlice::GetLaunchIgnoredInputAddressIdx() const {
  return {kIndex1, kIndex2, kIndex3};
}

MS_INTERNAL_KERNEL_FACTORY_REG(StridedSlice, InternalStridedSlice);
}  // namespace kernel
}  // namespace mindspore
