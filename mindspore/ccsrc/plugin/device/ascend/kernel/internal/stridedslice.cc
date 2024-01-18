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
#include <memory>
#include "stridedslice.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalStridedSlice::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  if (GetValue<int64_t>(primitive_->GetAttr("begin_mask")) != 0 ||
      GetValue<int64_t>(primitive_->GetAttr("end_mask")) != 0 ||
      GetValue<int64_t>(primitive_->GetAttr("ellipsis_mask")) != 0 ||
      GetValue<int64_t>(primitive_->GetAttr("new_axis_mask")) != 0 ||
      GetValue<int64_t>(primitive_->GetAttr("shrink_axis_mask")) != 0) {
    MS_LOG(ERROR) << "internal op StridedSlice only support: begin_mask=0, end_mask=0, ellipsis_mask=0, "
                     "new_axis_mask=0, shrink_axis_mask=0";
    return nullptr;
  }
  auto strides = inputs[3]->GetValue<std::vector<size_t>>().value();
  for (size_t value : strides) {
    if (value != 1) {
      std::ostringstream stream;
      if (!strides.empty()) {
        copy(strides.begin(), strides.end() - 1, std::ostream_iterator<size_t>(stream, ", "));
      }
      MS_LOG(ERROR) << "internal op StridedSlice only support stride 1, but input strides is" << stream.str() << ")";
      return nullptr;
    }
  }

  internal::SliceParam slice_param;
  std::vector<size_t> begin = inputs[1]->GetValue<std::vector<size_t>>().value();
  std::vector<size_t> end = inputs[2]->GetValue<std::vector<size_t>>().value();
  for (size_t i = 0; i < begin.size(); ++i) {
    slice_param.offsets.emplace_back(begin[i]);
    slice_param.size.emplace_back(end[i] - begin[i]);
  }

  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::Slice;
  param_ptr->specificParam = slice_param;
  return param_ptr;
}
void InternalStridedSlice::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  outputsIdxMap_[0] = 0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(StridedSlice, InternalStridedSlice);
}  // namespace kernel
}  // namespace mindspore
