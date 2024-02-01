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
#include "plugin/device/ascend/kernel/internal/split.h"

namespace {
const int64_t DEFAULT_OUTPUT_NUM = 2;
}

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalSplit::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::SplitParam split_param;
  param_ptr->opId = internal::OpId::Split;

  if (primitive_->HasAttr("axis")) {
    auto value_str = primitive_->GetAttr("axis");
    MS_EXCEPTION_IF_NULL(value_str);
    int64_t axis = GetValue<int64_t>(value_str);
    split_param.splitDim = axis;
  } else {
    int64_t default_axis = 0;
    split_param.splitDim = default_axis;
  }

  if (primitive_->HasAttr("output_num")) {
    auto value_str = primitive_->GetAttr("output_num");
    MS_EXCEPTION_IF_NULL(value_str);
    int64_t output_num = GetValue<int64_t>(value_str);
    split_param.splitNum = output_num;
  } else {
    split_param.splitNum = DEFAULT_OUTPUT_NUM;
  }

  param_ptr->specificParam = split_param;
  return param_ptr;
}
void InternalSplit::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  outputsIdxMap_[0] = 0;
  outputsIdxMap_[1] = 1;
  int64_t splitNum = DEFAULT_OUTPUT_NUM;
  if (primitive_->HasAttr("output_num")) {
    auto value_str = primitive_->GetAttr("output_num");
    MS_EXCEPTION_IF_NULL(value_str);
    splitNum = GetValue<int64_t>(value_str);
  }
  if (splitNum > DEFAULT_OUTPUT_NUM) {
    outputsIdxMap_[2] = 2;
  }
}

MS_INTERNAL_KERNEL_FACTORY_REG(Split, InternalSplit);
}  // namespace kernel
}  // namespace mindspore
