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

#include "plugin/device/ascend/kernel/internal/softmax.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalSoftmax::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::SoftmaxParam softmax_param;
  param_ptr->opId = internal::OpId::Softmax;

  if (primitive_->HasAttr("axis")) {
    auto value_str = primitive_->GetAttr("axis");
    MS_EXCEPTION_IF_NULL(value_str);

    if (value_str->isa<ValueSequence>()) {
      auto axis_list = GetValue<std::vector<int64_t>>(value_str);
      softmax_param.axes.clear();
      for (auto axis : axis_list) {
        softmax_param.axes.push_back(axis);
      }
    } else if (value_str->isa<Int64Imm>()) {
      int64_t axis = GetValue<int64_t>(value_str);
      softmax_param.axes = {axis};
    } else {
      MS_LOG(ERROR) << primitive_->name() << " attr axis mst be int or tuple";
    }
  } else {
    int64_t default_axis = -1;
    softmax_param.axes = {default_axis};
  }

  param_ptr->specificParam = softmax_param;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(Softmax, InternalSoftmax);
}  // namespace kernel
}  // namespace mindspore
