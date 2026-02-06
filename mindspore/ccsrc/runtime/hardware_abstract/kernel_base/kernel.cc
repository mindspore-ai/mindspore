/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

#include <algorithm>
#include <numeric>
#include <set>

#include "ir/format_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kInvalidShape = -2;

int KernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KRET_OK;
  workspace_size_list_.clear();
  output_size_list_.clear();

  for (size_t idx = 0; idx < outputs.size(); idx++) {
    auto &output = outputs[idx];
    size_t tensor_size = 0;
    MS_EXCEPTION_IF_NULL(output);
    size_t type_size = UnitSizeInBytes(output->dtype_id());
    if (type_size == 0) {
      MS_LOG(INFO) << "The type size is 0, type: " << TypeIdToType(output->dtype_id())->ToString();
    }

    const auto &shape = output->GetShapeVector();
    if (!IsValidShape(shape)) {
      MS_LOG(WARNING) << "Invalid shape:" << mindspore::ToString(shape) << ", kernel name:" << kernel_name();
      // Note:
      // If output shape is unknown, the op is a compute-depended op, and the output_size_list_ can be set by default
      // size: type_size.
      tensor_size = type_size;
      ret = KRET_UNKNOWN_OUT_SHAPE;
    } else {
      if (shape.empty()) {
        tensor_size = type_size;
      } else {
        auto cur_out_shape_num = SizeOf(shape);
        tensor_size = cur_out_shape_num * type_size;
        if (type_size != 0 && tensor_size / type_size != cur_out_shape_num) {
          MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", the shape of outputs[" << output_size_list_.size()
                                   << "]: " << shape
                                   << " is too big, mindspore cannot apply for such a large amount of memory.";
        }
      }
    }
    (void)output_size_list_.emplace_back(tensor_size);
  }
  return static_cast<int>(ret);
}

std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensor *> &tensors) {
  std::vector<std::vector<int64_t>> shapes(tensors.size());
  for (size_t idx = 0; idx < shapes.size(); idx++) {
    shapes[idx] = tensors[idx]->GetShapeVector();
  }
  return shapes;
}
}  // namespace kernel
}  // namespace mindspore
