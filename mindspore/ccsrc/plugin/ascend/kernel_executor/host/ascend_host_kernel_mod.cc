/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/kernel_executor/host/ascend_host_kernel_mod.h"

#include "utils/convert_utils_base.h"

namespace mindspore {
namespace kernel {
bool GetShapeSize(const ShapeVector &shape, const TypePtr &type_ptr, int64_t *size_i) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  MS_EXCEPTION_IF_NULL(size_i);
  size_i[0] = 1;

  size_t type_byte = GetTypeByte(type_ptr);
  if (type_byte == 0) {
    return false;
  }
  for (size_t j = 0; j < shape.size(); j++) {
    if (shape[j] <= 0) {
      MS_LOG(DEBUG) << "shape[" << shape << "] has invalid value(less equal 0), set size to 0";
      size_i[0] = 0;
      return true;
    }
    size_i[0] = LongMulWithOverflowCheck(size_i[0], shape[j]);
  }
  size_i[0] = LongMulWithOverflowCheck(size_i[0], SizeToInt(type_byte));
  return true;
}

int AscendHostKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  input_size_list_.clear();
  output_size_list_.clear();

  auto calc_size_list = [](const std::vector<KernelTensor *> &tensors, std::vector<size_t> *list_ptr) -> bool {
    for (KernelTensor *tensor : tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      int64_t size = 1;
      if (!GetShapeSize(tensor->GetShapeVector(), TypeIdToType(tensor->dtype_id()), &size)) {
        return false;
      }
      list_ptr->push_back(LongToSize(size));
    }
    return true;
  };

  if (!calc_size_list(inputs, &input_size_list_)) {
    return KRET_RESIZE_FAILED;
  }
  if (!calc_size_list(outputs, &output_size_list_)) {
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}
}  // namespace kernel
}  // namespace mindspore
