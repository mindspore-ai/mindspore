/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/internal/reshape.h"

#include <memory>
#include "kernel/framework_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
bool InternalReshape::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  return true;
}

int InternalReshape::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  input_size_list_.clear();
  output_size_list_.clear();
  auto calc_size_list = [](const std::vector<KernelTensor *> &tensors, std::vector<size_t> *list_ptr) -> bool {
    for (KernelTensor *tensor : tensors) {
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

std::vector<size_t> InternalReshape::GetLaunchIgnoredInputAddressIdx() const {
  static const std::map<std::string, std::vector<size_t>> launch_ignored_input_addr_idx = {{kReshapeOpName, {kIndex1}}};
  if (launch_ignored_input_addr_idx.count(kernel_name_) > 0) {
    return launch_ignored_input_addr_idx.at(kernel_name_);
  } else {
    return {};
  }
}

bool InternalReshape::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto status = aclrtMemcpyAsync(outputs[0]->device_ptr(), outputs[0]->size(), inputs[0]->device_ptr(),
                                 inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "ReshapeKernelMod Launch failed. kernel: " << kernel_name_
                  << ", call rtMemcpyAsync failed, ret = 0x" << status;
    return false;
  }

  return true;
}

MS_INTERNAL_KERNEL_FACTORY_REG(Reshape, InternalReshape);
MS_INTERNAL_KERNEL_FACTORY_REG(ReshapeExt, InternalReshape);
}  // namespace kernel
}  // namespace mindspore
