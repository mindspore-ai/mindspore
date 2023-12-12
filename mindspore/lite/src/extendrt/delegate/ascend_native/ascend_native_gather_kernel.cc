/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_native/ascend_native_gather_kernel.h"
#include <vector>
#include <memory>
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/gather.h"
#include "ops/gather.h"

namespace mindspore::kernel {
using mindspore::ops::kNameGather;

int AscendNativeGatherKernel::Prepare() { return kSuccess; }

int AscendNativeGatherKernel::Run() {
  MS_LOG(INFO) << "AscendNativeGatherKernel::Execute";
  const std::vector<InferTensor *> &in_tensors = this->in_tensors();
  if (in_tensors.size() != THREE_TENSOR) {
    MS_LOG(ERROR) << "AscendNativeGatherKernel inputs number should be 3, instead got " << in_tensors.size();
    return kLiteError;
  }
  int64_t axis = 0;
  auto axis_tensor = reinterpret_cast<int64_t *>(in_tensors.at(THIRD_INPUT)->device_data());
  ascend_native::CopyDeviceFp32ToHostFp32(axis_tensor, &axis, 2, const_cast<void *>(this->get_stream()));
  auto shape = in_tensors.at(FIRST_INPUT)->shape();
  size_t num_tiles = 1;
  size_t m = 1;
  for (auto i = 0; i < axis; i++) {
    num_tiles *= shape.at(i);
  }
  for (size_t i = axis + 1; i < shape.size(); i++) {
    m *= shape.at(i);
  }

  ascend_native::GatherFp16(out_tensors().at(FIRST_INPUT)->device_data(), in_tensors.at(FIRST_INPUT)->device_data(),
                            reinterpret_cast<int *>(in_tensors.at(SECOND_INPUT)->device_data()),
                            in_tensors.at(SECOND_INPUT)->shape().at(0), m, num_tiles, shape.at(axis),
                            const_cast<void *>(get_stream()));
  return kSuccess;
}
REGISTER_ASCEND_NATIVE_CREATOR(kNameGather, AscendNativeGatherKernel)
}  // namespace mindspore::kernel
