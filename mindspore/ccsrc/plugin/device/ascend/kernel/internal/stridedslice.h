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

#ifndef MINDSPORE_STRIDEDSLICE_H
#define MINDSPORE_STRIDEDSLICE_H
#include <vector>
#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
namespace mindspore {
namespace kernel {
class InternalStridedSlice : public InternalKernelMod {
 public:
  InternalStridedSlice() : InternalKernelMod("StridedSlice") {}
  ~InternalStridedSlice() = default;

 protected:
  internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs);
  void SetInOutIdx();
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_STRIDEDSLICE_H
