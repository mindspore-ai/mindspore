/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_STITCH_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_STITCH_GPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class DynamicStitchKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  DynamicStitchKernelMod();
  ~DynamicStitchKernelMod();

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;
  void ResetResource() noexcept override;
  constexpr static size_t kDivNum2 = 2;

 protected:
  void SyncData() override;
  void InitSizeLists() override;

 private:
  size_t n_;
  size_t real_ele_num_;
  int max_index_;
  size_t one_data_ele_num_;
  size_t data_type_size_;
  void *stream_ptr_;
};

MS_REG_GPU_KERNEL(DynamicStitch, DynamicStitchKernelMod)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_STITCH_GPU_KERNEL_H_
