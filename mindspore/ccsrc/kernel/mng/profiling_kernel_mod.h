/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PROFILING_PROFILING_KERNEL_MOD_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PROFILING_PROFILING_KERNEL_MOD_H_
#include <vector>
#include "kernel/mng/rt_kernel.h"
namespace mindspore {
namespace kernel {
class ProfilingKernelMod : public RtKernel {
 public:
  ProfilingKernelMod() = default;
  ~ProfilingKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;
  bool Init(const AnfNodePtr &anf_node) override;

 private:
  uint64_t log_id_{0};
  bool notify_{true};
  uint32_t flags_{0};
};
MS_REG_RTKERNEL(profiling, ProfilingKernelMod);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PROFILING_PROFILING_KERNEL_MOD_H_
