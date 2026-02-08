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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_MEMCPY_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_MEMCPY_H_

#include <memory>
#include <string>
#include <vector>
#include "kernel/ascend/simu/kernel_mod_impl/simu_kernel.h"

namespace mindspore {
namespace kernel {
class SimuMemcpyKernel : public SimuKernel {
 public:
  explicit SimuMemcpyKernel(const std::string &op_name) : op_name_(op_name) {}
  ~SimuMemcpyKernel() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  const std::string op_name_;
};

#define MS_SIMU_MEMCPY_REG(KNAME)                       \
  class Simu##KNAME##Kernel : public SimuMemcpyKernel { \
   public:                                              \
    Simu##KNAME##Kernel() : SimuMemcpyKernel(#KNAME) {} \
    ~Simu##KNAME##Kernel() override = default;          \
  };                                                    \
  MS_SIMU_REG_KERNEL(KNAME, Simu##KNAME##Kernel);

MS_SIMU_MEMCPY_REG(AllReduce);
MS_SIMU_MEMCPY_REG(AllGather);
MS_SIMU_MEMCPY_REG(ReduceScatter);
MS_SIMU_MEMCPY_REG(Gather);
MS_SIMU_MEMCPY_REG(Reduce);
MS_SIMU_MEMCPY_REG(Scatter);
MS_SIMU_MEMCPY_REG(Broadcast);
MS_SIMU_MEMCPY_REG(AllToAll);
MS_SIMU_MEMCPY_REG(AlltoAllV);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_MEMCPY_H_
