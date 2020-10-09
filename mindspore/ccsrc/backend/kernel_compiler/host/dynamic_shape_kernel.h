/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_SHAPE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_SHAPE_KERNEL_H_
#include <vector>
#include <memory>
#include <string>
#include "runtime/device/ascend/executor/host_dynamic_kernel.h"
#include "backend/kernel_compiler/host/host_kernel_mod.h"
using HostDynamicKernel = mindspore::device::ascend::HostDynamicKernel;
namespace mindspore {
namespace kernel {
class DynamicShapeKernel : public HostDynamicKernel {
 public:
  DynamicShapeKernel(void *stream, const CNodePtr &cnode_ptr) : HostDynamicKernel(stream, cnode_ptr) {}
  ~DynamicShapeKernel() override = default;
  void Execute() override;
};

class DynamicShapeKernelMod : public HostKernelMod {
 public:
  DynamicShapeKernelMod() = default;
  ~DynamicShapeKernelMod() override = default;
  device::DynamicKernelPtr GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) override;
};
MS_HOST_REG_KERNEL(DynamicShape, DynamicShapeKernelMod);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_DYNAMIC_SHAPE_KERNEL_H_
