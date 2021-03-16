/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/ascend_launch_mul.h"

#include <memory>
#include "abstract/utils.h"
#include "runtime/mem.h"
#include "backend/session/single_kernel_graph.h"
#include "frontend/parallel/context.h"

namespace mindspore::device::ascend {
void AscendLaunchMul::FreeDeviceMem(void *addr) { AscendLaunchKernel::FreeDeviceMem(addr); }

size_t AscendLaunchMul::AlignSizeForLaunchKernel(size_t size) {
  return AscendLaunchKernel::AlignSizeForLaunchKernel(size);
}

uint8_t *AscendLaunchMul::AllocDeviceMem(size_t size) { return AscendLaunchKernel::AllocDeviceMem(size); }

void AscendLaunchMul::KernelSelect(std::shared_ptr<session::KernelGraph> kernel_graph) {
  AscendLaunchKernel::KernelSelect(kernel_graph);
}

void AscendLaunchMul::KernelBuild(std::shared_ptr<session::KernelGraph> kernel_graph) {
  AscendLaunchKernel::KernelBuild(kernel_graph);
}

void AscendLaunchMul::LaunchOpKernel() {
  kernel_mod_ = ObtainLaunchMulKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  // construct mul inputs addr
  ObtainMulInputsAddr();
  // launch mul
  LaunchSingleKernel(inputs_addr_);
}

void AscendLaunchMul::FreeLaunchDeviceMem() {
  FreeInputDeviceMemory();
  FreeOutputAndWorkspaceDeviceMem();
}

void AscendLaunchMul::CopyHostMemToDevice(size_t origin_size, size_t dst_size) {
  auto ret = rtMemcpyAsync(input2_addr_, dst_size, &input2_value_, origin_size, RT_MEMCPY_HOST_TO_DEVICE, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "launch rtMemcpyAsync failed, ret:" << ret;
  }
}
}  // namespace mindspore::device::ascend
