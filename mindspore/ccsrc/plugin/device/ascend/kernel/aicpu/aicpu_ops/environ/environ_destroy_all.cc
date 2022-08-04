/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "environ/environ_destroy_all.h"
#include "environ/aicpu_environ_manager.h"

namespace aicpu {
uint32_t EnvironDestroyAllKernel::DoCompute() {
  AICPU_LOGD("Destroy all env handle");
  EnvironMgr::GetInstance().Clear();
  return kAicpuKernelStateSucess;
}

uint32_t EnvironDestroyAllKernel::ParseKernelParam() {
  AICPU_LOGD("Enter ParseKernelParam.");
  if (!EnvironMgr::GetInstance().IsScalarTensor(node_def_.outputs(aicpu::kIndex0))) {
    AICPU_LOGE("The output is not scalar tensor.");
    return kAicpuKernelStateInvalid;
  }
  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t EnvironDestroyAll(void *param) {
  aicpu::EnvironDestroyAllKernel environDestroyAllKernel;
  return environDestroyAllKernel.Compute(param);
}
}
