/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "runtime/device/gpu/gpu_kernel_build.h"
#include <string>
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/akg/gpu/akg_gpu_kernel_build.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/common_utils.h"
#include "frontend/operator/ops.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_build_client.h"
#include "runtime/device/gpu/cuda_env_checker.h"

namespace mindspore {
namespace device {
namespace gpu {
void CreateGPUKernel(const std::vector<CNodePtr> &kernels) {
  kernel::KernelMeta *bin_map = kernel::KernelMeta::GetInstance();
  MS_EXCEPTION_IF_NULL(bin_map);
  bool already_check_nvcc = false;
  std::vector<AnfNodePtr> akg_nodes;
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    std::string kernel_name = session::AnfRuntimeAlgorithm::GetCNodeName(kernel);
    if (kernel_name == prim::kPrimTupleGetItem->name() || kernel_name == prim::kPrimMakeTuple->name() ||
        kernel_name == prim::kPrimDepend->name() || kernel_name == prim::kPrimStateSetItem->name()) {
      continue;
    }

    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) == KernelType::AKG_KERNEL) {
      if (!bin_map->initialized()) {
        bin_map->Initialize();
      }
      if (!already_check_nvcc) {
        already_check_nvcc = true;
        if (!CudaEnvChecker::GetInstance().CheckNvccInPath()) {
          MS_LOG(EXCEPTION)
            << "Failed to find nvcc compiler, please add nvcc position to the PATH environment variable, run "
               "the command: export PATH=${CUDA_PATH}/bin:${PATH}, CUDA_PATH is the installation path of the "
               "cuda library(eg. /usr/local/cuda).";
        }
      }
      akg_nodes.push_back(kernel);
    } else if (!AnfAlgo::IsControlOpExecInBackend(kernel)) {
      auto gpu_kernel_ptr = kernel::GpuKernelFactory::GetInstance().Create(kernel_name, kernel);
      if (!gpu_kernel_ptr) {
        MS_LOG(EXCEPTION) << "Build gpu kernel op[" << kernel->fullname_with_scope() << "] failed";
      }
      MS_EXCEPTION_IF_NULL(kernel);
      if (!gpu_kernel_ptr->Init(kernel)) {
        MS_LOG(EXCEPTION) << "Initialize gpu kernel op[" << kernel->fullname_with_scope() << "] failed.";
      }
      gpu_kernel_ptr->InitDynamicKernel(kernel);
      gpu_kernel_ptr->DynamicKernel()->Initialize();
      session::AnfRuntimeAlgorithm::SetKernelMod((kernel::KernelModPtr)gpu_kernel_ptr, kernel.get());
    }
  }

  kernel::AkgGpuKernelBuilder akg_gpu_kernel_builder;
  (void)akg_gpu_kernel_builder.AkgKernelParallelBuild(akg_nodes);
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
