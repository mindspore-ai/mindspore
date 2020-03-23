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
#include "device/gpu/gpu_kernel_build.h"
#include <string>
#include "kernel/kernel.h"
#include "kernel/akg/akgkernelbuild.h"
#include "kernel/akg/gpu/akg_gpu_kernel_build.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "operator/ops.h"
#include "pybind11/stl.h"
#include "session/anf_runtime_algorithm.h"
namespace mindspore {
namespace device {
namespace gpu {
namespace py = pybind11;
void GpuBuild(const KernelGraphPtr &kernel_graph) {
  kernel::KernelMeta *bin_map = kernel::KernelMeta::GetInstance();
  if (!bin_map->ReadIndex(kernel::kGpuKernelMeta)) {
    MS_LOG(INFO) << "kernel cache miss, cache directory will be created later.";
  } else {
    MS_LOG(INFO) << "cache initialize to[" << kernel::kGpuKernelMeta << "].";
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    std::string kernel_name = session::AnfRuntimeAlgorithm::GetCNodeName(kernel);
    if (kernel_name == prim::kPrimTupleGetItem->name() || kernel_name == prim::kPrimMakeTuple->name() ||
        kernel_name == prim::kPrimDepend->name() || kernel_name == prim::kPrimStateSetItem->name()) {
      continue;
    }

    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) == KernelType::AUTO_DIFF_KERNEL) {
      auto gpu_kernel_ptr = kernel::AkgGpuKernelBuild(kernel);
      if (!gpu_kernel_ptr) {
        MS_LOG(EXCEPTION) << "Build akg kernel op[" << kernel_name << "] failed";
      }
      session::AnfRuntimeAlgorithm::SetKernelMod(gpu_kernel_ptr, kernel.get());
    } else {
      auto gpu_kernel_ptr = kernel::GpuKernelFactory::GetInstance().Create(kernel_name, kernel);
      if (!gpu_kernel_ptr) {
        MS_LOG(EXCEPTION) << "Build gpu kernel op[" << kernel_name << "] failed";
      }
      if (!gpu_kernel_ptr->Init(kernel)) {
        MS_LOG(EXCEPTION) << "Initialize gpu kernel op[" << kernel_name << "] failed.";
      }
      session::AnfRuntimeAlgorithm::SetKernelMod((kernel::KernelModPtr)gpu_kernel_ptr, kernel.get());
    }
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
