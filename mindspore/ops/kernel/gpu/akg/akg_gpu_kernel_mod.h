/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_AKG_GPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_AKG_GPU_KERNEL_MOD_H_
#include <cuda.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/gpu/gpu_kernel_mod.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/kash/kernel_pack.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::string;
using std::vector;

const int MAX_REGISTER_PER_THREAD_BLOCK = 65536;
const int REGISTER_UNIT_IN_WARP = 256;
const int WARP_SIZE = 32;
const int WARP_ALLOC_GRAN = 4;
const int AKG_KERNEL_MOD_BX_IDX = 0;
const int AKG_KERNEL_MOD_BY_IDX = 1;
const int AKG_KERNEL_MOD_BZ_IDX = 2;
const int AKG_KERNEL_MOD_TX_IDX = 3;
const int AKG_KERNEL_MOD_TY_IDX = 4;
const int AKG_KERNEL_MOD_TZ_IDX = 5;

struct GpuKernelMeta {
  CUfunction func_addr_;
  CUmodule module_;
  std::vector<uint32_t> thread_info_;
  GpuKernelMeta(CUfunction funcAddr, CUmodule module, const std::vector<uint32_t> &thread_info)
      : func_addr_(funcAddr), module_(module), thread_info_(thread_info) {}
};
using GpuKernelMetaPtr = std::shared_ptr<GpuKernelMeta>;

class AkgGpuKernelManager {
 public:
  AkgGpuKernelManager();
  virtual ~AkgGpuKernelManager() {
    for (auto iter = infotable_.begin(); iter != infotable_.end(); ++iter) {
      CUresult ret = cuModuleUnload(iter->second->module_);
      if (ret != CUDA_SUCCESS && ret != CUDA_ERROR_DEINITIALIZED) {
        const char *msg = nullptr;
        cuGetErrorName(ret, &msg);
        MS_LOG(ERROR) << "Unload GPU Module failed. cuModuleUnload error message: " << msg;
      }
    }
  }
  CUresult GetFunction(const KernelPackPtr &kernel_pack, bool force_reload, std::vector<uint32_t> *thread_info,
                       CUfunction *func);

 private:
  std::unordered_map<std::string, GpuKernelMetaPtr> infotable_;
};
using AkgGpuKernelManagerPtr = std::shared_ptr<AkgGpuKernelManager>;

class AkgGpuKernelMod : public GpuKernelMod {
 public:
  explicit AkgGpuKernelMod(const KernelPackPtr &kernel_pack);
  virtual ~AkgGpuKernelMod() {}

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  static AkgGpuKernelManagerPtr kernel_manager_;
  std::vector<KernelAttr> GetOpSupport() override { return {}; }

 private:
  KernelPackPtr kernel_pack_;
  std::vector<uint32_t> thread_info_;
  CUfunction kernel_addr_{nullptr};
};
class AkgGpuKernelModDebug : public AkgGpuKernelMod {
 public:
  explicit AkgGpuKernelModDebug(const KernelPackPtr &kernel_pack) : AkgGpuKernelMod(kernel_pack) {}
  virtual ~AkgGpuKernelModDebug() {}
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    auto ptr = reinterpret_cast<CUstream>(stream_ptr);
    CUresult before_launch = cuStreamSynchronize(ptr);
    const char *msg = nullptr;
    if (before_launch != CUDA_SUCCESS) {
      cuGetErrorName(before_launch, &msg);
      MS_LOG(ERROR) << "before_launch sycn failed, Kernel name is : " << kernel_name_ << ", Error message: " << msg;
    }
    auto result = AkgGpuKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
    CUresult after_launch = cuStreamSynchronize(ptr);
    if (after_launch != CUDA_SUCCESS) {
      cuGetErrorName(after_launch, &msg);
      MS_LOG(ERROR) << "after_launch sycn failed, Kernel name is : " << kernel_name_ << ", Error message: " << msg;
    }
    return result;
  }
};
using AkgGpuKernelModPtr = std::shared_ptr<AkgGpuKernelMod>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_GPU_AKG_GPU_KERNEL_MOD_H_
