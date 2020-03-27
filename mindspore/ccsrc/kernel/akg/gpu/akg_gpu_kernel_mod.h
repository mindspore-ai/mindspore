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

#ifndef MINDSPORE_CCSRC_KERNEL_AKG_GPU_AKG_GPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_KERNEL_AKG_GPU_AKG_GPU_KERNEL_MOD_H_
#include <cuda.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
struct GpuKernelMeta {
  CUfunction func_addr_;
  CUmodule module_;
  std::vector<uint32_t> thread_info_;
  GpuKernelMeta(CUfunction funcAddr, CUmodule module, const std::vector<uint32_t> &thread_info)
      : func_addr_(funcAddr), module_(module), thread_info_(thread_info) {}
};
using GpuKernelMetaPtr = std::shared_ptr<GpuKernelMeta>;

class GpuKernelManager {
 public:
  GpuKernelManager();
  virtual ~GpuKernelManager() {
    for (auto iter = infotable_.begin(); iter != infotable_.end(); ++iter) {
      CUresult ret = cuModuleUnload(iter->second->module_);
      if (ret != CUDA_SUCCESS && ret != CUDA_ERROR_DEINITIALIZED) {
        MS_LOG(ERROR) << "Unload GPU Module failed.";
      }
    }
  }
  CUresult GetFunction(const KernelPackPtr &kernel_pack, bool force_reload, std::vector<uint32_t> *thread_info,
                       CUfunction *func);

 private:
  std::unordered_map<std::string, GpuKernelMetaPtr> infotable_;
};
using GpuKernelManagerPtr = std::shared_ptr<GpuKernelManager>;

class GpuKernelMod : public KernelMod {
 public:
  explicit GpuKernelMod(const KernelPackPtr &kernel_pack);
  virtual ~GpuKernelMod() {}

  void SetInputSizeList(const std::vector<size_t> &size_list);
  void SetOutputSizeList(const std::vector<size_t> &size_list);
  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override;

  static GpuKernelManagerPtr kernelmanager_;

 private:
  KernelPackPtr kernel_pack_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

using GpuKernelModPtr = std::shared_ptr<GpuKernelMod>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_AKG_GPU_AKG_GPU_KERNEL_MOD_H_
