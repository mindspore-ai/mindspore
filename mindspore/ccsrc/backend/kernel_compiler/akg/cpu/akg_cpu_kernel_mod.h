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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_CPU_AKG_CPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_CPU_AKG_CPU_KERNEL_MOD_H_
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include "backend/kernel_compiler/kernel.h"

namespace mindspore {
namespace kernel {
class CpuKernelManager {
 public:
  CpuKernelManager() = default;
  ~CpuKernelManager();

  void *GetFunction(const std::string &kernel_name);

 private:
  void *SearchFunc(const std::string &kernel_name) const;
  void *SearchFuncWithSharedLock(const std::string &kernel_name) const;

  // cache the kernel function: kernel_name -> {kernel_func, so_handle}
  std::unordered_map<std::string, std::pair<void *, void *>> cpu_func_map_;
  mutable std::shared_mutex mutex_;
};
using CpuKernelManagerPtr = std::shared_ptr<CpuKernelManager>;
class CpuKernelMod : public KernelMod {
 public:
  explicit CpuKernelMod(const KernelPackPtr &kp);
  ~CpuKernelMod() = default;

  void SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }
  void SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }
  void SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  static CpuKernelManagerPtr kernelmanager_;

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;  // workspace is not used in cpu kernel.
  void *launch_func_;
  std::string kernel_name_;
};

using CpuKernelModPtr = std::shared_ptr<CpuKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_CPU_AKG_CPU_KERNEL_MOD_H_
