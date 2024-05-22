/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_MOD_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>

#include "kernel/kernel.h"
#include "./internal_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/ascend/kernel/internal/tiling_cache.h"
#include "utils/ms_context.h"
#include "include/backend/debug/profiler/profiling.h"

namespace mindspore {
namespace kernel {
class InternalKernelMod : public KernelMod {
 public:
  explicit InternalKernelMod(std::string &&op_type) : op_type_(std::move(op_type)) {
    auto ascend_profiler = profiler::Profiler::GetInstance(kAscendDevice);
    MS_EXCEPTION_IF_NULL(ascend_profiler);
    enable_profiler_flag_ = ascend_profiler->GetEnableFlag();
  }
  virtual ~InternalKernelMod();

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in internal kernel.";
  }

 protected:
  virtual int Build(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  virtual void SetInOutIdx() = 0;
  virtual internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) = 0;
  virtual uint64_t GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs);
  virtual void SetTilingInfo(const uint64_t key);
  std::shared_ptr<internal::InternelKernelImpl> impl_;
  std::unordered_map<size_t, size_t> inputsIdxMap_;
  std::unordered_map<size_t, size_t> outputsIdxMap_;
  std::vector<internal::Tensor *> inputs_;
  std::vector<internal::Tensor *> outputs_;
  TilingInfo tiling_info_;
  std::string op_type_;
  bool enable_profiler_flag_{False};
};

using InternalKernelModPtr = std::shared_ptr<InternalKernelMod>;
using InternalKernelModPtrList = std::vector<InternalKernelModPtr>;

#define MS_INTERNAL_KERNEL_FACTORY_REG(NAME, DERIVE) MS_KERNEL_FACTORY_REG(InternalKernelMod, NAME, DERIVE)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_MOD_H_
