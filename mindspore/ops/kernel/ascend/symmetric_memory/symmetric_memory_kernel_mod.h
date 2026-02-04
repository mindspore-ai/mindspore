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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_KERNEL_MOD_H_

#include <memory>
#include <vector>
#include <string>

#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/symmetric_memory.h"
#include "kernel/ascend/internal/tiling_mem_mgr.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

#include "kernel/ascend/symmetric_memory/symmetric_memory_tiling_cache.h"
#include "kernel/ascend/internal/internal_spinlock.h"
#include "kernel/ascend/symmetric_memory/symmetric_memory_kernel_in_out_map.h"
#include "kernel/ascend/symmetric_memory/symmetric_memory_helper.h"
#include "tools/profiler/profiling.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
class SymmetricMemoryKernelMod : public KernelMod {
 public:
  SymmetricMemoryKernelMod() {
    ascend_profiler_ = profiler::Profiler::GetInstance(kAscendDevice);
    MS_EXCEPTION_IF_NULL(ascend_profiler_);
  }

  virtual ~SymmetricMemoryKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in symmetric_memory kernel.";
  }

  void set_fullname(const std::string &fullname) override { fullname_ = fullname; }

 protected:
  virtual bool IsNeedRecreate(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  virtual symmetricmemory::SymmetricMemoryOpPtr CreateKernel(const symmetricmemory::InputsImmutableInfoList &inputs,
                                                             const symmetricmemory::OutputsImmutableInfoList &outputs,
                                                             const std::vector<KernelTensor *> &ms_inputs,
                                                             const std::vector<KernelTensor *> &ms_outputs) {
    return nullptr;
  }

  virtual uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs);

  symmetricmemory::SymmetricMemoryOpPtr symmetric_memory_op_{nullptr};
  std::vector<size_t> symmetric_memory_to_ms_input_indices_mapper_;
  std::vector<size_t> symmetric_memory_to_ms_output_indices_mapper_;
  symmetricmemory::ShapeInfoList symmetric_memory_inputs_shape_;
  symmetricmemory::ShapeInfoList symmetric_memory_outputs_shape_;
  symmetricmemory::InputsAddrList symmetric_memory_inputs_addr_;
  symmetricmemory::OutputsAddrList symmetric_memory_outputs_addr_;
  symmetricmemory::WsAddrList symmetric_memory_wss_addr_;

 private:
  std::shared_ptr<profiler::Profiler> ascend_profiler_{nullptr};
  void GetOrGenerateTiling(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  inline void UpdateAddr(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                         const std::vector<KernelTensor *> &workspace);
  void GetSymmetricMemoryKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  uint64_t last_key_{0};
  TilingCacheItemPtr last_item_{nullptr};
  std::vector<size_t> recreate_cared_indices_;
  std::string fullname_;
  static SimpleSpinLock lock_;
};

using SymmetricMemoryKernelModPtr = std::shared_ptr<SymmetricMemoryKernelMod>;
using SymmetricMemoryKernelModPtrList = std::vector<SymmetricMemoryKernelModPtr>;

#define MS_SYMMETRIC_MEMORY_KERNEL_FACTORY_REG(PRIM_NAME_STR, SYMMETRIC_MEMORY_NAME_VAR, DERIVE)      \
  MS_KERNEL_FACTORY_REG(SymmetricMemoryKernelMod, PRIM_NAME_STR, DERIVE);                             \
  static const NameMappingRegistrar g_##PRIM_NAME_STR##_ms_to_symmetric_memory_mapper(#PRIM_NAME_STR, \
                                                                                      SYMMETRIC_MEMORY_NAME_VAR);

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_KERNEL_MOD_H_
