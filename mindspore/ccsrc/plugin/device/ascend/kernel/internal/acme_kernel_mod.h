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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_KERNEL_MOD_H_

#include <memory>
#include <vector>
#include <string>

#include "kernel/kernel.h"
#include "acme/include/acme.h"
#include "acme/tiling_mem_mgr.h"
#include "plugin/factory/ms_factory.h"

#include "plugin/device/ascend/kernel/internal/acme/acme_tiling_cache.h"

namespace mindspore {
namespace kernel {
class AcmeKernelMod : public KernelMod {
 public:
  AcmeKernelMod() = default;
  virtual ~AcmeKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in internal kernel.";
  }

  void set_fullname(const std::string &fullname) override { fullname_ = fullname; }

 protected:
  virtual bool IsNeedRecreate(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  virtual acme::AcmeOpPtr CreateKernel(acme::InputsImmutableInfoList inputs, acme::OutputsImmutableInfoList outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) {
    return nullptr;
  }
  acme::AcmeOpPtr acme_op_{nullptr};
  std::vector<size_t> acme_to_ms_input_indices_mapper_;
  std::vector<size_t> acme_to_ms_output_indices_mapper_;
  acme::ShapeInfoList acme_inputs_shape_;
  acme::ShapeInfoList acme_outputs_shape_;
  acme::InputsAddrList acme_inputs_addr_;
  acme::OutputsAddrList acme_outputs_addr_;
  acme::WsAddrList acme_wss_addr_;

 private:
  void GetOrGenerateTiling(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  inline void UpdateAddr(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                         const std::vector<KernelTensor *> &workspace);
  void GetAcmeKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  MemoryType host_tiling_mem_type_{kMemoryUndefined};
  MemoryType device_tiling_mem_type_{kMemoryUndefined};
  uint64_t last_key_{0};
  TilingCacheItemPtr last_item_{nullptr};
  std::vector<size_t> recreate_cared_indices_;
  std::string fullname_;
};

using AcmeKernelModPtr = std::shared_ptr<AcmeKernelMod>;
using AcmeKernelModPtrList = std::vector<AcmeKernelModPtr>;

#define MS_ACME_KERNEL_FACTORY_REG(NAME, DERIVE) MS_KERNEL_FACTORY_REG(AcmeKernelMod, NAME, DERIVE)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_KERNEL_MOD_H_
