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
#ifndef MS_KERNELS_KERNEL_MOD_H_
#define MS_KERNELS_KERNEL_MOD_H_
#include "kernel_tensor.h"
#include <memory>
#include <unordered_map>
#include <vector>
#ifdef ENBALE_INTERNAL_KERNEL
#include "kernel.h"
#endif
#include "internal_kernel.h"
namespace mindspore {
#ifdef ENBALE_INTERNAL_KERNEL
class InternalKernelMod : public KernelMod {
#else
class InternalKernelMod {
#endif
public:
  InternalKernelMod() = default;
  virtual ~InternalKernelMod() = default;

  virtual bool Init(const std::vector<KernelTensor *> &inputs,
                    const std::vector<KernelTensor *> &outputs) = 0;
  virtual int Resize(const std::vector<KernelTensor *> &inputs,
                     const std::vector<KernelTensor *> &outputs) = 0;
  virtual bool Launch(const std::vector<KernelTensor *> &inputs,
                      const std::vector<KernelTensor *> &outputs,
                      const Address &tilingBuf,
                      const std::vector<Address> &workspace,
                      void *stream_ptr) = 0;
  size_t GetTilingBufSize() { return impl_->GetTilingBufSize(); }
  int Tiling(Address &tilingBuf) { return impl_->Tiling(tilingBuf); }

protected:
  virtual int Build(const std::vector<KernelTensor *> &inputs,
                    const std::vector<KernelTensor *> &outputs) = 0;
  virtual void SetInOutIdx() = 0;
  std::shared_ptr<internal::InternelKernelImpl> impl_;
  std::unordered_map<size_t, size_t> inputsIdxMap;
  std::unordered_map<size_t, size_t> outputsIdxMap;
};
} // namespace mindspore
#endif
