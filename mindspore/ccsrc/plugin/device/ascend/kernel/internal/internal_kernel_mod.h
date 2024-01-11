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
#ifndef MS_KERNEL_INTERNAL_KERNEL_MOD_H_
#define MS_KERNEL_INTERNAL_KERNEL_MOD_H_
#include <memory>
#include <unordered_map>
#include <vector>
#include "ccsrc/kernel/kernel.h"
#include "ms_kernels_internal/internal_kernel.h"
#include "ms_kernel_internals/include/types.h"
#include "ms_kernel_internals/include/op_param.h"
namespace mindspore {
namespace kernel {
class InternalKernelMod : public KernelMod {
 public:
  InternalKernelMod() = default;
  virtual ~InternalKernelMod();

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
              const Address &tilingBuf, const std::vector<Address> &workspace, void *stream_ptr) override;
  virtual size_t GetTilingBufSize() { return impl_->GetTilingBufSize(); }
  virtual int Tiling(const Address &tilingBuf) { return impl_->Tiling(tilingBuf); }

 protected:
  virtual int Build(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  virtual void SetInOutIdx() = 0;
  virtual internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) = 0;
  std::shared_ptr<internal::InternelKernelImpl> impl_;
  std::unordered_map<size_t, size_t> inputsIdxMap_;
  std::unordered_map<size_t, size_t> outputsIdxMap_;
  std::vector<internal::Tensor *> inputs_;
  std::vector<internal::Tensor *> outputs_;
};
}  // namespace kernel
}  // namespace mindspore
#endif
