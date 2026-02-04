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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_INPLACE_GROUPED_MATMUL_ADD_ATB_KERNEL_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_INPLACE_GROUPED_MATMUL_ADD_ATB_KERNEL_H_

#include <string>
#include <vector>
#include <utility>

#include "acl/acl.h"
#include "atb/atb_infer.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/atb/kernel_mod_impl/atb_kernel_mod.h"

namespace mindspore {
namespace kernel {
class InplaceGroupedMatmulAddATBKernelMod : public ATBKernelMod {
 public:
  InplaceGroupedMatmulAddATBKernelMod() : ATBKernelMod(std::move("inplace_grouped_matmul_add")) {}
  ~InplaceGroupedMatmulAddATBKernelMod() = default;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  TensorStorageInfoPtr CreateTensorStorageInfo(const KernelTensor *ori_tensor, const std::vector<int64_t> &new_shape);
  void SetTensorStorageInfo(const KernelTensorPtr &new_tensor, const KernelTensor *ori_tensor);

  KernelTensorPtr out_tensor_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_INPLACE_GROUPED_MATMUL_ADD_ATB_KERNEL_H_
