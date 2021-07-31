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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_TENSOR_COPY_SLICES_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_TENSOR_COPY_SLICES_H

#include <vector>
#include <memory>
#include "backend/kernel_compiler/rts/rt_kernel.h"
#include "backend/kernel_compiler/rts/rt_kernel_info.h"

namespace mindspore {
namespace kernel {
class TensorCopySlices : public RtKernel {
 public:
  TensorCopySlices();
  ~TensorCopySlices() override;

  bool Init(const AnfNodePtr &anf_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;

 private:
  void GetInputOutputInfo(const AnfNodePtr &anf_node);
  void GetInputOutputTotalCount(const AnfNodePtr &anf_node);
  void *VoidPointerOffset(void *ptr, size_t offset) const;

  std::vector<int64_t> input_shape_;
  std::vector<int64_t> update_shape_;
  std::vector<int64_t> output_shape_;
  TypeId input_type_id_{};
  TypeId output_type_id_{};
  TypeId update_type_id_{};

  size_t offset_{0};
  size_t copy_size_{0};
};

class TensorCopySlicesDesc : public RtKerDesc {
 public:
  TensorCopySlicesDesc();
  ~TensorCopySlicesDesc() override;
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> GetKernelInfo() override;
};

MS_REG_RTKERNEL_DESC(tensorcopyslices, TensorCopySlicesDesc);
MS_REG_RTKERNEL(tensorcopyslices, TensorCopySlices);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_TENSOR_COPY_SLICES_H
