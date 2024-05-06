/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_GROUPED_MATMUL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_GROUPED_MATMUL_H_

#include <vector>

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"

namespace mindspore {
namespace kernel {
class InternalGroupedMatmul : public InternalKernelMod {
 public:
  InternalGroupedMatmul() : InternalKernelMod("GroupedMatmul") {}
  virtual ~InternalGroupedMatmul() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 protected:
  int Build(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs);

  void SetInOutIdx() override;
  uint64_t GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                             const std::vector<KernelTensor *> &outputs) override;

  void SetTilingInfo(const uint64_t key) override;

  int PrepareInOutTensors(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr);

 private:
  // shape[0] will split into groups according to group_list.
  int CheckIntegratedInputShape(const std::vector<int64_t> &shape, const std::vector<int64_t> &group_list);
  // integrated weight shape should be (b, k, n), b should be same as group_list size.
  int CheckIntegratedWeightShape(const std::vector<int64_t> &shape, const std::vector<int64_t> &group_list);

  void SetHostList(const KernelTensor *kernel_tensor, const std::vector<int64_t> &group_num, const size_t &list_offset,
                   std::vector<void *> &host_list_ptr);

 private:
  size_t real_input_num_ = 0;
  int split_item_ = 0;
  std::vector<int64_t> dyn_input_sizes_{};
  std::vector<int64_t> group_num_{};
  std::vector<void *> input_host_list_ptr_{};
  std::vector<void *> output_host_list_ptr_{};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_GROUPED_MATMUL_H_
