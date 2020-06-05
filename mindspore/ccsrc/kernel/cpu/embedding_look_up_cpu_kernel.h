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
#ifndef MINDSPORE_CCSRC_KERNEL_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "kernel/cpu/cpu_kernel.h"
#include "kernel/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class EmbeddingLookUpCPUKernel : public CPUKernel {
 public:
  EmbeddingLookUpCPUKernel() {
    axis_ = 0;
    offset_ = 0;
    split_num_ = 0;
    input_lens_ = 0;
    indices_lens_ = 0;
    gatherv2_out_lens_ = 0;
    reduce_scatter_flag_ = false;
    gather_v2_out_ = nullptr;
  }
  ~EmbeddingLookUpCPUKernel() override {
    if (gather_v2_out_ != nullptr) {
      free(gather_v2_out_);
      gather_v2_out_ = nullptr;
    }
  }

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void LookUpTable(const std::vector<kernel::AddressPtr> &inputs, size_t dim0, size_t dim1, size_t dim2,
                   float **output_addr);
  void CheckParam(const CNodePtr &kernel_node);
  std::vector<size_t> input_shape_;
  std::vector<size_t> indices_shape_;
  std::vector<size_t> output_shape_;
  int axis_;
  int offset_;
  int split_num_;
  size_t input_lens_;
  size_t indices_lens_;
  size_t gatherv2_out_lens_;
  bool reduce_scatter_flag_;

  void *gather_v2_out_;
};

MS_REG_CPU_KERNEL(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  EmbeddingLookUpCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_
