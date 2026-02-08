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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_DIST_COMM_REDUCE_SCATTER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_DIST_COMM_REDUCE_SCATTER_H_

#include <vector>
#include <memory>
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/hccl_kernel.h"
#include "kernel/ascend/hccl/hcom_util.h"

namespace mindspore {
namespace kernel {
class InnerCommReduceScatterKernel : public HcclKernel {
 public:
  InnerCommReduceScatterKernel() = default;
  ~InnerCommReduceScatterKernel() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  uint64_t GetReduceScatterVParam(const std::vector<KernelTensor *> &inputs, uint64_t input_list_num);

 protected:
  HcclDataType data_type_ = {};
  uint64_t hccl_count_;
  HcclComm comm_;
  int64_t rank_size_;
  HcclReduceOp op_type_enum_;
  uint64_t input_size_;
  uint32_t type_size_ = 4;
  bool same_shape_ = true;
  hccl::HcclReduceScatterVParams params_ = {};
  std::vector<uint64_t> send_size_byte_;
};
MS_HCCL_REG_KERNEL(DistCommReduceScatter, InnerCommReduceScatterKernel);

}  // namespace kernel
}  // namespace mindspore

#endif
