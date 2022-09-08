/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_GATHER_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_GATHER_BASE_H_

#include <vector>
#include "include/errorcode.h"
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class GatherBaseCPUKernel : public LiteKernel {
 public:
  GatherBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~GatherBaseCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoGather(int task_id) const;

 protected:
  int UpdateThreadNumProcess(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num,
                             int64_t unit_num) override;
  virtual int AssignIndicesData(bool isIndicesInt32) = 0;
  int *indices_data_{nullptr};

 private:
  int InitDynamicStatus();
  int ChooseThreadCuttingStrategy();
  struct BlockBoundaryInfo {
    int64_t begin_batch;
    int64_t begin_index;
    int64_t end_batch;
    int64_t end_index;
  };
  int axis_ = 0;
  int64_t outer_size_{0};
  int64_t indices_size_{0};
  int64_t byte_inner_size_{0};
  int64_t limit_{0};
  std::vector<BlockBoundaryInfo> block_boundary_infos_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_GATHER_BASE_H_
