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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_CONCAT_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_CONCAT_BASE_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/concat_parameter.h"

namespace mindspore::kernel {
class ConcatBaseCPUKernel : public LiteKernel {
 public:
  ConcatBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    concat_param_ = reinterpret_cast<ConcatParameter *>(op_parameter_);
  }

  ~ConcatBaseCPUKernel() override = default;

  int Prepare() override;
  int ReSize() override;
  int DoConcat(int task_id);
  int Run() override;

 protected:
  int data_size_{4};
  int64_t outer_size_{0};
  uint8_t *output_{nullptr};
  std::vector<bool> is_with_data_;
  std::vector<const uint8_t *> inputs_ptr_;
  std::vector<int64_t> block_splits_;
  std::vector<int64_t> inner_sizes_;  // byte-inner-size (including axis) of each input and the last one is output's.

 private:
  // to balance each thread's load, we choose to chunk the output. the followed BlockInfo records the BlockBoundaryInfo
  // info of each block.
  struct BlockBoundaryInfo {
    int begin_input;      // input-index of upper boundary
    int end_input;        // input-index of lower boundary.
    int64_t begin_point;  // offset of begin-input.
    int64_t end_point;    // required size of end-input.
    BlockBoundaryInfo() : begin_input(0), end_input(0), begin_point(0), end_point(0) {}
  };
  int InitDynamicStatus();
  int ChooseThreadCuttingStrategy();
  ConcatParameter *concat_param_ = nullptr;
  std::vector<BlockBoundaryInfo> block_boundary_infos_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_CONCAT_BASE_H_
