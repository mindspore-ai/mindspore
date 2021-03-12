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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REDUCE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REDUCE_H_

#include <vector>
#include "src/lite_kernel.h"

#include "nnacl/fp32/reduce_fp32.h"
#include "src/runtime/kernel/arm/base/reduce_base.h"

using mindspore::schema::ReduceMode;

namespace mindspore::kernel {
typedef int (*Reducer)(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
                       float *dst_data, const int tid, const int thread_num);
typedef int (*IntReducer)(const int outer_size, const int inner_size, const int axis_size, const int *src_data,
                          int *dst_data, const int tid, const int thread_num);
typedef int (*BoolReducer)(const int outer_size, const int inner_size, const int axis_size, const bool *src_data,
                           bool *dst_data, const int tid, const int thread_num);
struct ReduceKernelList {
  int type_;
  Reducer float_func_;
  IntReducer int_func_;
  BoolReducer bool_func_;
};

class ReduceCPUKernel : public ReduceBaseCPUKernel {
 public:
  ReduceCPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ReduceBaseCPUKernel(param, inputs, outputs, ctx) {
    reduce_param_ = reinterpret_cast<ReduceParameter *>(param);
  }
  ~ReduceCPUKernel() override {
    src_data_ = nullptr;
    dst_data_ = nullptr;
    reducer_ = nullptr;
    int_reducer_ = nullptr;
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int CallReduceUnit(int task_id);

 protected:
  void InitialKernelList();

 private:
  ReduceParameter *reduce_param_;
  Reducer reducer_ = nullptr;
  BoolReducer bool_reducer_ = nullptr;
  IntReducer int_reducer_ = nullptr;
  std::vector<void *> data_buffers_;
  LiteDataType data_type_;

  const void *src_data_ = nullptr;
  void *dst_data_ = nullptr;

 private:
  int MallocTmpBuffer();
  void FreeTmpBuffer();
  int CalculateCoeffOutput();
  void HandleASumAndSumSquare();
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REDUCE_H_
