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

#include "src/runtime/kernel/arm/nnacl/fp32/reduce.h"
#include "ir/anf.h"
using mindspore::schema::ReduceMode;

namespace mindspore::kernel {
class ReduceCPUKernel : public LiteKernel {
  typedef int (*Reducer)(const int outer_size, const int inner_size, const int axis_size, const float *src_data,
                         const int *src_shape, float *dst_data, const int tid, const int thread_num);

 public:
  ReduceCPUKernel(ReduceParameter *param, const std::vector<lite::tensor::Tensor *> &inputs,
                  const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                  const lite::Primitive *primitive)
      : LiteKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, ctx, primitive),
        context_(ctx),
        keep_dims_(param->keep_dims_),
        num_axes_(param->num_axes_),
        mode_(param->mode_) {
    memcpy(axes_, param->axes_, sizeof(param->axes_));
  }
  ~ReduceCPUKernel() {
    for (auto i = 0; i < data_buffers_.size(); i++) {
      float *buffer = data_buffers_[i];
      if (buffer != nullptr) {
        free(buffer);
        buffer = nullptr;
      }
    }
    src_data_ = nullptr;
    dst_data_ = nullptr;
  }

  int Init() override;
  int ReSize() override { return 0; };
  int Run() override;
  int CallReduceUnit(int task_id);

 private:
  int CheckInputsOutputs();
  int CheckParameters();
  int MallocTmpBuffer();

 private:
  const lite::Context *context_ = nullptr;
  bool keep_dims_;
  int axes_[REDUCE_MAX_AXES_NUM];
  int num_axes_;
  int mode_;

 private:
  std::vector<float *> data_buffers_;
  int outer_size_;
  int inner_size_;
  int axis_size_;
  std::vector<int> tmp_shape_;
  const float *src_data_;
  float *dst_data_;
  Reducer reducer_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REDUCE_H_
