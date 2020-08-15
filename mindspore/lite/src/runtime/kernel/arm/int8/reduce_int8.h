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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_REDUCE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_REDUCE_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/reduce_parameter.h"
#include "nnacl/int8/reduce_int8.h"
#include "nnacl/quantization/quantize.h"
#include "ir/anf.h"
#include "src/runtime/kernel/arm/base/reduce_base.h"

using mindspore::schema::ReduceMode;

namespace mindspore::kernel {
class ReduceInt8CPUKernel : public ReduceBaseCPUKernel {
  typedef int (*Reducer)(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                         int32_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num);
  typedef int (*LastReducer)(const int outer_size, const int inner_size, const int axis_size, const int32_t *src_data,
                             int8_t *dst_data, const ReduceQuantArg *quant, const int tid, const int thread_num);

 public:
  ReduceInt8CPUKernel(OpParameter *param, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                      const lite::Primitive *primitive)
      : ReduceBaseCPUKernel(param, inputs, outputs, ctx, primitive) {}
  ~ReduceInt8CPUKernel() {
    for (auto i = 0; i < data_buffers_.size(); i++) {
      int32_t *buffer = data_buffers_[i];
      if (buffer != nullptr) {
        free(buffer);
        buffer = nullptr;
      }
    }
    for (auto qm : mean_multipliers_) {
      delete qm;
      qm = nullptr;
    }
    for (auto qm : prod_multipliers_) {
      delete qm;
      qm = nullptr;
    }
    for (auto qm : sum_square_multipliers_) {
      delete qm;
      qm = nullptr;
    }
    src_data_ = nullptr;
    dst_data_ = nullptr;
  }

  int Init() override;
  int ReSize() override { return 0; };
  int Run() override;
  int CallReduceUnit(int task_id);
  int ReduceLastAxis(int task_id);

 public:
  bool is_last_axis_ = true;

 private:
  int MallocTmpBuffer();
  int CalculateQuantArgs();

 private:
  ReduceParameter *param_ = nullptr;
  ReduceQuantArg quant_arg_;

 private:
  int32_t *begin_src_data_ = nullptr;
  int8_t *last_dst_data_ = nullptr;
  std::vector<int32_t *> data_buffers_;
  const int32_t *src_data_ = nullptr;
  int32_t *dst_data_ = nullptr;

  Reducer reducer_ = nullptr;
  LastReducer last_reducer_ = nullptr;
  std::vector<QuantMulArg *> mean_multipliers_;
  std::vector<QuantMulArg *> prod_multipliers_;
  std::vector<QuantMulArg *> sum_square_multipliers_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_REDUCE_INT8_H_
