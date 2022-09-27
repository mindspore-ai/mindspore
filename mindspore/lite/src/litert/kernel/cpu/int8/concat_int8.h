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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_CONCAT_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_CONCAT_INT8_H_

#include <vector>
#include <limits>
#include "nnacl/int8/concat_int8.h"
#include "include/errorcode.h"
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class ConcatInt8CPUKernel : public LiteKernel {
 public:
  ConcatInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const mindspore::lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    concat_param_ = reinterpret_cast<ConcatParameter *>(op_parameter_);
  }
  ~ConcatInt8CPUKernel() override {
    if (input_data_ != nullptr) {
      free(input_data_);
    }
    int *output_shape = concat_param_->output_shapes_;
    if (output_shape != nullptr) {
      free(output_shape);
    }
    if (concat_param_->input_shapes_ != nullptr) {
      for (std::size_t i = 0; i < in_tensors().size(); i++) {
        int *input_shape = concat_param_->input_shapes_[i];
        if (input_shape != nullptr) {
          free(input_shape);
        }
      }
      free(concat_param_->input_shapes_);
    }
    if (concat_param_->quant_arg_.in_args_ != nullptr) {
      free(concat_param_->quant_arg_.in_args_);
    }
  }

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  void DoExecute(int task_id);

 private:
  int64_t before_axis_size = 0;
  int64_t count_unit_ = 0;
  int8_t **input_data_ = nullptr;  // freed in ~ConcatInt8CPUKernel
  int8_t *output_data_ = nullptr;
  ConcatParameter *concat_param_ = nullptr;
};

int ConcatInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_CONCAT_INT8_H_
