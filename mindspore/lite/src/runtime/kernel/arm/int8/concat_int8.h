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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONCAT_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONCAT_INT8_H_

#include <vector>
#include <limits>
#include "nnacl/int8/concat_int8.h"
#include "include/errorcode.h"
#include "src/lite_kernel.h"
#include "include/context.h"
#include "src/runtime/runtime_api.h"

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
    for (std::size_t i = 0; i < in_tensors().size(); i++) {
      int *input_shape = concat_param_->input_shapes_[i];
      if (input_shape != nullptr) {
        free(input_shape);
      }
    }
    if (concat_param_->input_shapes_ != nullptr) {
      free(concat_param_->input_shapes_);
    }
    if (concat_param_->quant_arg_.in_args_ != nullptr) {
      free(concat_param_->quant_arg_.in_args_);
    }
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  int64_t before_axis_size;
  int64_t count_unit_;
  int8_t **input_data_ = nullptr;
  int8_t *output_data_ = nullptr;
  ConcatParameter *concat_param_ = nullptr;
};

int ConcatInt8Run(void *cdata, int task_id);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONCAT_INT8_H_
