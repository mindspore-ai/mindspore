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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_SCALE_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_SCALE_BASE_H_

#include <vector>
#include "src/runtime/lite_kernel.h"
#include "nnacl/fp32/scale_fp32.h"

namespace mindspore::kernel {
class ScaleBaseCPUKernel : public LiteKernel {
 public:
  ScaleBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    scale_param_ = reinterpret_cast<ScaleParameter *>(op_parameter_);
  }
  ~ScaleBaseCPUKernel() override {
    if (in_tensors_.size() == kInputSize1 && offset_ != nullptr) {
      ms_context_->allocator->Free(offset_);
      offset_ = nullptr;
    }
  }

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  virtual int Compute(int task_id) = 0;
  static int CalculateParameter(const std::vector<lite::Tensor *> &inputs, ScaleParameter *scale_param);

 private:
  int InitOffset();
  int ComputeThreadCuttingInfo();
  static bool IsValidScale(const std::vector<int> &in_shape, const std::vector<int> &scale_shape,
                           const std::vector<int> &offset_shape, ScaleParameter *scale_param);

 protected:
  int data_type_size_{C4NUM};
  void *input_ptr_{nullptr};
  void *scale_{nullptr};
  void *offset_{nullptr};
  void *output_ptr_{nullptr};
  std::vector<int64_t> split_points_;
  ScaleParameter *scale_param_{nullptr};
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_SCALE_BASE_H_
