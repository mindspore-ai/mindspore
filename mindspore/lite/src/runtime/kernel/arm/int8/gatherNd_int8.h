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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GATHERND_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GATHERND_INT8_H_

#include <vector>
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class GatherNdInt8CPUKernel : public LiteKernel {
 public:
  GatherNdInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                        const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {}
  ~GatherNdInt8CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoGatherNd(int task_id);

 private:
  void InitOffset();
  int thread_count_;
  int thread_sz_count_;
  int thread_sz_stride_;
  int count_;
  int area_;
  int *in_offset_ = nullptr;
  int8_t *in_ptr_;
  int8_t *out_ptr_;
  GatherQuantArg param_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_GATHERND_INT8_H_
