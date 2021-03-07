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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DETECTION_POST_PROCESS_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DETECTION_POST_PROCESS_BASE_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "nnacl/fp32/detection_post_process_fp32.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class DetectionPostProcessBaseCPUKernel : public LiteKernel {
 public:
  DetectionPostProcessBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_num_(ctx->thread_num_) {
    params_ = reinterpret_cast<DetectionPostProcessParameter *>(parameter);
  }
  virtual ~DetectionPostProcessBaseCPUKernel();

  int Init() override;
  int ReSize() override;
  int Run() override;

  int thread_num_ = 1;
  int num_boxes_ = 0;
  int num_classes_with_bg_ = 0;
  float *input_boxes_ = nullptr;
  float *input_scores_ = nullptr;
  DetectionPostProcessParameter *params_ = nullptr;

 protected:
  virtual int GetInputData() = 0;

 private:
  void FreeAllocatedBuffer();
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DETECTION_POST_PROCESS_BASE_H_
