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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSH_PROJECTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSH_PROJECTION_H_

#include <vector>

#include "nnacl/lsh_projection_parameter.h"
#include "src/lite_kernel.h"
#include "schema/model_generated.h"

namespace mindspore::kernel {
class LshProjectionCPUKernel : public LiteKernel {
 public:
  LshProjectionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                         const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive), thread_num_(ctx->thread_num_) {
    lsh_param_ = reinterpret_cast<LshProjectionParameter *>(op_parameter_);
  }
  ~LshProjectionCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);
  int GetSignBit(char *in_data, float *weight, float seed, LshProjectionParameter *para);
  void LshProjectionSparse(float *hash, char *in_data, float *weight, int32_t *output, LshProjectionParameter *param);
  void LshProjectionDense(float *hash, char *in_data, float *weight, int32_t *output, LshProjectionParameter *param);

 private:
  LshProjectionParameter *lsh_param_ = nullptr;
  const lite::InnerContext *ctx_;
  int thread_num_;
  int64_t elements_num_;
  int64_t count_unit_;
  float *hash = nullptr;
  char *in_data = nullptr;
  float *weight = nullptr;
  int32_t *output = nullptr;
};

int LshProjectionRun(void *cdata, int task_id);

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSH_PROJECTION_H_
