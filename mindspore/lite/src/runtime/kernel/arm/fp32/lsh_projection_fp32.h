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

namespace mindspore::kernel {
class LshProjectionCPUKernel : public LiteKernel {
 public:
  LshProjectionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<LshProjectionParameter *>(op_parameter_);
  }
  ~LshProjectionCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  int MallocKeys();
  void FreeKeys();
  int GetSignBit(int32_t *feature_, float *weight_, float seed, LshProjectionParameter *para, char *hash_buff);
  void LshProjectionSparse(float *hash_seed_, int32_t *feature_, float *weight_, int32_t *output_,
                           LshProjectionParameter *param, int32_t start, int32_t end, char *hash_buff);
  void LshProjectionDense(float *hash_seed_, int32_t *feature_, float *weight_, int32_t *output_,
                          LshProjectionParameter *param, int32_t start, int32_t end, char *hash_buff);
  LshProjectionParameter *param_ = nullptr;
  float *hash_seed_ = nullptr;
  int32_t *feature_ = nullptr;
  float *weight_ = nullptr;
  int32_t *output_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LSH_PROJECTION_H_
