/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_LAYER_NORM_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_LAYER_NORM_FP16_H_
#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/layer_norm_parameter.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class LayerNormFp16CPUKernel : public LiteKernel {
 public:
  LayerNormFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<LayerNormParameter *>(parameter);
  }
  ~LayerNormFp16CPUKernel() override{};

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoLayerNormFp16(int thread_id);

 private:
  LayerNormParameter *param_ = nullptr;
  float16_t *src_data_ = nullptr;
  float16_t *dst_data_ = nullptr;
  float16_t *gamma_data_ = nullptr;
  float16_t *beta_data_ = nullptr;
  float16_t *mean_data_ = nullptr;
  float16_t *var_data_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_LAYER_NORM_FP16_H_
