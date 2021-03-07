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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_DECONVOLUTION_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_DECONVOLUTION_INT8_CODER_H_

#include <vector>
#include <string>
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "nnacl/matmul_parameter.h"

namespace mindspore::lite::micro::nnacl {
class DeconvolutionInt8Coder final : public Conv2DBaseCoder {
 public:
  DeconvolutionInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                         const Model::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~DeconvolutionInt8Coder() override { delete matmul_param_; }

  int DoCode(CoderContext *const context) override;
  int Prepare(CoderContext *const context) override;

 private:
  int Init(CoderContext *const context);
  int InitData(CoderContext *ctx);
  int InitParam();
  int InitBiasWeight(CoderContext *ctx);
  void CheckSupportOptimize();
  int InitRunBuf(CoderContext *ctx);

  int32_t *tmp_buffer_{nullptr};
  int tmp_buffer_size_{0};
  int32_t *tmp_output_{nullptr};
  int tmp_output_size_{0};
  int32_t *input_sum_{nullptr};
  int input_sum_size_{0};

  int8_t *input_ptr_{nullptr};
  int input_ptr_size_{0};
  int8_t *weight_ptr_{nullptr};
  int32_t *weight_sum_{nullptr};
  size_t thread_count_{1};
  int thread_stride_{0};
  int32_t *bias_data_{nullptr};
  std::string matmul_func_str_;
  MatMulParameter *matmul_param_{nullptr};
  bool support_optimize_{true};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_DECONV_INT8_CODER_H_
