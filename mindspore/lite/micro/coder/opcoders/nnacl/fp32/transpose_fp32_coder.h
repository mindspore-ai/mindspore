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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_TRANSPOSE_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_TRANSPOSE_FP32_CODER_H_
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/transpose.h"
namespace mindspore::lite::micro::nnacl {

class TransposeFp32Coder final : public OperatorCoder {
 public:
  TransposeFp32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                     const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~TransposeFp32Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  int Resize();

  int Init();

 private:
  TransposeParameter *transpose_parameter_ = nullptr;
  int thread_num_{1};
  int thread_h_stride_{0};
  int thread_h_num_{0};
  int num_unit_{0};
  int *in_shape_{nullptr};
  int *out_shape_{nullptr};
  int *dim_size_{nullptr};
  int *position_{nullptr};
};

}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_FP32_TRANSPOSE_FP32_CODER_H_
