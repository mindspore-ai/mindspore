/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_1X1_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_1X1_FP16_CODER_H_

#include <vector>
#include <string>
#include "nnacl/conv_parameter.h"
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "base/float16.h"
#include "wrapper/base/micro_parameter.h"

namespace mindspore::lite::micro::nnacl {
class Convolution1x1FP16Coder final : public Conv2DBaseCoder {
 public:
  Convolution1x1FP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                          const LiteGraph::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {
    data_type_ = kNumberTypeFloat16;
  }

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  ~Convolution1x1FP16Coder() override;

 private:
  void CollectFilesForFunc(CoderContext *const context);
  int InitWeightBias(CoderContext *const context);
  int InitMatmulParam();
  void FreeTmpBuffer();
  MicroMatmulParameter *matmul_param_{nullptr};
  int row_tile_{C12NUM};
  int col_tile_{C8NUM};
  void *packed_weight_{nullptr};
  void *bias_data_{nullptr};
  void *pack_input_{nullptr};
  void *tmp_input_{nullptr};
  size_t pack_weight_size_{0};
  size_t bias_data_size_{0};
  size_t tmp_input_size_{0};
  size_t pack_input_size_{0};
  bool pre_trans_input_{false};
  std::string output_ptr_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_CONVOLUTION_1X1_FP16_CODER_H_
