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

#ifndef MINDSPORE_LITE_MICRO_CODER_CMSIS_NN_DWCONV_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_CMSIS_NN_DWCONV_INT8_CODER_H_

#include <vector>
#include "coder/opcoders/cmsis-nn/int8/conv2d_base_coder.h"
#include "src/runtime/kernel/arm/int8/convolution_depthwise_int8.h"

namespace mindspore::lite::micro::cmsis {
class DWConvInt8Coder final : public Conv2DBaseCoder {
 public:
  DWConvInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~DWConvInt8Coder() override = default;

  int Prepare(CoderContext *context) override;

  int DoCode(CoderContext *context) override;

 private:
  enum DwConvOpt {
    Basic = 0,
    Conv_3x3 = 1,
    Conv_opt = 2,
  };

  int SetParameters();

  void CheckSupportOptimize();

  int InitTmpBuffer();

  int InitWeightBias();

  int32_t input_x_{0};
  int32_t input_y_{0};
  int32_t input_ch_{0};
  int32_t output_ch_{0};
  int32_t ch_mult_{0};
  int32_t kernel_x_{0};
  int32_t kernel_y_{0};
  int32_t pad_x_{0};
  int32_t pad_y_{0};
  int32_t stride_x_{0};
  int32_t stride_y_{0};
  int32_t output_x_{0};
  int32_t output_y_{0};
  int32_t output_offset_{0};
  int32_t input_offset_{0};
  int32_t output_activation_min_{0};
  int32_t output_activation_max_{0};
  uint16_t dilation_x_{0};
  uint16_t dilation_y_{0};

  int8_t *packed_weight_{nullptr};
  DwConvOpt optimize_{Basic};
  size_t buffer_size_{0};
  int16_t *buffer{nullptr};
};
}  // namespace mindspore::lite::micro::cmsis

#endif  // MINDSPORE_LITE_MICRO_CODER_CMSIS_NN_DWCONV_INT8_CODER_H_
