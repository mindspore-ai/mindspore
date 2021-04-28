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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_RESIZE_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_RESIZE_FP32_CODER_H_

#include "coder/opcoders/base/resize_base_coder.h"
#include <vector>
#include <algorithm>
#include <string>
#include "include/errorcode.h"
#include "nnacl/fp32/resize_fp32.h"
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/fp32/resize_fp32.h"

namespace mindspore::lite::micro::nnacl {

class ResizeFP32Coder final : public ResizeBaseCoder {
 public:
  ResizeFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : ResizeBaseCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~ResizeFP32Coder() override { FreeTmpBuffer(); };
  int Prepare(CoderContext *const context) override;
  int ReSize();
  int DoCode(CoderContext *const context) override;

 private:
  int SelectCalculatorFunc();
  void CalTmpBufferLen();
  int MallocTmpBuffer();
  void FreeTmpBuffer();
  int ResizePrepare();

  ResizeCoordinate coordinate_;
  size_t x_len_{0};
  size_t y_len_{0};
  size_t x_weight_len_{0};
  size_t y_weight_len_{0};

  float *y_weights_{nullptr};
  float *x_weights_{nullptr};
  float *line_buffer_{nullptr};
  CalculateOriginalCoordinate calculate_{nullptr};
  std::string calculate_str_;
};

}  // namespace mindspore::lite::micro::nnacl

#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_RESIZE_FP32_CODER_H_
