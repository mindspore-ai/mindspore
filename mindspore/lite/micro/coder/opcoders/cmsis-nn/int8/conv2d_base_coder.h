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

#ifndef MINDSPORE_LITE_MICRO_CODER_CMSIS_NN_CONV2D_CMSIS_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_CMSIS_NN_CONV2D_CMSIS_CODER_H_

#include <string>
#include <vector>
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite::micro::cmsis {

class Conv2DBaseCoder : public micro::Conv2DBaseCoder {
 public:
  explicit Conv2DBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                           const Model::Node *node, size_t node_index, Target target)
      : micro::Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~Conv2DBaseCoder() override {
    free(output_mult_);
    free(output_shift_);
  }

 protected:
  int SetQuantArgs();

  int32_t *output_mult_{nullptr};
  int32_t *output_shift_{nullptr};
};
}  // namespace mindspore::lite::micro::cmsis

#endif  // MINDSPORE_LITE_MICRO_CODER_CMSIS_NN_CONV2D_CMSIS_CODER_H_
