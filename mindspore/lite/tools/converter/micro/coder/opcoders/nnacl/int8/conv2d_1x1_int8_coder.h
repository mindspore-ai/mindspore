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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_Conv2D_1X1_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_Conv2D_1X1_INT8_CODER_H_
#include "coder/opcoders/base/conv2d_base_coder.h"
#include <memory>
#include <string>
#include <vector>
#include "nnacl/conv_parameter.h"

namespace mindspore::lite::micro::nnacl {
class Conv2D1x1Int8Coder final : public Conv2DBaseCoder {
 public:
  Conv2D1x1Int8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                     const Model::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  ~Conv2D1x1Int8Coder() override = default;

 private:
  void CheckSupportOptimize();

  int InitWeightBias(CoderContext *const context);

  int InitFilterPeroc();

  int InitParam();

  int InitRunBuf();

  int32_t *input_sum_{nullptr};     /* per-oc */
  int32_t *filter_zp_ptr_{nullptr}; /* per-oc up round  */
  int32_t *left_shift_{nullptr};    /* per-oc up round  */
  int32_t *right_shift_{nullptr};   /* per-oc up round  */
  int32_t *multiplier_{nullptr};    /* per-oc up round  */
  int8_t *packed_weight_{nullptr};
  int32_t *bias_data_{nullptr};
  int8_t *packed_input_{nullptr};
  int8_t *input_ptr_{nullptr};
  int8_t *output_ptr_{nullptr};
  size_t input_sum_size_{0};
  MatMulParameter *matmul_param_{nullptr};
  std::string matmul_func_;
  bool pre_trans_input_{false};
  bool support_optimize_{false};
  bool filter_peroc_{false};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_Conv2D_1X1_INT8_CODER_H_
