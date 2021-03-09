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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_INT8_CODER_H_
#include <string>
#include <memory>
#include <vector>
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "nnacl/conv_parameter.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

namespace mindspore::lite::micro::nnacl {
class Conv2DINT8Coder final : public Conv2DBaseCoder {
 public:
  explicit Conv2DINT8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                           const Model::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

  ~Conv2DINT8Coder() override {
    packed_weight_ = nullptr;
    bias_data_ = nullptr;
    filter_zp_ptr_ = nullptr;
    matmul_packed_input_ = nullptr;
    packed_input_ = nullptr;
    input_sum_ = nullptr;
  }

 private:
  void CheckSupportOptimize();
  int InitWeightBias(CoderContext *ctx);
  int InitTmpBuffer(CoderContext *ctx);

  int Resize();

  int8_t *packed_weight_{nullptr};
  int32_t *bias_data_{nullptr};
  int32_t *filter_zp_ptr_{nullptr};

  int thread_count_{1};
  int tile_num_{0};

  bool support_optimize_{true};
  bool filter_peroc_{false};

  size_t packed_input_size_{0};
  size_t input_sum_size_{0};
  size_t matmul_packed_input_size_{0};

  int8_t *packed_input_{nullptr};
  int32_t *input_sum_{nullptr};
  int8_t *matmul_packed_input_{nullptr};

  std::string matmul_func_{"NULL"};

  std::function<int(nnacl::NNaclInt8Serializer &, const std::string &, const std::string &)> pack_weight_init_{nullptr};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_INT8_CODER_H_
