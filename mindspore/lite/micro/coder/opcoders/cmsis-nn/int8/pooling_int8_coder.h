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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_CMSIS_NN_POOLING_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_CMSIS_NN_POOLING_INT8_CODER_H_

#include <string>
#include <memory>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/int8/pooling_int8.h"

namespace mindspore::lite::micro::cmsis {

class PoolingInt8Coder final : public OperatorCoder {
 public:
  PoolingInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                   const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}
  ~PoolingInt8Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int SetParameters();

  int dim_src_height_{0};
  int dim_src_width_{0};
  int dim_dst_height_{0};
  int dim_dst_width_{0};
  int stride_height_{0};
  int stride_width_{0};
  int dim_kernel_height_{0};
  int dim_kernel_width_{0};
  int padding_height_{0};
  int padding_width_{0};
  int act_min_{0};
  int act_max_{0};
  int ch_src_{0};

  int32_t *buffer_{nullptr};
  size_t buffer_size_{0};
  PoolingParameter *pooling_parameter_{nullptr};
};

}  // namespace mindspore::lite::micro::cmsis

#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_CMSIS_NN_POOLING_INT8_CODER_H_
