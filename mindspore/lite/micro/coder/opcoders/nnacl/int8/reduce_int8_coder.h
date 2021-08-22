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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_INT8_REDUCE_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_INT8_REDUCE_INT8_CODER_H_

#include <string>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/int8/reduce_int8.h"
#include "coder/opcoders/base/reduce_base_coder.h"
namespace mindspore::lite::micro::nnacl {
class ReduceInt8Coder final : public ReduceBaseCoder {
 public:
  ReduceInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : ReduceBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ReduceInt8Coder() override { begin_src_data_ = nullptr; }

  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 private:
  int MallocTmpBuffer();
  int CalculateQuantArgs();
  void GetQuantArgs(size_t index);

 private:
  ReduceQuantArg quant_arg_{0};
  int32_t *begin_src_data_{nullptr};
  std::vector<int32_t *> data_buffers_;
  bool valid_shape_{false};
  bool is_last_axis{false};
  std::string reducer_;
  std::string last_reducer_;
  std::vector<QuantMulArg *> mean_multipliers_;
  std::vector<QuantMulArg *> prod_multipliers_;
  std::vector<QuantMulArg *> sum_square_multipliers_;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_INT8_REDUCE_INT8_CODER_H_
