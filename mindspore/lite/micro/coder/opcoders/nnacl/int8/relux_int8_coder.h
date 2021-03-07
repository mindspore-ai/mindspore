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
#ifndef MINDSPORE_LITE_MICRO_CODER_RELUX_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_RELUX_INT8_CODER_H_

#include <string>
#include <memory>
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/int8/relux_int8.h"
#include "coder/log.h"
#include "include/errorcode.h"

namespace mindspore::lite::micro::nnacl {

class ReluxInt8Coder : public OperatorCoder {
 public:
  ReluxInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                 const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ReluxInt8Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 protected:
  ReluXQuantArg quant_arg_;

 private:
  int type_;
};

class ReluInt8Coder final : public ReluxInt8Coder {
 public:
  ReluInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                const Model::Node *node, size_t node_index, Target target)
      : ReluxInt8Coder(in_tensors, out_tensors, node, node_index, target) {}

  ~ReluInt8Coder() override = default;

  int Prepare(CoderContext *const context) override {
    MS_CHECK_RET_CODE(ReluxInt8Coder::Prepare(context), "ReluxInt8Coder::Prepare failed");
    quant_arg_.quantized_output_min = quant_arg_.output_arg.zp_;
    quant_arg_.quantized_output_max = CHAR_MAX;
    return RET_OK;
  };
};

class Relu6Int8Coder final : public ReluxInt8Coder {
 public:
  Relu6Int8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                 const Model::Node *node, size_t node_index, Target target)
      : ReluxInt8Coder(in_tensors, out_tensors, node, node_index, target) {}

  ~Relu6Int8Coder() override = default;

  int Prepare(CoderContext *const context) override {
    MS_CHECK_RET_CODE(ReluxInt8Coder::Prepare(context), "ReluxInt8Coder::Prepare failed");
    quant_arg_.quantized_output_min = QuantizeToInt8(0, quant_arg_.output_arg.scale_, quant_arg_.output_arg.zp_);
    quant_arg_.quantized_output_max = QuantizeToInt8(6, quant_arg_.output_arg.scale_, quant_arg_.output_arg.zp_);
    return RET_OK;
  };
};

}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_RELUX_INT8_CODER_H_
