/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/cmsis-nn/int8/add_int8_coder.h"
#include <algorithm>
#include <limits>
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/nnacl/int8/add_int8_coder.h"
#include "coder/opcoders/serializers/serializer.h"
#include "coder/utils/common.h"
#include "mindspore/core/ops/array_ops.h"
#include "nnacl/arithmetic_parameter.h"
#include "nnacl/int8/quantize.h"

using mindspore::schema::PrimitiveType_AddFusion;

namespace mindspore::lite::micro::cmsis {
int AddInt8Coder::Prepare(CoderContext *const context) {
  MS_CHECK_GE(input_tensors_.size(), 2, RET_ERROR);
  input1_ = input_tensors_.at(0);
  input2 = input_tensors_.at(1);

  MS_CHECK_PTR(input1_);
  MS_CHECK_PTR(input2);

  MS_CHECK_TRUE(!input1_->quant_params().empty(), "input1_ quant_params is empty");
  MS_CHECK_TRUE(!input2->quant_params().empty(), "input2_ quant_params is empty");
  MS_CHECK_TRUE(!output_tensor_->quant_params().empty(), "output quant_params is empty");

  input_1_offset_ = -input1_->quant_params().at(0).zeroPoint;
  input_2_offset_ = -input2->quant_params().at(0).zeroPoint;
  out_offset_ = output_tensor_->quant_params().at(0).zeroPoint;
  const double input1_scale = input1_->quant_params().at(0).scale;
  const double input2_scale = input2->quant_params().at(0).scale;
  const double output_scale = output_tensor_->quant_params().at(0).scale;

  const double twice_max_input_scale = 2 * std::max(input1_scale, input2_scale);
  MS_CHECK_TRUE(twice_max_input_scale > 0, "twice_max_input_scale should larger than 0.");
  MS_CHECK_TRUE(output_scale > 0, "output_scale should larger than 0.");
  const double real_input1_multiplier = static_cast<double>(input1_scale) / twice_max_input_scale;
  const double real_input2_multiplier = static_cast<double>(input2_scale) / twice_max_input_scale;
  const double real_output_multiplier =
    twice_max_input_scale / ((1 << static_cast<size_t>(kLeftShift)) * static_cast<double>(output_scale));

  MS_CHECK_TRUE((real_input1_multiplier >= 0) && (real_input1_multiplier <= 1),
                "real_input1_multiplier should be in (0, 1)");
  QuantizeMultiplier(real_input1_multiplier, &input_1_mult_, &input_1_shift_);
  MS_CHECK_TRUE((real_input2_multiplier >= 0) && (real_input2_multiplier <= 1),
                "real_input2_multiplier should be in (0, 1)");
  QuantizeMultiplier(real_input2_multiplier, &input_2_mult_, &input_2_shift_);
  MS_CHECK_TRUE((real_output_multiplier >= 0) && (real_output_multiplier <= 1),
                "real_output_multiplier should be in (0, 1)");
  QuantizeMultiplier(real_output_multiplier, &out_mult_, &out_shift_);

  out_activation_min_ = std::numeric_limits<int8_t>::min();
  out_activation_max_ = std::numeric_limits<int8_t>::max();

  MS_CHECK_TRUE(input1_->ElementsNum() == input2->ElementsNum(), "tensor length not match");

  block_size_ = input1_->ElementsNum();

  return RET_OK;
}

int AddInt8Coder::DoCode(CoderContext *const context) {
  Serializer code;
  code.precision(kPrecision);

  Collect(context,
          {
            "CMSIS/NN/Include/arm_nnfunctions.h",
          },
          {
            "arm_elementwise_add_s8.c",
          });

  code.CodeFunction("arm_elementwise_add_s8", input1_, input2, input_1_offset_, input_1_mult_, input_1_shift_,
                    input_2_offset_, input_2_mult_, input_2_shift_, kLeftShift, output_tensor_, out_offset_, out_mult_,
                    out_shift_, out_activation_min_, out_activation_max_, block_size_);

  MS_LOG(INFO) << "AddInt8Coder has been called";
  context->AppendCode(code.str());
  return RET_OK;
}

std::unique_ptr<OperatorCoder> AddFusionInt8CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                         const std::vector<Tensor *> &out_tensors,
                                                         const LiteGraph::Node *node, size_t node_index, Target target,
                                                         int schema_version) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is null";
    return nullptr;
  }
  if (in_tensors.size() != kTwo) {
    MS_LOG(ERROR) << "in_tensors size error.";
    return nullptr;
  }
  std::unique_ptr<OperatorCoder> coder;
  if (in_tensors[0]->ElementsNum() == in_tensors[1]->ElementsNum()) {
    coder = std::make_unique<AddInt8Coder>(in_tensors, out_tensors, node, node_index, target);
  } else {
    coder =
      std::make_unique<mindspore::lite::micro::nnacl::AddInt8Coder>(in_tensors, out_tensors, node, node_index, target);
  }
  if (coder == nullptr) {
    return nullptr;
  }

  coder->SetSchemaVersion(schema_version);
  return coder;
}

REG_OPERATOR_CODER(kCortex_M, kNumberTypeInt8, PrimitiveType_AddFusion, AddFusionInt8CoderCreator)
}  // namespace mindspore::lite::micro::cmsis
