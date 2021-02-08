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

#include "coder/opcoders/cmsis-nn/int8/fullconnection_int8_coder.h"
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::lite::micro::cmsis {

int FullConnectionInt8Coder::Prepare(CoderContext *const context) {
  FullConnectionBaseCoder::Init();
  ConfigInputOutput();
  MS_CHECK_RET_CODE(SetParameters(), "SetParameters failed");
  return RET_OK;
}

void FullConnectionInt8Coder::ConfigInputOutput() { output_tensor_->set_format(schema::Format_NHWC); }

int FullConnectionInt8Coder::DoCode(CoderContext *const context) {
  Serializer code;
  code.precision(kPrecision);

  Collect(context, {"CMSIS/NN/Include/arm_nnfunctions.h"}, {"arm_fully_connected_s8.c", "arm_nn_vec_mat_mult_t_s8.c"});

  code.CodeFunction("arm_fully_connected_s8", input_tensor_, filter_tensor_, col_dim_, row_dim_, nb_batches_,
                    input_offset_, filter_offset_, out_multiplier_, out_shift_, output_offset_, bias_tensor_,
                    output_tensor_, output_activation_min_, output_activation_max_, "NULL");
  context->AppendCode(code.str());
  return RET_OK;
}

int FullConnectionInt8Coder::SetParameters() {
  MS_CHECK_TRUE(output_tensor_->shape().size() == 2, "output tensor size should be 2");
  MS_CHECK_TRUE(!input_tensor_->quant_params().empty(), "input quant_params is empty");
  MS_CHECK_TRUE(!filter_tensor_->quant_params().empty(), "filter quant_params is empty");
  MS_CHECK_TRUE(!output_tensor_->quant_params().empty(), "output quant_params is empty");
  QuantArg input_quant_arg = input_tensor_->quant_params().at(0);
  QuantArg filter_quant_arg = filter_tensor_->quant_params().at(0);
  QuantArg output_quant_arg = output_tensor_->quant_params().at(0);

  double real_multiplier = input_quant_arg.scale * filter_quant_arg.scale / output_quant_arg.scale;
  QuantizeMultiplier(real_multiplier, &out_multiplier_, &out_shift_);
  CalculateActivationRangeQuantized(fc_param_->act_type_ == ActType_Relu, fc_param_->act_type_ == ActType_Relu6,
                                    output_quant_arg.zeroPoint, output_quant_arg.scale, &output_activation_min_,
                                    &output_activation_max_);

  input_offset_ = -input_quant_arg.zeroPoint;
  filter_offset_ = -filter_quant_arg.zeroPoint;
  output_offset_ = output_quant_arg.zeroPoint;

  col_dim_ = filter_tensor_->DimensionSize(filter_tensor_->shape().size() - 1);
  row_dim_ = output_tensor_->DimensionSize(1);
  nb_batches_ = input_tensor_->Batch();
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32M, kNumberTypeInt8, PrimitiveType_FullConnection, CPUOpCoderCreator<FullConnectionInt8Coder>)
}  // namespace mindspore::lite::micro::cmsis
