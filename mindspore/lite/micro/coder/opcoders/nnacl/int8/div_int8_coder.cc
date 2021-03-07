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

#include "coder/opcoders/nnacl/int8/div_int8_coder.h"
#include <algorithm>
#include <limits>
#include "include/errorcode.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"

namespace mindspore::lite::micro::nnacl {

int DivInt8Coder::Prepare(CoderContext *context) {
  input0 = input_tensors_.at(0);
  input1 = input_tensors_.at(1);
  MS_ASSERT(input0);
  MS_ASSERT(input1);

  broadcast_ = input0->ElementsNum() != input1->ElementsNum();

  param_.in0_args_.scale_ = input0->quant_params().front().scale;
  param_.in0_args_.zp_ = -input0->quant_params().front().zeroPoint;
  param_.in1_args_.scale_ = input1->quant_params().front().scale;
  param_.in1_args_.zp_ = -input1->quant_params().front().zeroPoint;
  param_.out_args_.scale_ = output_tensor_->quant_params().front().scale;
  param_.out_args_.zp_ = output_tensor_->quant_params().front().zeroPoint;

  const double real_multiplier = param_.in0_args_.scale_ / (param_.in1_args_.scale_ * param_.out_args_.scale_);

  QuantizeMultiplier(real_multiplier, &param_.output_multiplier_, &param_.output_shift_);

  param_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  param_.output_activation_max_ = std::numeric_limits<int8_t>::max();

  return RET_OK;
}

int DivInt8Coder::DoCode(CoderContext *const context) {
  NNaclInt8Serializer code;
  int element_num = output_tensor_->ElementsNum();
  code.CodeStruct("param", param_);
  if (broadcast_) {
    ArithmeticParameter tile_para;
    tile_para.ndim_ = output_tensor_->shape().size();
    for (size_t i = 0; i < tile_para.ndim_; i++) {
      tile_para.in_shape0_[i] = input0->DimensionSize(i);
      tile_para.in_shape1_[i] = input1->DimensionSize(i);
      tile_para.out_shape_[i] = output_tensor_->DimensionSize(i);
    }
    tile0_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    tile1_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    MS_CHECK_PTR(tile0_data_);
    MS_CHECK_PTR(tile1_data_);
    code.CodeStruct("tile_para", tile_para);
    code.CodeFunction("TileDimensionsInt8", input0, input1, tile0_data_, tile1_data_, "&tile_para");
    code.CodeFunction("DivInt8", tile0_data_, tile1_data_, output_tensor_, element_num, "&param");
  } else {
    code.CodeFunction("DivInt8", input0, input1, output_tensor_, element_num, "&param");
  }

  return RET_OK;
}

}  // namespace mindspore::lite::micro::nnacl
