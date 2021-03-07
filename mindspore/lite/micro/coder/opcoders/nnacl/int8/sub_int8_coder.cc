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

#include "coder/opcoders/nnacl/int8/sub_int8_coder.h"
#include <algorithm>
#include <limits>
#include "include/errorcode.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::lite::micro::nnacl {

int SubInt8Coder::Prepare(CoderContext *const context) {
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

  const int left_shift = 20;
  const double twice_max_input_scale = 2 * std::max(param_.in0_args_.scale_, param_.in1_args_.scale_);
  const double real_input0_multiplier = param_.in0_args_.scale_ / twice_max_input_scale;
  const double real_input1_multiplier = param_.in1_args_.scale_ / twice_max_input_scale;
  const double real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * param_.out_args_.scale_);

  QuantizeMultiplierSmallerThanOne(real_input0_multiplier, &param_.input0_multiplier_, &param_.input0_shift_);
  QuantizeMultiplierSmallerThanOne(real_input1_multiplier, &param_.input1_multiplier_, &param_.input1_shift_);
  QuantizeMultiplierSmallerThanOne(real_output_multiplier, &param_.output_multiplier_, &param_.output_shift_);

  param_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  param_.output_activation_max_ = std::numeric_limits<int8_t>::max();

  int left_shift0 = -param_.input0_shift_ > 0 ? -param_.input0_shift_ : 0;
  param_.right_shift0_ = -param_.input0_shift_ > 0 ? 0 : param_.input0_shift_;

  int left_shift1 = -param_.input1_shift_ > 0 ? -param_.input1_shift_ : 0;
  param_.right_shift1_ = -param_.input1_shift_ > 0 ? 0 : param_.input1_shift_;

  param_.left_shift_out_ = -param_.output_shift_ > 0 ? -param_.output_shift_ : 0;
  param_.right_shift_out_ = -param_.output_shift_ > 0 ? 0 : param_.output_shift_;

  param_.left_shift_result0_ = (1 << left_shift) * ((1 << left_shift0));
  param_.left_shift_result1_ = (1 << left_shift) * ((1 << left_shift1));

  MS_CHECK_TRUE(left_shift + left_shift0 == left_shift, "shift not match");
  MS_CHECK_TRUE(left_shift + left_shift1 == left_shift, "shift not match");

  return RET_OK;
}

int SubInt8Coder::DoCode(CoderContext *const context) {
  NNaclInt8Serializer code;
  // Todo: Parallel run wrapper
  auto element_num = output_tensor_->ElementsNum();
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
    MS_CHECK_PTR(tile0_data_);
    tile1_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    MS_CHECK_PTR(tile1_data_);

    code.CodeStruct("tile_para", tile_para);

    code.CodeFunction("TileDimensionsInt8", input0, input1, tile0_data_, tile1_data_, "&tile_para");
    code.CodeFunction("SubInt8", tile0_data_, tile1_data_, output_tensor_, element_num, "&param");
  } else {
    code.CodeFunction("SubInt8", input0, input1, output_tensor_, element_num, "&param");
  }

  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_SubFusion, CPUOpCoderCreator<SubInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
