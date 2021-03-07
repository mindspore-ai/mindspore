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

#include "coder/opcoders/nnacl/int8/softmax_int8_coder.h"
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include "schema/inner/ops_generated.h"
#include "nnacl/softmax_parameter.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::lite::micro::nnacl {
int SoftMaxInt8Coder::Prepare(CoderContext *const context) {
  SoftmaxBaseCoder::Init();
  std::vector<QuantArg> in_quant_args = input_tensor_->quant_params();
  quant_params_.in_quant_args_.scale_ = in_quant_args.at(0).scale;
  quant_params_.in_quant_args_.zp_ = -in_quant_args.at(0).zeroPoint;

  std::vector<QuantArg> out_quant_args = output_tensor_->quant_params();
  quant_params_.out_quant_arg_.scale_ = out_quant_args.at(0).scale;
  quant_params_.out_quant_arg_.zp_ = out_quant_args.at(0).zeroPoint;
  quant_params_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  quant_params_.output_activation_max_ = std::numeric_limits<int8_t>::max();

  const double input_real_multiplier =
    MSMIN(quant_params_.in_quant_args_.scale_ * (1 << (unsigned int)(31 - 5)), (1ll << 31) - 1.0);
  int right_shift = 0;
  QuantizeMultiplierSmallerThanOne(input_real_multiplier, &quant_params_.output_multiplier_, &right_shift);
  quant_params_.shift_left_ = right_shift < 0 ? -right_shift : 0;
  quant_params_.shift_right_ = right_shift < 0 ? -right_shift : 0;
  // malloc tmp buffer
  exp_data_size_ = softmax_param_->element_size_ * sizeof(int);
  exp_data_ = static_cast<int *>(allocator_->Malloc(kNumberTypeInt32, exp_data_size_, kWorkspace));
  MS_CHECK_PTR(exp_data_);
  int inner_size = 1;
  MS_CHECK_TRUE(softmax_param_->n_dim_ < 5, "n_dim should be less than the length of maximum value of input_shape");
  for (int i = softmax_param_->axis_ + 1; i < softmax_param_->n_dim_; i++) {
    inner_size *= softmax_param_->input_shape_[i];
  }
  sum_data_size_ = inner_size * sizeof(int);
  sum_data_ = static_cast<int *>(allocator_->Malloc(kNumberTypeInt32, sum_data_size_, kWorkspace));
  MS_CHECK_PTR(sum_data_);
  ReSize();
  return RET_OK;
}

int SoftMaxInt8Coder::DoCode(CoderContext *const context) {
  int outter_size = 1;
  for (int i = 0; i < softmax_param_->axis_; i++) {
    outter_size *= softmax_param_->input_shape_[i];
  }
  MS_CHECK_TRUE(softmax_param_->n_dim_ < 5, "n_dim should be less than the length of maximum value of input_shape");
  Collect(context, {"nnacl/int8/softmax_int8.h"}, {"softmax_int8.c", "fixed_point.c"});

  NNaclInt8Serializer code;
  code.precision(kPrecision);

  code.CodeStruct("quant_args", quant_params_);
  code.CodeStruct("softmax_parameter", *softmax_param_);

  code.CodeFunction("memset", exp_data_, 0, exp_data_size_);
  code.CodeFunction("memset", sum_data_, 0, sum_data_size_);

  MS_CHECK_TRUE(thread_num_ > 0, "thread_num_ <= 0");
  int stride = UP_DIV(outter_size, thread_num_);
  int count = MSMIN(stride, outter_size - stride * kDefaultTaskId);
  code.CodeFunction("SoftmaxInt8", input_tensor_, output_tensor_, count, exp_data_, sum_data_, "quant_args",
                    "(SoftmaxParameter *)&softmax_parameter");
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Softmax, CPUOpCoderCreator<SoftMaxInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
