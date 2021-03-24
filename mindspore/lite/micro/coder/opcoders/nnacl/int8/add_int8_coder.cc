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

#include "coder/opcoders/nnacl/int8/add_int8_coder.h"
#include <algorithm>
#include <type_traits>
#include "nnacl/int8/quantize.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_AddFusion;

namespace mindspore::lite::micro::nnacl {

int AddInt8Coder::Prepare(CoderContext *const context) {
  input0 = input_tensors().at(0);
  input1 = input_tensors().at(1);
  MS_CHECK_PTR(input0);
  MS_CHECK_PTR(input1);

  MS_CHECK_RET_CODE(Init(), "Init failed");
  MS_CHECK_RET_CODE(ReSize(), "ReSize failed");

  return RET_OK;
}

int AddInt8Coder::Init() {
  arith_para_ = reinterpret_cast<ArithmeticParameter *>(parameter_);
  para_.in0_args_.zp_ = input0->quant_params().front().zeroPoint * -1;
  para_.in1_args_.zp_ = input1->quant_params().front().zeroPoint * -1;
  para_.out_zp_ = output_tensor_->quant_params().front().zeroPoint;

  const double in0_scale = input0->quant_params().front().scale;
  const double in1_scale = input1->quant_params().front().scale;
  const double out_scale = output_tensor_->quant_params().front().scale;

  para_.left_shift_ = 20;
  const double twice_max_input_scale = 2 * std::max(in0_scale, in1_scale);
  const double in0_multiplier = in0_scale / twice_max_input_scale;
  const double in1_multiplier = in1_scale / twice_max_input_scale;
  const double out_multiplier = twice_max_input_scale / ((1 << para_.left_shift_) * out_scale);

  QuantizeMultiplierSmallerThanOne(in0_multiplier, &para_.in0_args_.multiplier_, &para_.in0_args_.left_shift_);
  QuantizeMultiplierSmallerThanOne(in1_multiplier, &para_.in1_args_.multiplier_, &para_.in1_args_.left_shift_);
  QuantizeMultiplierSmallerThanOne(out_multiplier, &para_.out_multiplier_, &para_.out_left_shift_);

  para_.in0_args_.right_shift_ = -para_.in0_args_.left_shift_ > 0 ? 0 : para_.in0_args_.left_shift_;
  para_.in1_args_.right_shift_ = -para_.in1_args_.left_shift_ > 0 ? 0 : para_.in1_args_.left_shift_;
  para_.out_right_shift_ = -para_.out_left_shift_ > 0 ? 0 : para_.out_left_shift_;

  para_.in0_args_.left_shift_ = -para_.in0_args_.left_shift_ > 0 ? -para_.in0_args_.left_shift_ : 0;
  para_.in1_args_.left_shift_ = -para_.in1_args_.left_shift_ > 0 ? -para_.in1_args_.left_shift_ : 0;
  para_.out_left_shift_ = -para_.out_left_shift_ > 0 ? -para_.out_left_shift_ : 0;

  auto act = arith_para_->activation_type_;
  CalculateActivationRangeQuantized(act == ActType_Relu, act == ActType_Relu6, para_.out_zp_,
                                    static_cast<float>(out_scale), &para_.min_, &para_.max_);
  return RET_OK;
}

int AddInt8Coder::ReSize() {
  support_opt_add_ = (input0->ElementsNum() == 1) || (input1->ElementsNum() == 1);
  if (support_opt_add_) {
    arith_para_->broadcasting_ = false;
  }

  elements_num_ = output_tensor_->ElementsNum();

  arith_para_->in_elements_num0_ = input_tensors_[0]->ElementsNum();
  arith_para_->in_elements_num1_ = input_tensors_[1]->ElementsNum();
  arith_para_->out_elements_num_ = output_tensors_[0]->ElementsNum();

  for (size_t i = 0; i < input_tensors_.at(0)->shape().size(); i++) {
    if (arith_para_->in_shape0_[i] == -1) {
      MS_CHECK_RET_CODE(memcpy_s(arith_para_->in_shape0_, std::extent<decltype(arith_para_->in_shape0_)>::value,
                                 input0->shape().data(), input0->shape().size() * sizeof(int)),
                        "memcpy failed");
      break;
    }
  }
  for (size_t i = 0; i < input_tensors_.at(1)->shape().size(); i++) {
    if (arith_para_->in_shape1_[i] == -1) {
      MS_CHECK_RET_CODE(memcpy_s(arith_para_->in_shape1_, std::extent<decltype(arith_para_->in_shape1_)>::value,
                                 input1->shape().data(), input1->shape().size() * sizeof(int)),
                        "memcpy failed");
      break;
    }
  }
  for (size_t i = 0; i < output_tensor_->shape().size(); i++) {
    if (arith_para_->out_shape_[i] == -1) {
      MS_CHECK_RET_CODE(memcpy_s(arith_para_->out_shape_, std::extent<decltype(arith_para_->out_shape_)>::value,
                                 output_tensor_->shape().data(), output_tensor_->shape().size() * sizeof(int)),
                        "memcpy failed");
      break;
    }
  }

  if (arith_para_->broadcasting_) {
    size_t break_pos_ = 0;
    for (auto i = arith_para_->ndim_ - 1; i >= 0; --i) {
      if (arith_para_->in_shape0_[i] != arith_para_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
    }
    in_size_ = 1;
    out_size_ = 1;
    for (size_t i = 0; i < arith_para_->ndim_; i++) {
      if (i > break_pos_) {
        in_size_ *= arith_para_->out_shape_[i];
      } else {
        out_size_ *= arith_para_->out_shape_[i];
      }
    }

    ComputeStrides(arith_para_->in_shape0_, arith_para_->in_strides0_, arith_para_->ndim_);
    ComputeStrides(arith_para_->in_shape1_, arith_para_->in_strides1_, arith_para_->ndim_);
    ComputeStrides(arith_para_->out_shape_, arith_para_->out_strides_, arith_para_->ndim_);
  }
  return RET_OK;
}

int AddInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"wrapper/int8/add_int8_wrapper.h"},
          {"add_int8_wrapper.c", "add_int8.c", "arithmetic_base.c", "arithmetic_int8.c"});

  nnacl::NNaclInt8Serializer code;

  code.CodeStruct("para", para_);
  code.CodeStruct("arith_para", *arith_para_);
  code.CodeBaseStruct("AddInt8Args", kRunArgs, "&para", "&arith_para", in_size_, out_size_, gThreadNum, elements_num_,
                      support_opt_add_, input0, input1, output_tensor_);
  if (support_parallel_) {
    if (arith_para_->broadcasting_) {
      code.CodeFunction(kParallelLaunch, gThreadPool, "AddBroadcastInt8Run", kRunArgsAddr, gThreadNum);
    } else {
      code.CodeFunction(kParallelLaunch, gThreadPool, "AddInt8Run", kRunArgsAddr, gThreadNum);
    }
  } else {
    if (arith_para_->broadcasting_) {
      code.CodeFunction("AddBroadcastInt8Run", kRunArgsAddr, kDefaultTaskId);
    } else {
      code.CodeFunction("AddInt8Run", kRunArgsAddr, kDefaultTaskId);
    }
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_AddFusion, CPUOpCoderCreator<AddInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
