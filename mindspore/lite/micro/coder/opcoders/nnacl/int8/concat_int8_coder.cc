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

#include "coder/opcoders/nnacl/int8/concat_int8_coder.h"
#include <limits>
#include "nnacl/int8/concat_int8.h"
#include "nnacl/int8/arithmetic_int8.h"
#include "nnacl/int8/quantize.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

int MallocQuantArgForConcat(ConcatQuantArg *quant_arg, size_t input_num) {
  quant_arg->in_args_ = static_cast<QuantArg *>(malloc(sizeof(QuantArg) * input_num));
  MS_CHECK_PTR(quant_arg->in_args_);
  return mindspore::lite::RET_OK;
}

using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::lite::micro::nnacl {
int ConcatInt8Coder::Prepare(CoderContext *const context) {
  this->concat_param_ = reinterpret_cast<ConcatParameter *>(parameter_);

  concat_param_->input_shapes_ = nullptr;
  size_t input_num = input_tensors().size();
  MS_CHECK_RET_CODE(MallocQuantArgForConcat(&concat_param_->quant_arg_, input_num),
                    "Null pointer reference: quant_concat_parm_->in_quant_args_.");
  for (int i = 0; i < static_cast<int>(input_num); i++) {
    auto *input_tensor = input_tensors().at(i);
    auto quant_args = input_tensor->quant_params();
    concat_param_->quant_arg_.in_args_[i].scale_ = quant_args.at(0).scale;
    concat_param_->quant_arg_.in_args_[i].zp_ = quant_args.at(0).zeroPoint;
  }

  auto quant_args = output_tensor_->quant_params();
  concat_param_->quant_arg_.out_args_.scale_ = quant_args.at(0).scale;
  concat_param_->quant_arg_.out_args_.zp_ = quant_args.at(0).zeroPoint;

  concat_param_->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  concat_param_->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  // concat base resize
  axis_ = concat_param_->axis_ >= 0 ? concat_param_->axis_ : input_tensor_->shape().size() + concat_param_->axis_;
  // concat int8 resize
  concat_param_->input_num_ = input_num;
  concat_param_->input_shapes_ = reinterpret_cast<int **>(malloc(sizeof(int *) * input_num));
  MS_CHECK_PTR(concat_param_->input_shapes_);
  for (int i = 0; i < static_cast<int>(input_num); i++) {
    auto in_shape = input_tensors_.at(i)->shape();
    concat_param_->input_shapes_[i] = reinterpret_cast<int *>(malloc(in_shape.size() * sizeof(int)));
    MS_CHECK_PTR(concat_param_->input_shapes_[i]);
    memcpy(reinterpret_cast<void *>(concat_param_->input_shapes_[i]), in_shape.data(), sizeof(int) * in_shape.size());
  }

  before_axis_size = 1;
  for (int i = 0; i < axis_ && i < static_cast<int>(output_tensor_->shape().size()); i++) {
    before_axis_size *= output_tensor_->DimensionSize(i);
  }

  int64_t after_axis_size = 1;
  int output_dim = static_cast<int>(output_tensor_->shape().size());
  concat_param_->output_shapes_ = reinterpret_cast<int *>(malloc(output_dim * sizeof(int)));
  MS_CHECK_PTR(concat_param_->output_shapes_);
  memcpy(reinterpret_cast<void *>(concat_param_->output_shapes_), output_tensor_->shape().data(),
         sizeof(int) * output_dim);
  for (int i = axis_ + 1; i < output_dim; i++) {
    after_axis_size *= concat_param_->output_shapes_[i];
  }
  concat_param_->after_axis_size = after_axis_size;
  return RET_OK;
}

int ConcatInt8Coder::DoCode(CoderContext *const context) {
  concat_param_->thread_count_ = thread_num_;
  MS_CHECK_TRUE(thread_num_ > 0, "thread_num_ <= 0");
  count_unit_ = thread_num_ > 1 ? UP_DIV(before_axis_size, thread_num_) : before_axis_size;
  concat_param_->count_unit_ = count_unit_;

  Collect(context, {"nnacl/int8/concat_int8.h", "wrapper/int8/concat_int8_wrapper.h"},
          {"concat_int8.c", "concat_int8_wrapper.c"});
  NNaclInt8Serializer code;

  int in_tensor_count = input_tensors().size();
  code << "int8_t *input_data[" << in_tensor_count << "];\n";
  // input data
  for (int i = 0; i < static_cast<int>(input_tensors().size()); ++i) {
    MS_CHECK_PTR(input_tensors().at(i));
    code << "input_data[" << i << "] = " << allocator_->GetRuntimeAddr(input_tensors().at(i)) << ";\n";
  }
  code.CodeStruct("concat_param", *concat_param_, in_tensor_count, input_tensor_->shape().size(),
                  output_tensor_->shape().size());
  code.CodeBaseStruct<false>("ConcatInt8Args", kRunArgs, "input_data", output_tensor_, "&concat_param", axis_,
                             before_axis_size, count_unit_);
  if (support_parallel_) {
    code.CodeFunction(kParallelLaunch, gThreadPool, "ConcatInt8Run", kRunArgsAddr, gThreadNum);
  } else {
    code.CodeFunction("ConcatInt8Run", kRunArgsAddr, kDefaultTaskId);
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Concat, CPUOpCoderCreator<ConcatInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
