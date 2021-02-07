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
#include "micro/coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include <string>
#include "src/common/log_adapter.h"
#include "micro/coder/log.h"

namespace mindspore::lite::micro::nnacl {
void NNaclInt8Serializer::CodeStruct(const std::string &name, const ArithmeticParameter &arithmetic_parameter) {
  CodeBaseStruct("ArithmeticParameter", name, arithmetic_parameter.op_parameter_, arithmetic_parameter.broadcasting_,
                 arithmetic_parameter.ndim_, arithmetic_parameter.activation_type_,
                 ToString(arithmetic_parameter.in_shape0_), arithmetic_parameter.in_elements_num0_,
                 ToString(arithmetic_parameter.in_shape1_), arithmetic_parameter.in_elements_num1_,
                 ToString(arithmetic_parameter.out_shape_), arithmetic_parameter.out_elements_num_,
                 ToString(arithmetic_parameter.in_strides0_), ToString(arithmetic_parameter.in_strides1_),
                 ToString(arithmetic_parameter.out_strides_), ToString(arithmetic_parameter.multiples0_),
                 ToString(arithmetic_parameter.multiples1_));
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const PoolingParameter &pooling_parameter) {
  std::string quant_name = name + "_quant";
  std::string in_quant_name = quant_name + "_in";
  std::string out_quant_name = quant_name + "_out";

  MS_CHECK_PTR_IF_NULL(pooling_parameter.quant_args_);
  ::QuantArg *in_quant_args = pooling_parameter.quant_args_[0];
  ::QuantArg *out_quant_args = pooling_parameter.quant_args_[1];
  MS_CHECK_PTR_IF_NULL(in_quant_args);
  MS_CHECK_PTR_IF_NULL(out_quant_args);

  code << "static QuantArg " << in_quant_name << " = " << *out_quant_args << ";\n";
  code << "static QuantArg " << out_quant_name << " = " << *out_quant_args << ";\n";

  code << "static QuantArg *" << quant_name << "[2] = {"
       << " &" << in_quant_name << ", "
       << " &" << out_quant_name << "};\n";

  CodeBaseStruct("PoolingParameter", name, pooling_parameter.op_parameter_, pooling_parameter.pool_mode_,
                 pooling_parameter.round_mode_, pooling_parameter.act_type_, pooling_parameter.avg_mode_,
                 pooling_parameter.global_, pooling_parameter.window_w_, pooling_parameter.window_h_,
                 pooling_parameter.stride_w_, pooling_parameter.stride_h_, pooling_parameter.input_w_,
                 pooling_parameter.input_h_, pooling_parameter.input_batch_, pooling_parameter.input_channel_,
                 pooling_parameter.output_w_, pooling_parameter.output_h_, pooling_parameter.output_batch_,
                 pooling_parameter.output_channel_, pooling_parameter.pad_u_, pooling_parameter.pad_d_,
                 pooling_parameter.pad_l_, pooling_parameter.pad_r_, pooling_parameter.op_parameter_.thread_num_,
                 quant_name, pooling_parameter.quantize_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const SoftmaxParameter &softmax_parameter) {
  CodeBaseStruct("SoftmaxParameter", name, softmax_parameter.op_parameter_, softmax_parameter.axis_,
                 ToString(softmax_parameter.input_shape_), softmax_parameter.element_size_, softmax_parameter.n_dim_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const SoftmaxQuantArg &softmax_quant_parameter) {
  CodeBaseStruct("SoftmaxQuantArg", name, softmax_quant_parameter.in_quant_args_,
                 softmax_quant_parameter.out_quant_arg_, softmax_quant_parameter.output_activation_min_,
                 softmax_quant_parameter.output_activation_max_, softmax_quant_parameter.output_multiplier_,
                 softmax_quant_parameter.shift_left_, softmax_quant_parameter.shift_right_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const ConcatParameter &concat_parameter,
                                     int in_tensor_count, int in_shape, int out_shape) {
  std::string quant_arg_name = name + "_quant_arg";
  std::string in_args_name = quant_arg_name + "_in_args";
  std::string input_shapes_name = name + "_input_shapes";
  std::string output_shapes_name = name + "_output_shapes";

  CodeArray(in_args_name, concat_parameter.quant_arg_.in_args_, in_tensor_count, false);
  CodeBaseStruct("ConcatQuantArg", quant_arg_name, in_args_name, concat_parameter.quant_arg_.out_args_,
                 concat_parameter.quant_arg_.output_activation_min_,
                 concat_parameter.quant_arg_.output_activation_max_);

  auto get_shape_name = [&input_shapes_name](int i) { return input_shapes_name + "_" + std::to_string(i); };
  // input_shape
  for (int i = 0; i < in_tensor_count; ++i) {
    CodeArray(get_shape_name(i), concat_parameter.input_shapes_[i], in_shape);
  }

  code << "const int *" << input_shapes_name << "[] = {";
  for (int i = 0; i < in_tensor_count; ++i) {
    code << get_shape_name(i) << " ,";
  }
  code << "};\n";
  // output_shape
  CodeArray(output_shapes_name, concat_parameter.output_shapes_, out_shape, false);

  CodeBaseStruct("ConcatParameter", name, concat_parameter.op_parameter_, quant_arg_name, concat_parameter.axis_,
                 concat_parameter.thread_count_, concat_parameter.input_num_, input_shapes_name, output_shapes_name,
                 concat_parameter.after_axis_size, concat_parameter.count_unit_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const ReduceQuantArg &reduce_quant_arg) {
  CodeBaseStruct(
    "ReduceQuantArg", name, reduce_quant_arg.in_scale_, reduce_quant_arg.in_zp_, reduce_quant_arg.out_scale_,
    reduce_quant_arg.out_zp_, reduce_quant_arg.in_out_multiplier_, reduce_quant_arg.in_out_left_shift_,
    reduce_quant_arg.in_out_right_shift_, reduce_quant_arg.mean_multiplier_, reduce_quant_arg.mean_left_shift_,
    reduce_quant_arg.mean_right_shift_, reduce_quant_arg.prod_multiplier_, reduce_quant_arg.prod_left_shift_,
    reduce_quant_arg.prod_right_shift_, reduce_quant_arg.sum_square_multiplier_,
    reduce_quant_arg.sum_square_left_shift_, reduce_quant_arg.sum_square_right_shift_);
}
void NNaclInt8Serializer::CodeStruct(const std::string &name, const ReshapeQuantArg &reshape_quant_arg) {
  CodeBaseStruct("ReshapeQuantArg", name, reshape_quant_arg.in_args_, reshape_quant_arg.out_args_,
                 reshape_quant_arg.output_activation_min_, reshape_quant_arg.output_activation_max_);
}

}  // namespace mindspore::lite::micro::nnacl
