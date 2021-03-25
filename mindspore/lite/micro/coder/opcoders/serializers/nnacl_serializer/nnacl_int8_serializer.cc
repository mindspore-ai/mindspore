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
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include <string>
#include "src/common/log_adapter.h"
#include "coder/opcoders/parallel.h"
#include "coder/log.h"

namespace mindspore::lite::micro::nnacl {
void NNaclInt8Serializer::CodeStruct(const std::string &name, const ConvParameter &conv_parameter) {
  const ConvQuantArg &quant_arg = conv_parameter.conv_quant_arg_;
  std::string quant_arg_in = name + "_quant_arg_in";
  std::string quant_arg_w = name + "_quant_arg_w";
  std::string quant_arg_out = name + "_quant_arg_out";
  CodeArray(quant_arg_in, quant_arg.input_quant_args_, quant_arg.input_arg_num_, false);
  CodeArray(quant_arg_w, quant_arg.filter_quant_args_, quant_arg.filter_arg_num_, false);
  CodeArray(quant_arg_out, quant_arg.output_quant_args_, quant_arg.output_arg_num_, false);

  std::string real_multiplier = name + "_real_multiplier";
  std::string left_shift = name + "_left_shift";
  std::string right_shift = name + "_right_shift";
  std::string quant_multiplier = name + "_quant_multiplier";
  CodeArray(real_multiplier, quant_arg.real_multiplier_, quant_arg.filter_arg_num_, false);
  CodeArray(left_shift, quant_arg.left_shift_, quant_arg.filter_arg_num_, false);
  CodeArray(right_shift, quant_arg.right_shift_, quant_arg.filter_arg_num_, false);
  CodeArray(quant_multiplier, quant_arg.quant_multiplier_, quant_arg.filter_arg_num_, false);

  std::string out_act_min = name + "_out_act_min";
  std::string out_act_max = name + "_out_act_max";
  CodeArray(out_act_min, quant_arg.out_act_min_, 1, false);
  CodeArray(out_act_max, quant_arg.out_act_max_, 1, false);

  std::string conv_quant_arg = name + "_conv_quant_arg";

  CodeBaseStruct<false>("ConvQuantArg", conv_quant_arg, quant_arg.round_mode_, quant_arg.quant_multiplier_mode_,
                        quant_arg_in, quant_arg_w, quant_arg_out, real_multiplier, left_shift, right_shift,
                        quant_multiplier, out_act_min, out_act_max, quant_arg.input_arg_num_, quant_arg.filter_arg_num_,
                        quant_arg.output_arg_num_, quant_arg.per_channel_);
  code << "int thread_num = MSMIN(" << gThreadNum << ", " << conv_parameter.output_h_ << ");\n";
  CodeBaseStruct<false>(
    "ConvParameter", name, conv_parameter.op_parameter_, conv_quant_arg, conv_parameter.kernel_h_,
    conv_parameter.kernel_w_, conv_parameter.stride_h_, conv_parameter.stride_w_, conv_parameter.dilation_h_,
    conv_parameter.dilation_w_, conv_parameter.pad_u_, conv_parameter.pad_d_, conv_parameter.pad_l_,
    conv_parameter.pad_r_, conv_parameter.group_, conv_parameter.tile_num_, conv_parameter.input_batch_,
    conv_parameter.input_h_, conv_parameter.input_w_, conv_parameter.input_channel_, conv_parameter.output_batch_,
    conv_parameter.output_h_, conv_parameter.output_w_, conv_parameter.output_channel_, "thread_num",
    conv_parameter.input_unit_, conv_parameter.output_unit_, conv_parameter.pad_mode_, conv_parameter.act_type_,
    conv_parameter.channel_multiplie_, conv_parameter.output_padding_w_, conv_parameter.output_padding_h_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const MatMulParameter &matmul_parameter) {
  CodeBaseStruct<false>("MatMulParameter", name, matmul_parameter.op_parameter_, matmul_parameter.has_bias_,
                        matmul_parameter.row_, matmul_parameter.col_, matmul_parameter.row_4_, matmul_parameter.row_6_,
                        matmul_parameter.row_12_, matmul_parameter.row_16_, matmul_parameter.row_align_,
                        matmul_parameter.col_4_, matmul_parameter.col_8_, matmul_parameter.col_align_,
                        matmul_parameter.deep_, matmul_parameter.deep_4_, matmul_parameter.deep_16_,
                        matmul_parameter.batch, matmul_parameter.a_transpose_, matmul_parameter.b_transpose_,
                        matmul_parameter.a_const_, matmul_parameter.b_const_, matmul_parameter.act_type_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const AddQuantParameter &add_quant_parameter) {
  CodeBaseStruct<false>("AddQuantParameter", name, add_quant_parameter.left_shift_, add_quant_parameter.min_,
                        add_quant_parameter.max_, add_quant_parameter.in0_args_, add_quant_parameter.in1_args_,
                        add_quant_parameter.out_zp_, add_quant_parameter.out_left_shift_,
                        add_quant_parameter.out_right_shift_, add_quant_parameter.out_multiplier_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const ArithmeticParameter &arithmetic_parameter) {
  CodeBaseStruct<false>("ArithmeticParameter", name, arithmetic_parameter.op_parameter_,
                        arithmetic_parameter.broadcasting_, arithmetic_parameter.ndim_,
                        arithmetic_parameter.activation_type_, ToString(arithmetic_parameter.in_shape0_),
                        arithmetic_parameter.in_elements_num0_, ToString(arithmetic_parameter.in_shape1_),
                        arithmetic_parameter.in_elements_num1_, ToString(arithmetic_parameter.out_shape_),
                        arithmetic_parameter.out_elements_num_, ToString(arithmetic_parameter.in_strides0_),
                        ToString(arithmetic_parameter.in_strides1_), ToString(arithmetic_parameter.out_strides_),
                        ToString(arithmetic_parameter.multiples0_), ToString(arithmetic_parameter.multiples1_));
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

  CodeBaseStruct("PoolingParameter", name,
                 // Primitive parameter
                 pooling_parameter.op_parameter_, pooling_parameter.pool_mode_, pooling_parameter.round_mode_,
                 pooling_parameter.pad_mode_, pooling_parameter.act_type_, pooling_parameter.avg_mode_,
                 pooling_parameter.global_, pooling_parameter.window_w_, pooling_parameter.window_h_,
                 pooling_parameter.stride_w_, pooling_parameter.stride_h_,
                 // shape correlative
                 pooling_parameter.input_w_, pooling_parameter.input_h_, pooling_parameter.input_batch_,
                 pooling_parameter.input_channel_, pooling_parameter.output_w_, pooling_parameter.output_h_,
                 pooling_parameter.output_batch_, pooling_parameter.output_channel_, pooling_parameter.pad_u_,
                 pooling_parameter.pad_d_, pooling_parameter.pad_l_, pooling_parameter.pad_r_,
                 // other parameter
                 pooling_parameter.thread_num_, quant_name, pooling_parameter.quantize_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const SoftmaxParameter &softmax_parameter) {
  CodeBaseStruct("SoftmaxParameter", name, softmax_parameter.op_parameter_, softmax_parameter.axis_,
                 ToString(softmax_parameter.input_shape_), softmax_parameter.element_size_, softmax_parameter.n_dim_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const SliceParameter &slice_parameter) {
  CodeBaseStruct("SliceParameter", name, slice_parameter.op_parameter_, ToString(slice_parameter.shape_),
                 ToString(slice_parameter.begin_), ToString(slice_parameter.end_), ToString(slice_parameter.size_),
                 slice_parameter.quant_arg_, slice_parameter.param_length_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const BatchNormParameter &batchnorm_parameter) {
  CodeBaseStruct("BatchNormParameter", name, batchnorm_parameter.op_parameter_, batchnorm_parameter.epsilon_,
                 batchnorm_parameter.momentum_, batchnorm_parameter.unit_, batchnorm_parameter.units_,
                 batchnorm_parameter.channel_, batchnorm_parameter.fused_);
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
    CodeArray(get_shape_name(i), concat_parameter.input_shapes_[i], in_shape, false);
  }

  code << "int *" << input_shapes_name << "[] = {";
  for (int i = 0; i < in_tensor_count; ++i) {
    code << get_shape_name(i) << " ,";
  }
  code << "};\n";
  // output_shape
  CodeArray(output_shapes_name, concat_parameter.output_shapes_, out_shape, false);

  CodeBaseStruct<false>("ConcatParameter", name, concat_parameter.op_parameter_, quant_arg_name, concat_parameter.axis_,
                        concat_parameter.thread_count_, concat_parameter.input_num_, "(int **)" + input_shapes_name,
                        output_shapes_name, concat_parameter.after_axis_size, concat_parameter.count_unit_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const ::QuantArg &quant_arg) {
  CodeBaseStruct("QuantArg", name, quant_arg.scale_, quant_arg.zp_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const ::QuantMulArg &quant_mul_arg) {
  CodeBaseStruct("QuantMulArg", name, quant_mul_arg.multiplier_, quant_mul_arg.left_shift_, quant_mul_arg.right_shift_);
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

void NNaclInt8Serializer::CodeStruct(const std::string &name, const MatmulQuantParameter &matmul_quant_arg,
                                     int weight_quant_num) {
  CodeArray("filter_scale", matmul_quant_arg.filter_scale_, weight_quant_num, false);
  CodeArray("filter_zp", matmul_quant_arg.filter_zp_, weight_quant_num, false);
  CodeArray("left_shift", matmul_quant_arg.left_shift_, weight_quant_num, false);
  CodeArray("right_shift", matmul_quant_arg.right_shift_, weight_quant_num, false);
  CodeArray("multiplier", matmul_quant_arg.quant_multiplier_, weight_quant_num, false);
  CodeBaseStruct("MatmulQuantParameter", name, matmul_quant_arg.input_, matmul_quant_arg.weight_,
                 matmul_quant_arg.output_, matmul_quant_arg.out_act_min_, matmul_quant_arg.out_act_max_, "filter_scale",
                 "filter_zp", "left_shift", "right_shift", "multiplier");
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const SubQuantArg &sub_quant_arg) {
  CodeBaseStruct("SubQuantArg", name, sub_quant_arg.in0_args_, sub_quant_arg.in1_args_, sub_quant_arg.out_args_,
                 sub_quant_arg.output_activation_min_, sub_quant_arg.output_activation_max_,
                 sub_quant_arg.input0_multiplier_, sub_quant_arg.input1_multiplier_, sub_quant_arg.output_multiplier_,
                 sub_quant_arg.input0_shift_, sub_quant_arg.input1_shift_, sub_quant_arg.output_shift_,
                 sub_quant_arg.left_shift_result0_, sub_quant_arg.left_shift_result1_, sub_quant_arg.right_shift0_,
                 sub_quant_arg.right_shift1_, sub_quant_arg.left_shift_out_, sub_quant_arg.right_shift_out_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const DivQuantArg &div_quant_arg) {
  CodeBaseStruct("DivQuantArg", name, div_quant_arg.in0_args_, div_quant_arg.in1_args_, div_quant_arg.out_args_,
                 div_quant_arg.output_activation_min_, div_quant_arg.output_activation_max_,
                 div_quant_arg.output_multiplier_, div_quant_arg.output_shift_);
}

void NNaclInt8Serializer::CodeStruct(const std::string &name, const ReluXQuantArg &relu_quant_arg) {
  CodeBaseStruct("ReluXQuantArg", name, relu_quant_arg.input_arg, relu_quant_arg.output_arg,
                 relu_quant_arg.input_multiplier_, relu_quant_arg.left_shift_, relu_quant_arg.right_shift_,
                 relu_quant_arg.quantized_output_min, relu_quant_arg.quantized_output_max);
}

}  // namespace mindspore::lite::micro::nnacl
