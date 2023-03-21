/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "src/common/log_adapter.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "nnacl/pooling_parameter.h"

namespace mindspore::lite::micro::nnacl {
int NNaclFp32Serializer::count = 0;
void NNaclFp32Serializer::CodeStruct(const std::string &name, const PoolingParameter &pooling_parameter) {
  CodeBaseStruct<false>("PoolingParameter", name,
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
                        gThreadNum, nullptr, pooling_parameter.quantize_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const BatchNormParameter &batch_norm_parameter) {
  CodeBaseStruct("BatchNormParameter", name, batch_norm_parameter.op_parameter_, batch_norm_parameter.epsilon_,
                 batch_norm_parameter.momentum_, batch_norm_parameter.unit_, batch_norm_parameter.units_,
                 batch_norm_parameter.channel_, batch_norm_parameter.fused_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const InstanceNormParameter &param) {
  CodeBaseStruct("InstanceNormParameter", name, param.op_parameter_, param.epsilon_, param.batch_, param.channel_,
                 param.inner_size_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const ArithmeticParameter &arithmetic_parameter) {
  CodeBaseStruct<false>("ArithmeticParameter", name, arithmetic_parameter.op_parameter_,
                        arithmetic_parameter.broadcasting_, arithmetic_parameter.ndim_,
                        arithmetic_parameter.activation_type_, ToString(arithmetic_parameter.in_shape0_),
                        arithmetic_parameter.in_elements_num0_, ToString(arithmetic_parameter.in_shape1_),
                        arithmetic_parameter.in_elements_num1_, ToString(arithmetic_parameter.out_shape_),
                        arithmetic_parameter.out_elements_num_, ToString(arithmetic_parameter.in_strides0_),
                        ToString(arithmetic_parameter.in_strides1_), ToString(arithmetic_parameter.out_strides_),
                        ToString(arithmetic_parameter.multiples0_), ToString(arithmetic_parameter.multiples1_));
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const SoftmaxParameter &softmax_parameter) {
  CodeBaseStruct<false>("SoftmaxParameter", name, softmax_parameter.op_parameter_, softmax_parameter.axis_,
                        ToString(softmax_parameter.input_shape_), softmax_parameter.element_size_,
                        softmax_parameter.n_dim_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const ConvParameter &conv_parameter) {
  code << "    int thread_num = MSMIN(" << gThreadNum << ", " << conv_parameter.output_h_ << ");\n";
  CodeBaseStruct<false>(
    "ConvParameter", name, conv_parameter.op_parameter_, "{0}", conv_parameter.kernel_h_, conv_parameter.kernel_w_,
    conv_parameter.stride_h_, conv_parameter.stride_w_, conv_parameter.dilation_h_, conv_parameter.dilation_w_,
    conv_parameter.pad_u_, conv_parameter.pad_d_, conv_parameter.pad_l_, conv_parameter.pad_r_, conv_parameter.group_,
    conv_parameter.tile_num_, conv_parameter.input_batch_, conv_parameter.input_h_, conv_parameter.input_w_,
    conv_parameter.input_channel_, conv_parameter.output_batch_, conv_parameter.output_h_, conv_parameter.output_w_,
    conv_parameter.output_channel_, "thread_num", conv_parameter.input_unit_, conv_parameter.output_unit_,
    conv_parameter.pad_mode_, conv_parameter.act_type_, conv_parameter.channel_multiplie_,
    conv_parameter.output_padding_w_, conv_parameter.output_padding_h_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const MatMulParameter &mat_mul_parameter) {
  CodeBaseStruct<false>(
    "MatMulParameter", name, mat_mul_parameter.op_parameter_, mat_mul_parameter.has_bias_, mat_mul_parameter.row_,
    mat_mul_parameter.col_, mat_mul_parameter.row_4_, mat_mul_parameter.row_6_, mat_mul_parameter.row_12_,
    mat_mul_parameter.row_16_, mat_mul_parameter.row_align_, mat_mul_parameter.col_4_, mat_mul_parameter.col_8_,
    mat_mul_parameter.col_align_, mat_mul_parameter.deep_, mat_mul_parameter.deep_4_, mat_mul_parameter.deep_16_,
    mat_mul_parameter.deep_align_, mat_mul_parameter.batch, mat_mul_parameter.a_transpose_,
    mat_mul_parameter.b_transpose_, mat_mul_parameter.a_const_, mat_mul_parameter.b_const_, mat_mul_parameter.act_type_,
    mat_mul_parameter.use_axis_, mat_mul_parameter.axis_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const ScaleParameter &scale_parameter) {
  CodeBaseStruct("ScaleParameter", name, scale_parameter.op_parameter_, scale_parameter.axis_,
                 scale_parameter.activation_type_, scale_parameter.outer_size_, scale_parameter.axis_size_,
                 scale_parameter.inner_size_, scale_parameter.const_scale_, scale_parameter.const_offset_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const SliceParameter &slice_parameter) {
  CodeBaseStruct("SliceParameter", name, slice_parameter.op_parameter_, ToString(slice_parameter.shape_),
                 ToString(slice_parameter.begin_), ToString(slice_parameter.end_), ToString(slice_parameter.size_),
                 "{0}", slice_parameter.param_length_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const SplitParameter &split_parameter) {
  CodeBaseStruct("SplitParameter", name, split_parameter.op_parameter_, split_parameter.num_split_, "split_sizes",
                 split_parameter.split_dim_, ToString(split_parameter.strides_), "{0}", split_parameter.n_dims_,
                 split_parameter.split_count_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const TileParameter &tile_parameter) {
  CodeBaseStruct("TileParameter", name, tile_parameter.op_parameter_, ToString(tile_parameter.multiples_),
                 ToString(tile_parameter.in_shape_), ToString(tile_parameter.out_shape_),
                 ToString(tile_parameter.in_strides_), ToString(tile_parameter.out_strides_), tile_parameter.in_dim_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const TransposeParameter &transpose_parameter) {
  CodeBaseStruct<false>(
    "TransposeParameter", name, transpose_parameter.op_parameter_, ToString(transpose_parameter.perm_),
    transpose_parameter.perm_size_, transpose_parameter.conjugate_, ToString(transpose_parameter.strides_),
    ToString(transpose_parameter.out_strides_), transpose_parameter.num_axes_, transpose_parameter.data_num_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const LstmParameter &lstm_parameter) {
  CodeBaseStruct("LstmParameter", name, lstm_parameter.op_parameter_, lstm_parameter.input_size_,
                 lstm_parameter.hidden_size_, lstm_parameter.seq_len_, lstm_parameter.batch_,
                 lstm_parameter.output_step_, lstm_parameter.bidirectional_, lstm_parameter.zoneout_cell_,
                 lstm_parameter.zoneout_hidden_, lstm_parameter.input_row_align_, lstm_parameter.input_col_align_,
                 lstm_parameter.state_row_align_, lstm_parameter.state_col_align_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const DeQuantArg &de_quant_arg) {
  // this clusters is meaningless which will be supported in future
  CodeBaseStruct("DeQuantArg", name, de_quant_arg.scale, de_quant_arg.zeroPoint, de_quant_arg.var_corr,
                 de_quant_arg.mean_corr, "NULL", de_quant_arg.clusters_nums, de_quant_arg.bitNum);
}
void NNaclFp32Serializer::CodeStruct(const std::string &name, const SpliceParameter &splice_parameter) {
  CodeArray("splice_context", splice_parameter.context_, splice_parameter.context_dim_, false);
  CodeBaseStruct("SpliceParameter", name, splice_parameter.op_parameter_, splice_parameter.context_dim_,
                 splice_parameter.forward_indexes_dim_, splice_parameter.src_to_dst_row_offset_, "splice_context",
                 nullptr, splice_parameter.output_dim_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const ExpParameter &exp_parameter) {
  CodeBaseStruct("ExpParameter", name, exp_parameter.op_parameter_, exp_parameter.base_, exp_parameter.scale_,
                 exp_parameter.shift_, exp_parameter.in_scale_, exp_parameter.out_scale_, exp_parameter.element_num_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const StridedSliceParameter &strided_slice_parameter) {
  CodeBaseStruct("StridedSliceParameter", name, strided_slice_parameter.op_parameter_,
                 ToString(strided_slice_parameter.begins_), ToString(strided_slice_parameter.ends_),
                 ToString(strided_slice_parameter.strides_), strided_slice_parameter.isScale,
                 strided_slice_parameter.in_shape_length_, ToString(strided_slice_parameter.in_shape_),
                 strided_slice_parameter.num_axes_, strided_slice_parameter.data_type,
                 strided_slice_parameter.begins_mask_, strided_slice_parameter.ellipsisMask_,
                 strided_slice_parameter.newAxisMask_, strided_slice_parameter.shrinkAxisMask_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const ArithmeticWrapperInfo &arithmetic_wrapper_info) {
  CodeBaseStruct("ArithmeticWrapperInfo", name, arithmetic_wrapper_info.offset0_, arithmetic_wrapper_info.stride0_,
                 arithmetic_wrapper_info.offset1_, arithmetic_wrapper_info.stride1_,
                 arithmetic_wrapper_info.out_offset_, arithmetic_wrapper_info.out_stride_,
                 arithmetic_wrapper_info.arithmetic_func_type_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const SpliceWrapperParam &splice_param) {
  CodeBaseStruct("SpliceWrapperParam", name, splice_param.src_row, splice_param.src_col, splice_param.dst_row,
                 splice_param.dst_col, splice_param.context_size, ToString(splice_param.context),
                 splice_param.src_to_dst_row_offset);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const TransFuncStr trans_func_str) {
  CodeBaseStruct("TransFuncList", name, trans_func_str.in_func_, nullptr, nullptr, trans_func_str.out_func_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const GroupNormParameter &gn_param) {
  CodeBaseStruct<false>("GroupNormParameter", name, gn_param.op_parameter_, gn_param.epsilon_, gn_param.num_groups_,
                        gn_param.channel_, gn_param.unit_, gn_param.batch_, gn_param.affine_);
}
void NNaclFp32Serializer::CodeStruct(const std::string &name, const ActivationParameter &activation_parameter) {
  CodeBaseStruct("ActivationParameter", name, activation_parameter.op_parameter_, activation_parameter.type_,
                 activation_parameter.alpha_, activation_parameter.min_val_, activation_parameter.max_val_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const OpParameter &op_param) {
  CodeBaseStruct<false>("OpParameter", name, op_param.name_, op_param.type_, op_param.thread_num_, op_param.quant_type_,
                        op_param.is_train_session_, op_param.is_zero_shape_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const LayerNormParameter &op_param) {
  CodeBaseStruct<false>("LayerNormParameter", name, op_param.op_parameter_, op_param.epsilon_,
                        op_param.elementwise_mode_, op_param.elementwise_affine_, op_param.begin_norm_axis_,
                        op_param.begin_params_axis_, op_param.norm_inner_size_, op_param.norm_outer_size_,
                        op_param.params_inner_size_, op_param.params_outer_size_, op_param.normalized_dims_,
                        ToString(op_param.normalized_shape_), op_param.thread_count_, op_param.thread_outsize_);
}

void NNaclFp32Serializer::CodeArrayStruct(const std::string &name, TensorC *tensorC, std::vector<Tensor *> tensor) {
  std::vector<std::string> tensor_names;
  int size = tensor.size();
  for (int i = 0; i < size; ++i) {
    std::string tensor_name = "tensor" + std::to_string(count++);
    CodeBaseStruct<false>("TensorC", name, tensorC[i].shape_changed_, tensorC[i].data_type_, tensorC[i].format_,
                          tensor[i], tensorC[i].shape_size_, ToString(tensorC[i].shape_), tensor_name);
    tensor_names.emplace_back(tensor_name);
  }
  code << "    TensorC"
       << " " << name << "[" << std::to_string(size) << "]"
       << " = {";
  for (int i = 0; i < size - 1; ++i) {
    code << tensor_names[i] << ", ";
  }
  code << tensor_names[size - 1];
  code << "};\n";
}
}  // namespace mindspore::lite::micro::nnacl
