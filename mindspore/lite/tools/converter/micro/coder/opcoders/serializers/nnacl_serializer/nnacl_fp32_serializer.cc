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

namespace mindspore::lite::micro::nnacl {
int NNaclFp32Serializer::count = 0;
void NNaclFp32Serializer::CodeStruct(const std::string &name, const PoolingParameter &pooling_parameter) {
  CodeBaseStruct<false>("PoolingParameter", name, pooling_parameter.op_parameter_, pooling_parameter.pool_mode_,
                        pooling_parameter.round_type_, pooling_parameter.pad_mode_, pooling_parameter.act_type_,
                        pooling_parameter.avg_mode_, pooling_parameter.global_, pooling_parameter.window_w_,
                        pooling_parameter.window_h_, pooling_parameter.stride_w_, pooling_parameter.stride_h_,
                        pooling_parameter.pad_u_, pooling_parameter.pad_d_, pooling_parameter.pad_l_,
                        pooling_parameter.pad_r_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const PoolingComputeParam &pooling_compute) {
  CodeBaseStruct<false>("PoolingComputeParam", name, pooling_compute.input_w_, pooling_compute.input_h_,
                        pooling_compute.input_batch_, pooling_compute.input_channel_, pooling_compute.output_w_,
                        pooling_compute.output_h_, pooling_compute.output_batch_, pooling_compute.output_channel_,
                        pooling_compute.window_w_, pooling_compute.window_h_, pooling_compute.minf,
                        pooling_compute.maxf);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const BatchNormStruct &bn_struct) {
  CodeBaseStruct<false>("BatchNormStruct", name, "{}", "{}", "{}", "{}", bn_struct.momentum_, bn_struct.unit_,
                        bn_struct.channel_, bn_struct.epsilon_);
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
  CodeBaseStruct<false>("SoftmaxParameter", name, softmax_parameter.op_parameter_, softmax_parameter.axis_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const int *list, int size) {
  code << "int32_t " << name << "[] = {";
  for (int i = 0; i < size - 1; i++) {
    code << list[i] << ",";
  }
  code << list[size - 1] << "};\n";
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

void NNaclFp32Serializer::CodeStruct(const std::string &name, const MicroMatmulParameter &micro_matmul_parameter) {
  CodeBaseStruct<false>(
    "MicroMatmulParameter", name, micro_matmul_parameter.act_type_, micro_matmul_parameter.thread_num_,
    micro_matmul_parameter.row_, micro_matmul_parameter.col_, micro_matmul_parameter.row_4_,
    micro_matmul_parameter.row_6_, micro_matmul_parameter.row_12_, micro_matmul_parameter.row_16_,
    micro_matmul_parameter.row_align_, micro_matmul_parameter.col_4_, micro_matmul_parameter.col_8_,
    micro_matmul_parameter.col_align_, micro_matmul_parameter.deep_, micro_matmul_parameter.deep_4_,
    micro_matmul_parameter.deep_16_, micro_matmul_parameter.deep_align_, micro_matmul_parameter.a_batch_,
    micro_matmul_parameter.b_batch_, micro_matmul_parameter.batch, micro_matmul_parameter.a_transpose_,
    micro_matmul_parameter.b_transpose_, micro_matmul_parameter.a_const_, micro_matmul_parameter.b_const_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const ScaleStruct &scale_struct) {
  CodeBaseStruct<false>("ScaleStruct", name, "{}", scale_struct.axis_, scale_struct.data_type_, scale_struct.axis_size_,
                        scale_struct.outer_size_, scale_struct.inner_size_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const SplitParameter &split_parameter) {
  CodeBaseStruct("SplitParameter", name, split_parameter.op_parameter_, split_parameter.num_split_, "split_sizes",
                 split_parameter.split_dim_, ToString(split_parameter.strides_), "{0}", split_parameter.n_dims_,
                 split_parameter.split_count_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const TileStruct &tile_struct) {
  CodeBaseStruct<false>(
    "TileStruct", name, "{}", tile_struct.one_dim_tile_, tile_struct.resize_done_, ToString(tile_struct.dims_),
    tile_struct.dims_size_, "NULL", "NULL", ToString(tile_struct.multiples_), ToString(tile_struct.in_shape_),
    ToString(tile_struct.out_shape_), ToString(tile_struct.in_strides_), ToString(tile_struct.out_strides_));
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const TransposeParameter &transpose_parameter) {
  CodeBaseStruct<false>(
    "TransposeParameter", name, transpose_parameter.op_parameter_, ToString(transpose_parameter.perm_),
    transpose_parameter.perm_size_, transpose_parameter.conjugate_, ToString(transpose_parameter.strides_),
    ToString(transpose_parameter.out_strides_), transpose_parameter.num_axes_, transpose_parameter.data_num_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const TransposeParameter &transpose_param,
                                     const TransposeDynamicParameter &dynamic_transpose_param) {
  CodeBaseStruct<false>("TransposeParameter", name, transpose_param.op_parameter_, ToString(transpose_param.perm_),
                        transpose_param.perm_size_, transpose_param.conjugate_, dynamic_transpose_param.strides_,
                        dynamic_transpose_param.out_strides_, transpose_param.num_axes_,
                        dynamic_transpose_param.data_num_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const LstmParameter &lstm_parameter) {
  CodeBaseStruct("LstmParameter", name, lstm_parameter.op_parameter_, lstm_parameter.input_size_,
                 lstm_parameter.hidden_size_, lstm_parameter.project_size_, lstm_parameter.output_size_,
                 lstm_parameter.seq_len_, lstm_parameter.batch_, lstm_parameter.output_step_,
                 lstm_parameter.bidirectional_, lstm_parameter.zoneout_cell_, lstm_parameter.zoneout_hidden_,
                 lstm_parameter.input_row_align_, lstm_parameter.input_col_align_, lstm_parameter.state_row_align_,
                 lstm_parameter.state_col_align_, lstm_parameter.proj_col_align_, lstm_parameter.has_bias_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const DeQuantArg &de_quant_arg) {
  // this clusters is meaningless which will be supported in future
  CodeBaseStruct("DeQuantArg", name, de_quant_arg.scale, de_quant_arg.zeroPoint, de_quant_arg.var_corr,
                 de_quant_arg.mean_corr, "NULL", de_quant_arg.clusters_nums, de_quant_arg.bitNum);
}
void NNaclFp32Serializer::CodeStruct(const std::string &name, const SpliceParameter &splice_parameter) {
  CodeArray("splice_context", splice_parameter.context_, splice_parameter.context_dim_, false);
  CodeBaseStruct("SpliceParameter", name, splice_parameter.op_parameter_, splice_parameter.context_dim_,
                 splice_parameter.forward_indexes_dim_, "splice_context", nullptr, splice_parameter.output_dim_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const ExpStruct &exp_struct) {
  CodeBaseStruct("ExpParameter", "exp_param", reinterpret_cast<ExpParameter *>(exp_struct.base_.param_)->op_parameter_,
                 reinterpret_cast<ExpParameter *>(exp_struct.base_.param_)->base_,
                 reinterpret_cast<ExpParameter *>(exp_struct.base_.param_)->scale_,
                 reinterpret_cast<ExpParameter *>(exp_struct.base_.param_)->shift_);
  CodeBaseStruct("ExpStruct", name, "{}", exp_struct.in_scale_, exp_struct.out_scale_, exp_struct.element_num_);
  code << "    " << name << ".base_.param_ = &exp_param;\n";
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

void NNaclFp32Serializer::CodeStruct(const std::string &name, const SliceStruct &param) {
  CodeBaseStruct("SliceStruct", name, "{}", param.data_type_size_, ToString(param.begin_), ToString(param.size_),
                 ToString(param.shape_), ToString(param.end_), param.param_length_);
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

void NNaclFp32Serializer::CodeStruct(const std::string &name, const LayerNormComputeParam &op_param) {
  CodeBaseStruct<false>("LayerNormComputeParam", name, op_param.epsilon_, op_param.elementwise_affine_,
                        op_param.begin_norm_axis_, op_param.begin_params_axis_, op_param.norm_inner_size_,
                        op_param.norm_outer_size_, op_param.params_inner_size_, op_param.params_outer_size_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const BroadcastShapeInfo &param) {
  CodeBaseStruct<false>("BroadcastShapeInfo", name, ToString(param.input_shape_), param.input_shape_size_,
                        ToString(param.output_shape_), param.output_shape_size_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const CustomGruParameter &op_param) {
  CodeBaseStruct<false>("CustomGruParameter", name, op_param.op_parameter_, op_param.num_step, op_param.batch_size,
                        op_param.input_size, op_param.hidden_size);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const SlidingWindowParam &param) {
  CodeBaseStruct<false>("SlidingWindowParam", name, param.left_, param.right_, param.top_, param.bottom_,
                        param.c_block_, param.block_channel_, param.ic_align_, param.out_step_, param.out_h_step_,
                        param.out_c_step_, param.out_w_step_, param.out_block_step_, param.in_step_, param.in_h_step_,
                        param.in_sh_step_, param.in_sw_step_, param.in_kh_step_, param.in_kw_step_, param.kernel_step_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const UnstackParameter &param) {
  CodeBaseStruct<false>("UnstackParameter", name, param.op_parameter_, param.num_, param.axis_, param.pre_dims_,
                        param.axis_dim_, param.after_dims_);
}

void NNaclFp32Serializer::CodeStruct(const std::string &name, const FillStruct &param) {
  CodeBaseStruct<false>("FillParameter", name, "{}", param.thread_sz_count_, param.thread_sz_stride_, param.data_size_,
                        param.src_data_, param.out_ptr_, param.thread_count_);
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
