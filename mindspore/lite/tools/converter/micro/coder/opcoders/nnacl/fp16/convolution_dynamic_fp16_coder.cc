/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/fp16/convolution_dynamic_fp16_coder.h"
#include <algorithm>
#include "coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.h"
#include "nnacl/fp32/winograd_utils.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/coder_utils.h"
#include "base/float16.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore::lite::micro::nnacl {
int ConvolutionDynamicFP16Coder::Prepare(CoderContext *const context) {
  CHECK_LESS_RETURN(input_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(output_tensors_.size(), 1);
  if (target_ == kARM64) {
    row_tile_ = C16NUM;
  }
  conv_param_ = reinterpret_cast<ConvParameter *>(parameter_);
  MS_CHECK_PTR(conv_param_);
  dynamic_param_.input_batch_ = shape_info_container_->GetTemplateShape(input_tensor_)[0];
  conv_param_->input_h_ = input_tensor_->Height();
  conv_param_->input_w_ = input_tensor_->Width();
  conv_param_->input_channel_ = input_tensor_->Channel();
  dynamic_param_.output_batch_ = shape_info_container_->GetTemplateShape(output_tensor_)[0];
  conv_param_->output_h_ = output_tensor_->Height();
  conv_param_->output_w_ = output_tensor_->Width();
  conv_param_->output_channel_ = output_tensor_->Channel();
  conv_param_->thread_num_ = 1;
  MS_CHECK_RET_CODE(InitWeightBias(context), "Init weight bias failed.");
  MS_CHECK_RET_CODE(InitTmpBuffer(), "Init tmp buffer failed.");
  return RET_OK;
}

int ConvolutionDynamicFP16Coder::InitTmpBuffer() {
  int uint_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * row_tile_ *
                  conv_param_->thread_num_;
  packed_input_size_ = uint_size * DataTypeSize(data_type_);
  auto input_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  size_t scene_num = 0;
  for (auto &dim_template : input_shape) {
    auto dim_nums = shape_info_container_->GetRealNums(dim_template);
    MS_CHECK_TRUE_MSG(!dim_nums.empty(), RET_ERROR, "Dynamic shape's num must be greater than 0.");
    scene_num = std::max(scene_num, dim_nums.size());
  }
  for (size_t i = 0; i < scene_num; ++i) {
    packed_input_str_ = dynamic_mem_manager_->AllocWorkSpace(packed_input_size_ * C2NUM, i);
    MS_CHECK_TRUE_MSG(!packed_input_str_.empty(), RET_ERROR, "Convolution cannot alloc workspace.");
  }
  col_major_input_str_ = packed_input_str_ + " + " + std::to_string(packed_input_size_);
  return RET_OK;
}

int ConvolutionDynamicFP16Coder::InitWeightBias(CoderContext *const context) {
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(filter_tensor_);
  auto shape = filter_tensor_->shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(WARNING) << "The shape of weight tensor is not ready, the weight and bias would be inited in runtime.";
    return RET_OK;
  }
  int in_channel = filter_tensor_->Channel();
  int out_channel = filter_tensor_->Batch();
  MS_CHECK_TRUE_RET(in_channel > 0 && out_channel > 0, RET_ERROR);
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  int oc8 = UP_ROUND(out_channel, col_tile_);
  int kernel_plane = filter_tensor_->Height() * filter_tensor_->Width();
  pack_weight_size_ = oc8 * in_channel * kernel_plane * DataTypeSize(data_type_);
  // init weight
  packed_weight_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(packed_weight_);
  NNaclFp32Serializer init_code;
  std::string ori_weight_addr = allocator_->GetRuntimeAddr(filter_tensor_);
  size_t w_buf_size = 0;
  w_buf_size += pack_weight_size_;
  auto packed_weight_str = allocator_->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), pack_weight_size_);
  init_code.CodeFunction("RowMajor2Col8MajorFp16", ori_weight_addr, packed_weight_str, out_channel,
                         in_channel * kernel_plane, false);
  if (input_tensors_.size() == C3NUM) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    bias_data_ =
      allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, bias_tensor_->tensor_name() + "_online_pack");
    MS_CHECK_PTR(bias_data_);
  } else {
    bias_data_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight, node_->name_ + "_bias_online_pack");
    MS_CHECK_PTR(bias_data_);
  }
  auto bias_data_size = static_cast<size_t>(oc8 * DataTypeSize(data_type_));
  w_buf_size += bias_data_size;
  init_code.CodeBufferOffsetExpression(bias_data_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), bias_data_size);
  bias_data_str_ = allocator_->GetRuntimeAddr(bias_data_);
  if (input_tensors_.size() == C3NUM) {
    auto origin_bias_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", bias_data_str_, origin_bias_str, bias_tensor_->Size());
  } else {
    init_code.CodeFunction("memset", bias_data_str_, 0, bias_data_size);
  }
  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

void ConvolutionDynamicFP16Coder::CollectFilesForFunc(CoderContext *const context) {
  Collect(context, {}, {},
          {
            "MatmulFp16.S",
            "MatmulFp16Opt.S",
            "MatVecMulFp16.S",
            "Matmul12X16Fp16.S",
          });
  Collect(context,
          {
            "nnacl/fp16/matmul_fp16.h",
            "nnacl/conv_parameter.h",
            "nnacl/op_base.h",
            "nnacl/fp16/conv_fp16.h",
          },
          {
            "common_func.c",
            "matmul_fp16.c",
            "pack_fp16.c",
            "conv_fp16.c",
          });
}

int ConvolutionDynamicFP16Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  NNaclFp32Serializer code;
  // call the op function
  auto packed_weight_str = allocator_->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  auto input_str =
    "(float16_t *)(" + GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  auto output_str =
    "(float16_t *)(" + GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  code.CodeStruct("conv_parameter", *conv_param_, dynamic_param_);
  packed_input_str_ = "(float16_t *)(" + packed_input_str_ + ")";
  col_major_input_str_ = "(float16_t *)(" + col_major_input_str_ + ")";
  if (output_tensor_->format() == NC4HW4) {
    code.CodeFunction("ConvOutNc8hw8Fp16", input_str, packed_input_str_, packed_weight_str, bias_data_str_,
                      col_major_input_str_, output_str, kDefaultTaskId, "&conv_parameter");
  } else {
    code.CodeFunction("ConvFp16", input_str, packed_input_str_, packed_weight_str, bias_data_str_, col_major_input_str_,
                      output_str, kDefaultTaskId, "&conv_parameter");
  }
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
