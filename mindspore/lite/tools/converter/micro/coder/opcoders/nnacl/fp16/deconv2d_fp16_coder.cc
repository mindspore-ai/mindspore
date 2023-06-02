/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/nnacl/fp16/deconv2d_fp16_coder.h"
#include <memory>
#include <string>
#include <vector>
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "src/common/version_manager.h"
#include "coder/opcoders/nnacl/dequant/de_quant.h"
#include "base/float16.h"
#include "opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore::lite::micro::nnacl {
int DeConvolutionFP16Coder::InitRunBuf() {
  pack_output_size_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * output_plane_ * DataTypeSize(data_type_);
  packed_output_ = allocator_->Malloc(data_type_, pack_output_size_, kWorkspace);
  MS_CHECK_PTR(packed_output_);

  tmp_buffer_size_ = matmul_param_.row_16_ * matmul_param_.col_8_ * DataTypeSize(data_type_);
  pack_input_size_ = matmul_param_.row_16_ * matmul_param_.deep_ * DataTypeSize(data_type_);
  tmp_buffer_ = allocator_->Malloc(data_type_, tmp_buffer_size_, kWorkspace);
  MS_CHECK_PTR(tmp_buffer_);
  packed_input_ = allocator_->Malloc(data_type_, pack_input_size_, kWorkspace);
  MS_CHECK_PTR(packed_input_);
  return RET_OK;
}

int DeConvolutionFP16Coder::InitWeightBias(CoderContext *const context) {
  int kernel_h = filter_tensor_->Height();
  int kernel_w = filter_tensor_->Width();
  int in_channel = filter_tensor_->Batch();
  int out_channel = filter_tensor_->Channel();

  bias_data_size_ = UP_ROUND(out_channel, C8NUM) * DataTypeSize(data_type_);
  packed_bias_ = allocator_->Malloc(data_type_, bias_data_size_, kOnlinePackWeight);
  MS_CHECK_PTR(packed_bias_);

  int kernel_plane = kernel_h * kernel_w;
  int pack_weight_size = in_channel * kernel_plane;
  pack_weight_size_ = pack_weight_size * UP_ROUND(out_channel, C8NUM) * DataTypeSize(data_type_);
  packed_weight_ = allocator_->Malloc(data_type_, pack_weight_size_, kOnlinePackWeight);
  MS_CHECK_PTR(packed_weight_);

  NNaclFp32Serializer init_code;
  size_t w_buf_size = 0;
  if (input_tensors_.size() == kInputSize2) {
    init_code.CodeBufferOffsetExpression(packed_bias_, context->weight_name(), context->weight_offset_name(),
                                         context->weight_size_name(), bias_data_size_);
    std::string bias_tensor_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", packed_bias_, bias_tensor_str, out_channel * DataTypeSize(data_type_));
    w_buf_size += bias_data_size_;
  }

  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_weight_));
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), pack_weight_size_);
  w_buf_size += pack_weight_size_;
  init_code.CodeFunction("PackNHWCFp16ToC8HWN8Fp16", filter_tensor_, packed_weight_str, in_channel, kernel_plane,
                         out_channel);
  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

void DeConvolutionFP16Coder::CollectFilesForFunc(CoderContext *const context) {
  if (target_ == kARM64) {
    Collect(context, {}, {},
            {
              "MatmulFp16.S",
              "MatmulFp16Opt.S",
              "MatVecMulFp16.S",
              "Matmul12X16Fp16.S",
              "MatmulBaseFp16Neon.S",
              "MatmulWinogradFp16.S",
              "PostFuncBiasReluC8Fp16.S",
            });
  } else if (target_ == kARM32) {
    Collect(context, {}, {},
            {
              "MatVecMulFp16.S",
              "Matmul12x8Fp16.S",
            });
  }
  Collect(context,
          {
            "nnacl/fp16/deconv_fp16.h",
            "nnacl/fp16/pack_fp16.h",
            "nnacl/fp16/matmul_fp16.h",
            "nnacl/fp16/common_func_fp16.h",
            "nnacl/base/minimal_filtering_generator.h",
            "nnacl/conv_parameter.h",
            "nnacl/common_func.h",
            "nnacl/matmul_parameter.h",
            "wrapper/base/micro_parameter.h",
            "nnacl/op_base.h",
          },
          {
            "common_func.c",
            "common_func_fp16.c",
            "matmul_fp16.c",
            "pack_fp16.c",
            "deconv_fp16.c",
            "minimal_filtering_generator.c",
          });
}

int DeConvolutionFP16Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);

  NNaclFp32Serializer code;
  auto packed_input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_input_));
  auto packed_output_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(packed_output_));
  auto tmp_buffer_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(tmp_buffer_));
  code.CodeFunction("memset", packed_input_str, "0", pack_input_size_);
  code.CodeFunction("memset", packed_output_str, "0", pack_output_size_);
  code.CodeFunction("memset", tmp_buffer_str, "0", tmp_buffer_size_);

  code << "  for (int batch_index = 0; batch_index < " << conv_param_->input_batch_ << "; batch_index++) {\n";
  int col_8 = UP_DIV(matmul_param_.col_8_, C8NUM);
  int col_stride = UP_DIV(col_8, 1) * C8NUM;
  int cur_stride = matmul_param_.col_8_ - kDefaultTaskId * col_stride;
  int current_oc = MSMIN(col_stride, cur_stride);
  if (current_oc > 0) {
    auto tmp_output_str = tmp_buffer_str + " + " + std::to_string(kDefaultTaskId * col_stride * matmul_param_.row_16_);
    auto tmp_weight_str = allocator_->GetRuntimeAddr(static_cast<float16 *>(packed_weight_)) + " + " +
                          std::to_string(kDefaultTaskId * col_stride * matmul_param_.deep_);
    code.CodeFunction("MatMulFp16", packed_input_str, tmp_weight_str, tmp_output_str, nullptr, ActType_No,
                      matmul_param_.deep_, matmul_param_.row_, current_oc, 0, OutType_C8);
  }

  int oc8 = UP_DIV(conv_param_->output_channel_, C8NUM);
  int oc_stride = UP_DIV(oc8, 1) * C8NUM;
  cur_stride = conv_param_->output_channel_ - kDefaultTaskId * oc_stride;
  int cur_res = MSMIN(oc_stride, cur_stride);
  if (cur_res > 0) {
    auto tmp_buf_str =
      tmp_buffer_str + " + " + std::to_string(kDefaultTaskId * oc_stride * kernel_plane_ * matmul_param_.row_16_);
    auto tmp_out_str = packed_output_str + " + " + std::to_string(kDefaultTaskId * oc_stride * output_plane_);
    auto tmp_bias_str = allocator_->GetRuntimeAddr(static_cast<float16 *>(packed_bias_)) + " + " +
                        std::to_string(kDefaultTaskId * oc_stride);
    output_ptr_ = allocator_->GetRuntimeAddr(output_tensor_) + " + batch_index * " +
                  std::to_string(output_plane_ * conv_param_->output_channel_ + kDefaultTaskId * oc_stride);
    code.CodeStruct("conv_parameter", *conv_param_);
    code.CodeFunction("DeConvPostFp16", tmp_buf_str, tmp_out_str, tmp_bias_str, output_ptr_, cur_res,
                      "&conv_parameter");
  }

  output_ptr_ = allocator_->GetRuntimeAddr(output_tensor_) + " + batch_index * " +
                std::to_string(output_plane_ * conv_param_->output_channel_);
  code.CodeFunction("PackNC8HW8ToNHWCFp16", packed_output_str, output_ptr_, 1,
                    conv_param_->output_w_ * conv_param_->output_h_, conv_param_->output_channel_);
  code << "  }\n";
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Conv2dTransposeFusion,
                   CPUOpCoderCreator<DeConvolutionFP16Coder>);
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Conv2dTransposeFusion,
                   CPUOpCoderCreator<DeConvolutionFP16Coder>);
}  // namespace mindspore::lite::micro::nnacl
