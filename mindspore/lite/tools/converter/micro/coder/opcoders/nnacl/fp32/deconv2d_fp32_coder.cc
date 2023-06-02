/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/nnacl/fp32/deconv2d_fp32_coder.h"
#include <memory>
#include <string>
#include <vector>
#include "coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.h"
#include "nnacl/fp32/winograd_utils.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "src/common/version_manager.h"
#include "coder/opcoders/nnacl/dequant/de_quant.h"

using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;
namespace mindspore::lite::micro::nnacl {
int DeConvolutionFP32Coder::InitRunBuf() {
  pack_output_size_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * output_plane_ * DataTypeSize(data_type_);
  packed_output_ = allocator_->Malloc(data_type_, pack_output_size_, kWorkspace);
  MS_CHECK_PTR(packed_output_);

  if (target_ == kARM32) {
    tmp_buffer_size_ = matmul_param_.row_4_ * matmul_param_.col_8_ * DataTypeSize(data_type_);
    pack_input_size_ = matmul_param_.row_4_ * matmul_param_.deep_ * DataTypeSize(data_type_);
  } else {
    tmp_buffer_size_ = matmul_param_.row_12_ * matmul_param_.col_8_ * DataTypeSize(data_type_);
    pack_input_size_ = matmul_param_.row_12_ * matmul_param_.deep_ * DataTypeSize(data_type_);
  }

  tmp_buffer_ = allocator_->Malloc(data_type_, tmp_buffer_size_, kWorkspace);
  MS_CHECK_PTR(tmp_buffer_);
  packed_input_ = allocator_->Malloc(data_type_, pack_input_size_, kWorkspace);
  MS_CHECK_PTR(packed_input_);
  return RET_OK;
}

int DeConvolutionFP32Coder::InitParam() {
  input_plane_ = conv_param_->input_h_ * conv_param_->input_w_;
  kernel_plane_ = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  output_plane_ = conv_param_->output_h_ * conv_param_->output_w_;

  matmul_param_.row_ = input_plane_;
  matmul_param_.deep_ = conv_param_->input_channel_;
  matmul_param_.col_ = conv_param_->output_channel_ * kernel_plane_;
  matmul_param_.row_12_ = UP_ROUND(matmul_param_.row_, C12NUM);
  matmul_param_.row_4_ = UP_ROUND(matmul_param_.row_, C4NUM);
  matmul_param_.row_16_ = UP_ROUND(matmul_param_.row_, C16NUM);
  matmul_param_.col_8_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * kernel_plane_;
  return RET_OK;
}

int DeConvolutionFP32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(Conv2DBaseCoder::Init(), "Conv2DBaseCoder::Init() failed.");
  MS_CHECK_RET_CODE(InitWeightBias(context), "Init weight bias failed.");
  return Resize();
}

int DeConvolutionFP32Coder::Resize() {
  MS_CHECK_RET_CODE(Conv2DBaseCoder::Init(), "init failed.");
  MS_CHECK_RET_CODE(InitParam(), "init param  failed.");
  MS_CHECK_RET_CODE(InitRunBuf(), "init run buffer failed.");
  return RET_OK;
}

int DeConvolutionFP32Coder::InitWeightBias(CoderContext *const context) {
  int kernel_h = filter_tensor_->Height();
  int kernel_w = filter_tensor_->Width();
  int in_channel = filter_tensor_->Batch();
  int out_channel = filter_tensor_->Channel();

  if (input_tensors_.size() == kInputSize2) {
    bias_data_size_ = UP_ROUND(out_channel, C4NUM) * DataTypeSize(data_type_);
    packed_bias_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
    MS_CHECK_PTR(packed_bias_);
  }

  int kernel_plane = kernel_h * kernel_w;
  int pack_weight_size = in_channel * kernel_plane;
  pack_weight_size_ = pack_weight_size * UP_ROUND(out_channel, C8NUM) * DataTypeSize(data_type_);
  packed_weight_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(packed_weight_);

  NNaclFp32Serializer init_code;

  size_t w_buf_size = 0;
  if (input_tensors_.size() == kInputSize2) {
    auto packed_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float *>(packed_bias_));
    init_code.CodeBufferOffsetExpression(packed_bias_, context->weight_name(), context->weight_offset_name(),
                                         context->weight_size_name(), bias_data_size_);
    std::string bias_tensor_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", packed_bias_str, bias_tensor_str, out_channel * DataTypeSize(data_type_));
    w_buf_size += bias_data_size_;
  }
  auto packed_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float *>(packed_weight_));
  init_code.CodeBufferOffsetExpression(packed_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), pack_weight_size_);
  w_buf_size += pack_weight_size_;
  init_code.CodeFunction("PackNHWCToC8HWN8Fp32", filter_tensor_, packed_weight_str, in_channel, kernel_plane,
                         out_channel);

  context->AppendInitWeightSizeCode(w_buf_size);
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

void DeConvolutionFP32Coder::CollectFilesForFunc(CoderContext *const context) {
  if (target_ == kARM32) {
    Collect(context, {}, {},
            {
              "MatmulFp32.S",
              "MatmulFp32Opt.S",
              "PreSum4x16Int8Peroc.S",
              "MatVecMulFp32.S",
              "PreSum4x16Int8Pert.S",
              "IndirectGemmInt16to32_8x4.S",
              "MatmulInt8.S",
              "MatmulFp32Opt12x4.S",
              "PostFuncBiasReluC8.S",
            });
  } else if (target_ == kARM64) {
    Collect(context, {}, {},
            {
              "BigMatmulFp32Opt.S",
              "MatmulFp32.S",
              "MatmulFp32Opt.S",
              "PreSum4x16Int8Peroc.S",
              "MatVecMulFp32.S",
              "PreSum4x16Int8Peroc.S",
              "PreSum4x16Int8Pert.S",
              "IndirectGemmInt16to32_8x4.S",
              "MatmulInt8.S",
              "PostFuncBiasReluC8.S",
            });
  }
  Collect(context,
          {
            "wrapper/fp32/deconvolution_fp32_wrapper.h",
            "nnacl/fp32/conv_common_fp32.h",
            "nnacl/pack.h",
            "nnacl/fp32/common_func_fp32.h",
            "nnacl/base/minimal_filtering_generator.h",
            "nnacl/fp32/matmul_fp32.h",
            "nnacl/conv_parameter.h",
            "nnacl/matmul_parameter.h",
            "wrapper/base/micro_parameter.h",
            "nnacl/op_base.h",
          },
          {
            "deconvolution_fp32_wrapper.c",
            "common_func.c",
            "common_func_fp32.c",
            "conv_common_fp32.c",
            "matmul_fp32.c",
            "pack_fp32.c",
            "deconv_fp32.c",
            "minimal_filtering_generator.c",
          });
}

int DeConvolutionFP32Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  NNaclFp32Serializer code;
  // call the op function
  auto packed_input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float *>(packed_input_));
  auto packed_output_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float *>(packed_output_));
  auto tmp_buffer_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float *>(tmp_buffer_));
  code.CodeFunction("memset", packed_input_str, "0", pack_input_size_);
  code.CodeFunction("memset", packed_output_str, "0", pack_output_size_);
  code.CodeFunction("memset", tmp_buffer_str, "0", tmp_buffer_size_);
  code.CodeStruct("conv_parameter", *conv_param_);
  code.CodeStruct("matmul_parameter", matmul_param_);

  std::string src_in_ptr_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string src_out_ptr_str = allocator_->GetRuntimeAddr(output_tensor_);

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    input_ptr_ = src_in_ptr_str + "+" + std::to_string(batch_index * input_plane_ * conv_param_->input_channel_);
    output_ptr_ = src_out_ptr_str + "+" + std::to_string(batch_index * output_plane_ * conv_param_->output_channel_);

    if (target_ == kARM32) {
      code.CodeFunction("RowMajor2Col4Major", input_ptr_, packed_input_str, matmul_param_.row_, matmul_param_.deep_);
    } else {
      code.CodeFunction("RowMajor2Col12Major", input_ptr_, packed_input_str, matmul_param_.row_, matmul_param_.deep_);
    }
    code.CodeBaseStruct("DeConvFp32Args", kRunArgs, packed_input_str, packed_weight_, packed_bias_, packed_output_str,
                        output_ptr_, tmp_buffer_str, "&matmul_parameter", "&conv_parameter");
    if (!support_parallel_) {
      code.CodeFunction("DeConvFp32Run", kRunArgsAddr, kDefaultTaskId, kLhsScale, kRhsScale);
    } else {
      code.CodeFunction(kParallelLaunch, "DeConvFp32Run", kRunArgsAddr, "conv_parameter.thread_num_");
    }
  }
  context->AppendCode(code.str());
  return RET_OK;
}
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Conv2dTransposeFusion,
                   CPUOpCoderCreator<DeConvolutionFP32Coder>);
}  // namespace mindspore::lite::micro::nnacl
