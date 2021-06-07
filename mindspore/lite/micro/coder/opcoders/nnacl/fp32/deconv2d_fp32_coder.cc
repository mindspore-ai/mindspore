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
  pack_output_size_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * output_plane_ * sizeof(float);
  packed_output_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, pack_output_size_, kWorkspace));
  MS_CHECK_PTR(packed_output_);

  if (target_ == kARM32A) {
    tmp_buffer_size_ = matmul_param_.row_4_ * matmul_param_.col_8_ * sizeof(float);
  } else {
    tmp_buffer_size_ = matmul_param_.row_12_ * matmul_param_.col_8_ * sizeof(float);
  }
  tmp_buffer_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, tmp_buffer_size_, kWorkspace));
  MS_CHECK_PTR(tmp_buffer_);

  if (target_ == kARM32A) {
    pack_input_size_ = matmul_param_.row_4_ * matmul_param_.deep_ * sizeof(float);
  } else {
    pack_input_size_ = matmul_param_.row_12_ * matmul_param_.deep_ * sizeof(float);
  }
  packed_input_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, pack_input_size_, kWorkspace));
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
  int in_channel = filter_tensor_->Channel();
  int out_channel = filter_tensor_->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;

  if (input_tensors_.size() == kInputSize2) {
    bias_data_size_ = UP_ROUND(out_channel, C4NUM) * sizeof(float);
    packed_bias_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
    MS_CHECK_PTR(packed_bias_);
  }

  int kernel_plane = kernel_h * kernel_w;
  int pack_weight_size = in_channel * kernel_plane;
  pack_weight_size_ = pack_weight_size * UP_ROUND(out_channel, C8NUM) * sizeof(float);

  packed_weight_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(packed_weight_);

  NNaclFp32Serializer init_code;
  if (input_tensors_.size() == kInputSize2) {
    init_code.CodeMallocExpression(packed_bias_, bias_data_size_);
    init_code.CodeFunction("memset", packed_bias_, 0, pack_weight_size_);
    init_code.CodeFunction("memcpy", packed_bias_, bias_tensor_, out_channel * sizeof(float));
  }

  init_code.CodeMallocExpression(packed_weight_, pack_weight_size_);
  init_code.CodeFunction("memset", packed_weight_, 0, pack_weight_size_);
  init_code.CodeFunction("PackNHWCToC8HWN8Fp32", filter_tensor_, packed_weight_, in_channel, kernel_plane, out_channel);

  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int DeConvolutionFP32Coder::DoCode(CoderContext *const context) {
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
            "nnacl/op_base.h",
          },
          {
            "deconvolution_fp32_wrapper.c",
            "common_func.c",
            "conv_common_fp32.c",
            "matmul_fp32.c",
            "pack_fp32.c",
            "deconv_fp32.c",
            "minimal_filtering_generator.c",
          });
  if (target_ == kARM32A) {
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
            });
  } else if (target_ == kARM64) {
    Collect(context, {}, {},
            {
              "MatmulFp32.S",
              "MatmulFp32Opt.S",
              "PreSum4x16Int8Peroc.S",
              "MatVecMulFp32.S",
              "PreSum4x16Int8Peroc.S",
              "PreSum4x16Int8Pert.S",
              "IndirectGemmInt16to32_8x4.S",
              "MatmulInt8.S",
            });
  }

  NNaclFp32Serializer code;
  // call the op function
  code.CodeFunction("memset", packed_input_, "0", pack_input_size_);
  code.CodeFunction("memset", packed_output_, "0", pack_output_size_);
  code.CodeFunction("memset", tmp_buffer_, "0", tmp_buffer_size_);
  code.CodeStruct("conv_parameter", *conv_param_);
  code.CodeStruct("matmul_parameter", matmul_param_);

  std::string src_in_ptr_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string src_out_ptr_str = allocator_->GetRuntimeAddr(output_tensor_);

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    input_ptr_ = src_in_ptr_str + "+" + std::to_string(batch_index * input_plane_ * conv_param_->input_channel_);
    output_ptr_ = src_out_ptr_str + "+" + std::to_string(batch_index * output_plane_ * conv_param_->output_channel_);

    if (target_ == kARM32A) {
      code.CodeFunction("RowMajor2Col4Major", input_ptr_, packed_input_, matmul_param_.row_, matmul_param_.deep_);
    } else {
      code.CodeFunction("RowMajor2Col12Major", input_ptr_, packed_input_, matmul_param_.row_, matmul_param_.deep_);
    }
    code.CodeBaseStruct("DeConvFp32Args", kRunArgs, packed_input_, packed_weight_, packed_bias_, packed_output_,
                        output_ptr_, tmp_buffer_, "&matmul_parameter", "&conv_parameter");
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
