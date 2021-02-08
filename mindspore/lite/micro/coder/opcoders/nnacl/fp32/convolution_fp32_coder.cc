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

#include "micro/coder/opcoders/nnacl/fp32/convolution_fp32_coder.h"
#include <memory>
#include <string>
#include <vector>
#include "micro/coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.h"
#include "nnacl/fp32/winograd_utils.h"
#include "src/ops/populate/populate_register.h"
#include "micro/coder/opcoders/file_collector.h"
#include "micro/coder/log.h"

using mindspore::schema::PrimitiveType_Conv2D;
namespace mindspore::lite::micro::nnacl {
int ConvolutionFP32Coder::InitTmpBuffer() {
  int in_channel = conv_param_->input_channel_;
  int uint_size;
  if (target_ == kARM32A) {
    uint_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * in_channel * C4NUM * thread_num_;
  } else {
    uint_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * in_channel * C12NUM * thread_num_;
  }
  packed_input_size_ = uint_size * sizeof(float);
  packed_input_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, packed_input_size_, kWorkspace));
  col_major_input_size_ = uint_size * sizeof(float);
  col_major_input_ =
    reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, col_major_input_size_, kWorkspace));
  return RET_OK;
}

int ConvolutionFP32Coder::Prepare(CoderContext *const context) {
  int ret = Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(ret, "Conv2DBaseCoder::Init() failed.");
  ret = InitWeightBias(context);
  MS_CHECK_RET_CODE(ret, "Init weight bias failed.");
  return Resize();
}

int ConvolutionFP32Coder::Resize() {
  int ret = Conv2DBaseCoder::CheckResizeValid();
  MS_CHECK_RET_CODE(ret, "Resize is invalid.");
  ret = Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(ret, "init failed.");
  ret = InitTmpBuffer();
  MS_CHECK_RET_CODE(ret, "init tmp buffer failed.");
  return RET_OK;
}

int ConvolutionFP32Coder::InitWeightBias(CoderContext *const context) {
  int kernel_h = filter_tensor_->Height();
  int kernel_w = filter_tensor_->Width();
  int in_channel = filter_tensor_->Channel();
  int out_channel = filter_tensor_->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  int kernel_plane = kernel_h * kernel_w;
  const int oc_block = C8NUM;
  int oc_block_num = UP_DIV(out_channel, C8NUM);
  int pack_weight_size = oc_block_num * oc_block * in_channel * kernel_plane;

  auto origin_weight = reinterpret_cast<float *>(filter_tensor_->MutableData());
  MS_CHECK_PTR(origin_weight);
  packed_weight_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat32, pack_weight_size * sizeof(float), kOnlinePackWeight));
  MS_CHECK_PTR(packed_weight_);
  auto out_channel_size = static_cast<size_t>(out_channel);

  NNaclFp32Serializer code;
  code.CodeMallocExpression(packed_weight_, pack_weight_size * sizeof(float));
  code.CodeFunction("memset", packed_weight_, 0, pack_weight_size * sizeof(float));
  code.CodeFunction("RowMajor2Col8Major", filter_tensor_, packed_weight_, out_channel_size, in_channel * kernel_plane);

  auto bias_data_size = static_cast<size_t>(oc_block_num * oc_block * sizeof(float));
  bias_data_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, bias_data_size, kOnlinePackWeight));
  MS_CHECK_PTR(bias_data_);
  if (input_tensors_.size() == kInputSize2) {
    code.CodeMallocExpression(bias_data_, bias_data_size);
    code.CodeFunction("memset", bias_data_, 0, bias_data_size);
    code.CodeFunction("memcpy", bias_data_, bias_tensor_, out_channel_size * sizeof(float));
  } else {
    return RET_ERROR;
  }
  context->AppendInitCode(code.str());
  return RET_OK;
}

int ConvolutionFP32Coder::DoCode(CoderContext *const context) {
  {
    std::vector<string> asmFiles;
    if (target_ == kARM32A) {
      asmFiles = {"MatmulFp32.S",
                  "MatmulFp32Opt.S",
                  "PreSum4x16Int8Peroc.S",
                  "PreSum4x16Int8Pert.S",
                  "IndirectGemmInt16to32_8x4.S",
                  "MatmulInt8.S"};
    } else if (target_ == kARM64) {
      asmFiles = {"MatmulFp32.S",          "MatmulFp32Opt.S",      "PreSum4x16Int8Peroc.S",       "MatVecMulFp32.S",
                  "PreSum4x16Int8Peroc.S", "PreSum4x16Int8Pert.S", "IndirectGemmInt16to32_8x4.S", "MatmulInt8.S"};
    }
    Collect(context,
            {"nnacl/kernel/fp32/conv_fp32_slim.h", "nnacl/fp32/matmul.h", "nnacl/conv_parameter.h", "nnacl/op_base.h"},
            {"common_func.c", "conv_fp32_slim.c", "matmul.c"}, asmFiles);
  }
  NNaclFp32Serializer code;
  // call the op function
  code.CodeFunction("memset", packed_input_, "0", packed_input_size_);
  code.CodeFunction("memset", col_major_input_, "0", col_major_input_size_);
  code.CodeStruct("conv_parameter", *conv_param_);
  int task_id = 0;
  code.CodeFunction("ConvFp32Slim", input_tensor_, packed_input_, packed_weight_, bias_data_, col_major_input_,
                    output_tensor_, task_id, "(ConvParameter *)&conv_parameter");

  context->AppendCode(code.str());
  return RET_OK;
}

std::unique_ptr<OperatorCoder> CPUConvolutionFP32CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                              const std::vector<Tensor *> &out_tensors,
                                                              const Model::Node *node, size_t node_index,
                                                              Target target) {
  std::vector<Tensor *> inputs = in_tensors;
  std::vector<Tensor *> outputs = out_tensors;
  auto primitive = node->primitive_;
  if (!primitive) {
    return nullptr;
  }
  OpParameter *parameter =
    PopulateRegistry::GetInstance()->GetParameterCreator((schema::PrimitiveType(primitive->Type())))(primitive);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: "
                  << schema::EnumNamePrimitiveType((schema::PrimitiveType)(primitive->Type()));
    return nullptr;
  }
  auto conv_param = reinterpret_cast<ConvParameter *>(parameter);
  bool use_winograd = false;
  int out_unit = 0;
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  conv_param->input_h_ = inputs.at(kInputIndex)->Height();
  conv_param->input_w_ = inputs.at(kInputIndex)->Width();
  conv_param->input_channel_ = inputs.at(kInputIndex)->Channel();
  conv_param->output_h_ = outputs.at(kOutputIndex)->Height();
  conv_param->output_w_ = outputs.at(kOutputIndex)->Width();
  conv_param->output_channel_ = outputs.at(kOutputIndex)->Channel();
  conv_param->op_parameter_.thread_num_ = 1;
  CheckIfUseWinograd(&use_winograd, &out_unit, conv_param);
  free(parameter);
  // weight de quant
  std::unique_ptr<OperatorCoder> coder;
  if (kernel_h == 1 && kernel_w == 1) {
    MS_LOG(DEBUG) << "create ConvolutionFP32Coder";
    coder = CPUOpCoderCreator<ConvolutionFP32Coder>(in_tensors, out_tensors, node, node_index, target);
  } else if (use_winograd) {
    MS_LOG(DEBUG) << "create Conv2DWinogradFP32Coder";
    coder = std::make_unique<ConvolutionWinogradFP32Coder>(in_tensors, out_tensors, node, node_index, target, out_unit);
  } else {
    MS_LOG(DEBUG) << "create ConvolutionFP32Coder";
    coder = CPUOpCoderCreator<ConvolutionFP32Coder>(in_tensors, out_tensors, node, node_index, target);
  }
  return coder;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Conv2D, CPUConvolutionFP32CoderCreator)
}  // namespace mindspore::lite::micro::nnacl
