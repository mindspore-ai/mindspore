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
#include "coder/opcoders/nnacl/fp16/convolution_winograd_fp16_coder.h"
#include <array>
#include "nnacl/base/minimal_filtering_generator.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/utils/common.h"
#include "base/float16.h"

namespace mindspore::lite::micro::nnacl {
const std::array<std::string, kNine> InputTransFp16FuncList = {
  "", "", "", "", "InputTransform4x4UnitFp16", "", "InputTransform6x6UnitFp16", "", "InputTransform8x8UnitFp16"};

const std::array<std::string, kNine> InputTransStepFp16FuncList = {
  "", "", "", "", "InputTransform4x4StepFp16", "", "InputTransform6x6StepFp16", "", "InputTransform8x8StepFp16"};

const std::array<std::string, kNine> InputTransPackFp16FuncList = {
  "", "", "", "", "InputTransform4x4Pack16Fp16", "", "InputTransform6x6Pack16Fp16", "", "InputTransform8x8Pack16Fp16"};

const std::array<std::string, kFour> OutputTransFp16FuncList4 = {"", "", "OutputTransform4x2UnitFp16",
                                                                 "OutputTransform4x3UnitFp16"};

const std::array<std::string, kFour> OutputTransFp16FuncReluList4 = {"", "", "OutputTransform4x2ReluUnitFp16",
                                                                     "OutputTransform4x3ReluUnitFp16"};

const std::array<std::string, kFour> OutputTransFp16FuncRelu6List4 = {"", "", "OutputTransform4x2Relu6UnitFp16",
                                                                      "OutputTransform4x3Relu6UnitFp16"};
const std::array<std::string, kSix> OutputTransFp16FuncList6 = {"",
                                                                "",
                                                                "OutputTransform6x2UnitFp16",
                                                                "OutputTransform6x3UnitFp16",
                                                                "OutputTransform6x4UnitFp16",
                                                                "OutputTransform6x5UnitFp16"};

const std::array<std::string, kEight> OutputTransFp16FuncReluList6 = {"",
                                                                      "",
                                                                      "OutputTransform6x2ReluUnitFp16",
                                                                      "OutputTransform6x3ReluUnitFp16",
                                                                      "OutputTransform6x4ReluUnitFp16",
                                                                      "OutputTransform6x5ReluUnitFp16"};

const std::array<std::string, kEight> OutputTransFp16FuncRelu6List6 = {"",
                                                                       "",
                                                                       "OutputTransform6x2Relu6UnitFp16",
                                                                       "OutputTransform6x3Relu6UnitFp16",
                                                                       "OutputTransform6x4Relu6UnitFp16",
                                                                       "OutputTransform6x5Relu6UnitFp16"};

const std::array<std::string, kEight> OutputTransFp16FuncList8 = {"",
                                                                  "",
                                                                  "OutputTransform8x2UnitFp16",
                                                                  "OutputTransform8x3UnitFp16",
                                                                  "OutputTransform8x4UnitFp16",
                                                                  "OutputTransform8x5UnitFp16",
                                                                  "OutputTransform8x6UnitFp16",
                                                                  "OutputTransform8x7UnitFp16"};

const std::array<std::string, kEight> OutputTransFp16FuncReluList8 = {"",
                                                                      "",
                                                                      "OutputTransform8x2ReluUnitFp16",
                                                                      "OutputTransform8x3ReluUnitFp16",
                                                                      "OutputTransform8x4ReluUnitFp16",
                                                                      "OutputTransform8x5ReluUnitFp16",
                                                                      "OutputTransform8x6ReluUnitFp16",
                                                                      "OutputTransform8x7ReluUnitFp16"};
const std::array<std::string, kEight> OutputTransFp16FuncRelu6List8 = {"",
                                                                       "",
                                                                       "OutputTransform8x2Relu6UnitFp16",
                                                                       "OutputTransform8x3Relu6UnitFp16",
                                                                       "OutputTransform8x4Relu6UnitFp16",
                                                                       "OutputTransform8x5Relu6UnitFp16",
                                                                       "OutputTransform8x6Relu6UnitFp16",
                                                                       "OutputTransform8x7Relu6UnitFp16"};

int ConvolutionWinogradFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  if (target_ == kARM64) {
    row_tile_ = C16NUM;
  }
  return ConvolutionWinogradFP32Coder::Prepare(context);
}

void ConvolutionWinogradFP16Coder::InitCodeOnline(CoderContext *const context) {
  NNaclFp32Serializer init_code;
  init_code.CodeBufferOffsetExpression(trans_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), trans_weight_size_);
  auto ori_weight_addr = allocator_->GetRuntimeAddr(filter_tensor_);
  init_code.CodeArray("matrix_g", matrix_g_.data(), k64);
  init_code.CodeArray("matrix_gt", matrix_gt_.data(), k64);
  auto trans_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(trans_weight_));
  init_code.CodeFunction("WinogradWeightTransformFp16", ori_weight_addr, trans_weight_str, "matrix_g", "matrix_gt",
                         col_tile_, input_unit_, kernel_unit_, conv_param_->input_channel_,
                         conv_param_->output_channel_, true);

  init_code.CodeBufferOffsetExpression(new_bias_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), packed_bias_size_);
  if (input_tensors_.size() == kInputSize2) {
    auto bias_str = allocator_->GetRuntimeAddr(bias_tensor_);
    init_code.CodeFunction("memcpy", new_bias_, bias_str, bias_tensor_->Size());
  } else {
    init_code.CodeFunction("memcpy", new_bias_, 0, packed_bias_size_);
  }
  context->AppendInitWeightSizeCode(trans_weight_size_ + packed_bias_size_);
  context->AppendInitCode(init_code.str());
}

int ConvolutionWinogradFP16Coder::InitTmpBuffer() {
  int channel_out = conv_param_->output_channel_;
  tile_buffer_size_ =
    thread_num_ * row_tile_ * input_unit_ * input_unit_ * conv_param_->input_channel_ * DataTypeSize(data_type_);
  trans_input_ = allocator_->Malloc(data_type_, tile_buffer_size_, kWorkspace);
  MS_CHECK_PTR(trans_input_);
  gemm_out_size_ =
    thread_num_ * row_tile_ * input_unit_ * input_unit_ * UP_ROUND(channel_out, C8NUM) * DataTypeSize(data_type_);
  gemm_out_ = allocator_->Malloc(data_type_, gemm_out_size_, kWorkspace);
  MS_CHECK_PTR(gemm_out_);
  tmp_data_size_ = thread_num_ * C8NUM * input_unit_ * input_unit_ * DataTypeSize(data_type_);
  tmp_data_ = allocator_->Malloc(data_type_, tmp_data_size_, kWorkspace);
  MS_CHECK_PTR(tmp_data_);
  col_buffer_size_ = thread_num_ * row_tile_ * conv_param_->input_channel_ * DataTypeSize(data_type_);
  col_buffer_ = allocator_->Malloc(data_type_, col_buffer_size_, kWorkspace);
  MS_CHECK_PTR(col_buffer_);
  return RET_OK;
}

int ConvolutionWinogradFP16Coder::ConfigInputOutput() {
  trans_func_str_.in_func_ = GetInputTransFunc(input_unit_);
  MS_CHECK_TRUE(!trans_func_str_.in_func_.empty(), "Get input_trans_func failed.");
  if (target_ == kARM64) {
    trans_func_str_.in_step_func_ = GetInputTransStepFunc(input_unit_);
    MS_CHECK_TRUE(!trans_func_str_.in_step_func_.empty(), "Get in_step_func_ failed.");
    trans_func_str_.in_pack_func_ = GetInputTransPackFunc(input_unit_);
    MS_CHECK_TRUE(!trans_func_str_.in_pack_func_.empty(), "Get in_pack_func_ failed.");
  }
  trans_func_str_.out_func_ = GetOutputTransFunc(input_unit_, output_unit_, conv_param_->act_type_);
  MS_CHECK_TRUE(!trans_func_str_.out_func_.empty(), "Get output_trans_func_ failed.");
  return RET_OK;
}

std::string ConvolutionWinogradFP16Coder::GetInputTransFunc(int input_unit) {
  MS_CHECK_TRUE_RET(input_unit_ >= 0 && input_unit_ < static_cast<int>(InputTransFp16FuncList.size()), std::string());
  return InputTransFp16FuncList.at(input_unit);
}

std::string ConvolutionWinogradFP16Coder::GetInputTransStepFunc(int input_unit) {
  MS_CHECK_TRUE_RET(input_unit_ >= 0 && input_unit_ < static_cast<int>(InputTransStepFp16FuncList.size()),
                    std::string());
  return InputTransStepFp16FuncList.at(input_unit);
}

std::string ConvolutionWinogradFP16Coder::GetInputTransPackFunc(int input_unit) {
  MS_CHECK_TRUE_RET(input_unit_ >= 0 && input_unit_ < static_cast<int>(InputTransPackFp16FuncList.size()),
                    std::string());
  return InputTransPackFp16FuncList.at(input_unit);
}

std::string ConvolutionWinogradFP16Coder::GetOutputTransFunc(int input_unit, int output_unit, ActType act_type) {
  std::string res;
  if (input_unit == DIMENSION_4D && output_unit < DIMENSION_4D) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList4.at(output_unit);
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List4.at(output_unit);
    } else {
      return OutputTransFp16FuncList4.at(output_unit);
    }
  } else if (input_unit == DIMENSION_6D && output_unit < DIMENSION_6D) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList6.at(output_unit);
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List6.at(output_unit);
    } else {
      return OutputTransFp16FuncList6.at(output_unit);
    }
  } else if (input_unit == DIMENSION_8D && output_unit < DIMENSION_8D) {
    if (act_type == ActType_Relu) {
      return OutputTransFp16FuncReluList8.at(output_unit);
    } else if (act_type == ActType_Relu6) {
      return OutputTransFp16FuncRelu6List8.at(output_unit);
    } else {
      return OutputTransFp16FuncList8.at(output_unit);
    }
  } else {
    return res;
  }
}

void ConvolutionWinogradFP16Coder::CollectFilesForFunc(CoderContext *const context) {
  Collect(context,
          {"nnacl/fp16/conv_fp16.h", "nnacl/fp16/winograd_utils_fp16.h",
           "nnacl/fp16/winograd_transform_fp16.h"
           "nnacl/base/minimal_filtering_generator.h"
           "nnacl/base/conv_common_base.h"},
          {
            "conv_fp16.c",
            "winograd_utils_fp16.c",
            "conv_common_base.c",
            "minimal_filtering_generator.c",
            "winograd_transform_fp16.c",
          });
}

int ConvolutionWinogradFP16Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  InitCodeOnline(context);

  NNaclFp32Serializer code;
  // call the op function
  code.CodeFunction("memset", trans_input_, "0", tile_buffer_size_);
  code.CodeFunction("memset", gemm_out_, "0", gemm_out_size_);
  code.CodeFunction("memset", tmp_data_, "0", tmp_data_size_);
  code.CodeFunction("memset", col_buffer_, "0", col_buffer_size_);
  code << "    float16_t *tmp_buffer_address_list[4] = {"
       << allocator_->GetRuntimeAddr(static_cast<float16 *>(trans_input_)) << ", "
       << allocator_->GetRuntimeAddr(static_cast<float16 *>(gemm_out_)) << ", "
       << allocator_->GetRuntimeAddr(static_cast<float16 *>(tmp_data_)) << ", "
       << allocator_->GetRuntimeAddr(static_cast<float16 *>(col_buffer_)) << "};\n";
  code.CodeStruct("conv_parameter", *conv_param_);
  std::string trans_func = "    TransFuncFp16Str trans_func = {" + trans_func_str_.in_func_ +
                           trans_func_str_.in_step_func_ + trans_func_str_.in_pack_func_ + trans_func_str_.out_func_ +
                           "};";

  // code operator func
  auto input_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensor_);
  auto output_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(output_tensor_);
  auto trans_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(trans_weight_));
  auto new_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float16 *>(new_bias_));
  code.CodeFunction("ConvWinogardFp16", input_str, trans_weight_str, new_bias_str, output_str,
                    "tmp_buffer_address_list", kDefaultTaskId, "&conv_parameter", trans_func);
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
