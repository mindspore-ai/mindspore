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
#include "coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.h"
#include <array>
#include "nnacl/base/minimal_filtering_generator.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/utils/common.h"

namespace mindspore::lite::micro::nnacl {
const std::array<std::string, kNine> InputTransFuncList = {
  "", "", "", "", "InputTransform4x4Unit", "", "InputTransform6x6Unit", "", "InputTransform8x8Unit"};

const std::array<std::string, kFour> OutputTransFuncList4 = {"", "", "OutputTransform4x2Unit",
                                                             "OutputTransform4x3Unit"};

const std::array<std::string, kFour> OutputTransFuncReluList4 = {"", "", "OutputTransform4x2ReluUnit",
                                                                 "OutputTransform4x3ReluUnit"};

const std::array<std::string, kFour> OutputTransFuncRelu6List4 = {"", "", "OutputTransform4x2Relu6Unit",
                                                                  "OutputTransform4x3Relu6Unit"};
const std::array<std::string, kSix> OutputTransFuncList6 = {
  "", "", "OutputTransform6x2Unit", "OutputTransform6x3Unit", "OutputTransform6x4Unit", "OutputTransform6x5Unit"};

const std::array<std::string, kEight> OutputTransFuncReluList6 = {"",
                                                                  "",
                                                                  "OutputTransform6x2ReluUnit",
                                                                  "OutputTransform6x3ReluUnit",
                                                                  "OutputTransform6x4ReluUnit",
                                                                  "OutputTransform6x5ReluUnit"};

const std::array<std::string, kEight> OutputTransFuncRelu6List6 = {"",
                                                                   "",
                                                                   "OutputTransform6x2Relu6Unit",
                                                                   "OutputTransform6x3Relu6Unit",
                                                                   "OutputTransform6x4Relu6Unit",
                                                                   "OutputTransform6x5Relu6Unit"};

const std::array<std::string, kEight> OutputTransFuncList8 = {"",
                                                              "",
                                                              "OutputTransform8x2Unit",
                                                              "OutputTransform8x3Unit",
                                                              "OutputTransform8x4Unit",
                                                              "OutputTransform8x5Unit",
                                                              "OutputTransform8x6Unit",
                                                              "OutputTransform8x7Unit"};

const std::array<std::string, kEight> OutputTransFuncReluList8 = {"",
                                                                  "",
                                                                  "OutputTransform8x2ReluUnit",
                                                                  "OutputTransform8x3ReluUnit",
                                                                  "OutputTransform8x4ReluUnit",
                                                                  "OutputTransform8x5ReluUnit",
                                                                  "OutputTransform8x6ReluUnit",
                                                                  "OutputTransform8x7ReluUnit"};
const std::array<std::string, kEight> OutputTransFuncRelu6List8 = {"",
                                                                   "",
                                                                   "OutputTransform8x2Relu6Unit",
                                                                   "OutputTransform8x3Relu6Unit",
                                                                   "OutputTransform8x4Relu6Unit",
                                                                   "OutputTransform8x5Relu6Unit",
                                                                   "OutputTransform8x6Relu6Unit",
                                                                   "OutputTransform8x7Relu6Unit"};

int ConvolutionWinogradFP32Coder::InitTmpBuffer() {
  int channel_out = conv_param_->output_channel_;
  int oc8 = UP_DIV(channel_out, C8NUM);
  tile_buffer_size_ =
    thread_num_ * row_tile_ * input_unit_ * input_unit_ * conv_param_->input_channel_ * DataTypeSize(data_type_);
  trans_input_ = allocator_->Malloc(data_type_, tile_buffer_size_, kWorkspace);
  gemm_out_size_ = thread_num_ * row_tile_ * input_unit_ * input_unit_ * oc8 * col_tile_ * DataTypeSize(data_type_);
  gemm_out_ = allocator_->Malloc(data_type_, gemm_out_size_, kWorkspace);
  tmp_data_size_ = thread_num_ * C4NUM * input_unit_ * input_unit_ * DataTypeSize(data_type_);
  tmp_data_ = allocator_->Malloc(data_type_, tmp_data_size_, kWorkspace);
  col_buffer_size_ = thread_num_ * row_tile_ * conv_param_->input_channel_ * DataTypeSize(data_type_);
  col_buffer_ = allocator_->Malloc(data_type_, col_buffer_size_, kWorkspace);
  return RET_OK;
}

int ConvolutionWinogradFP32Coder::Prepare(CoderContext *const context) {
  int ret = Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(ret, "convolution base coder init failed.");
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;
  MS_CHECK_RET_CODE(InitParameter(), "Winograd convolution do InitParameter failed");
  if (input_tensor_->data_type() == kNumberTypeFloat32) {
    is_weight_online_ = Configurator::GetInstance()->keep_original_weight();
  }
  if (is_weight_online_) {
    MS_CHECK_RET_CODE(InitWeightBiasOnline(), "Winograd convolution do InitWeightBiasOnline failed");
  } else {
    MS_CHECK_RET_CODE(InitWeightBiasOffline(), "Winograd convolution do InitWeightBiasOffline failed");
  }
  ret = ConfigInputOutput();
  MS_CHECK_RET_CODE(ret, "ConfigInputOutput failed.");
  ret = InitTmpBuffer();
  MS_CHECK_RET_CODE(ret, "Init tmp buffer failed.");
  return RET_OK;
}

int ConvolutionWinogradFP32Coder::InitParameter() {
  int in_channel = filter_tensor_->Channel();
  int out_channel = filter_tensor_->Batch();
  MS_CHECK_TRUE(in_channel > 0, "invalid in channel size");
  MS_CHECK_TRUE(out_channel > 0, "invalid out channel size");
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;

  int oc4 = UP_DIV(out_channel, C4NUM);
  int oc_block_num = UP_DIV(out_channel, col_tile_);
  // init weight
  trans_weight_size_ = input_unit_ * input_unit_ * in_channel * oc_block_num * col_tile_ * DataTypeSize(data_type_);
  packed_bias_size_ = oc4 * C4NUM * DataTypeSize(data_type_);
  matrix_g_.resize(k64);
  matrix_gt_.resize(k64);
  float matrix_a[k64];
  float matrix_at[k64];
  float matrix_b[k64];
  float matrix_bt[k64];
  float coef = 1.0f;
  if (input_unit_ == DIMENSION_8D) {
    coef = 0.5f;
  }
  auto ret = CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, matrix_g_.data(), matrix_gt_.data(), coef,
                            output_unit_, kernel_unit_);
  MS_CHECK_RET_CODE(ret, "CookToomFilter failed!");
  return RET_OK;
}

int ConvolutionWinogradFP32Coder::InitWeightBiasOffline() {
  trans_weight_ = allocator_->Malloc(data_type_, trans_weight_size_, kOfflinePackWeight);
  MS_CHECK_PTR(trans_weight_);
  int ret = memset_s(trans_weight_, trans_weight_size_, 0, trans_weight_size_);
  MS_CHECK_RET_CODE(ret, "memset_s failed!");
  auto weight_data = reinterpret_cast<float *>(filter_tensor_->data());
  MS_CHECK_PTR(weight_data);
  ret = WinogradWeightTransform(weight_data, reinterpret_cast<float *>(trans_weight_), matrix_g_.data(),
                                matrix_gt_.data(), C8NUM, input_unit_, kernel_unit_, conv_param_->input_channel_,
                                conv_param_->output_channel_, true);
  MS_CHECK_RET_CODE(ret, "winograd filter transform failed!");

  new_bias_ = allocator_->Malloc(data_type_, packed_bias_size_, kOfflinePackWeight);
  MS_CHECK_PTR(new_bias_);
  ret = memset_s(new_bias_, packed_bias_size_, 0, packed_bias_size_);
  MS_CHECK_RET_CODE(ret, "memset_s failed!");
  if (input_tensors_.size() == kInputSize2) {
    auto ori_bias_addr = reinterpret_cast<float *>(bias_tensor_->data());
    MS_CHECK_PTR(ori_bias_addr);
    MS_CHECK_RET_CODE(memcpy_s(new_bias_, packed_bias_size_, ori_bias_addr, bias_tensor_->Size()), "memcpy_s failed!");
  }
  return RET_OK;
}

int ConvolutionWinogradFP32Coder::InitWeightBiasOnline() {
  trans_weight_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(trans_weight_);
  new_bias_ = allocator_->Malloc(data_type_, kOnlineSize, kOnlinePackWeight);
  MS_CHECK_PTR(new_bias_);
  return RET_OK;
}

int ConvolutionWinogradFP32Coder::ConfigInputOutput() {
  trans_func_str_.in_func_ = GetInputTransFunc(input_unit_);
  MS_CHECK_TRUE(!trans_func_str_.in_func_.empty(), "Get input_trans_func failed.");
  trans_func_str_.out_func_ = GetOutputTransFunc(input_unit_, output_unit_, conv_param_->act_type_);
  MS_CHECK_TRUE(!trans_func_str_.out_func_.empty(), "Get output_trans_func_ failed.");
  return RET_OK;
}

std::string ConvolutionWinogradFP32Coder::GetInputTransFunc(int input_unit) {
  MS_CHECK_TRUE_RET(input_unit_ >= 0 && input_unit_ < static_cast<int>(InputTransFuncList.size()), std::string());
  return InputTransFuncList.at(input_unit);
}

std::string ConvolutionWinogradFP32Coder::GetOutputTransFunc(int input_unit, int output_unit, ActType act_type) {
  std::string res;
  if (input_unit == DIMENSION_4D && output_unit < DIMENSION_4D) {
    if (act_type == ActType_Relu) {
      return OutputTransFuncReluList4.at(output_unit);
    } else if (act_type == ActType_Relu6) {
      return OutputTransFuncRelu6List4.at(output_unit);
    } else {
      return OutputTransFuncList4.at(output_unit);
    }
  } else if (input_unit == DIMENSION_6D && output_unit < DIMENSION_6D) {
    if (act_type == ActType_Relu) {
      return OutputTransFuncReluList6.at(output_unit);
    } else if (act_type == ActType_Relu6) {
      return OutputTransFuncRelu6List6.at(output_unit);
    } else {
      return OutputTransFuncList6.at(output_unit);
    }
  } else if (input_unit == DIMENSION_8D && output_unit < DIMENSION_8D) {
    if (act_type == ActType_Relu) {
      return OutputTransFuncReluList8.at(output_unit);
    } else if (act_type == ActType_Relu6) {
      return OutputTransFuncRelu6List8.at(output_unit);
    } else {
      return OutputTransFuncList8.at(output_unit);
    }
  } else {
    return res;
  }
}

void ConvolutionWinogradFP32Coder::InitCodeOnline(CoderContext *const context) {
  if (!Configurator::GetInstance()->keep_original_weight()) {
    return;
  }
  Collect(context,
          {
            "nnacl/base/minimal_filtering_generator.h",
            "nnacl/fp32/pack_fp32.h",
          },
          {"minimal_filtering_generator.c", "nnacl/fp32/pack_fp32.h"});
  NNaclFp32Serializer init_code;
  init_code.CodeBufferOffsetExpression(trans_weight_, context->weight_name(), context->weight_offset_name(),
                                       context->weight_size_name(), trans_weight_size_);
  auto filter_str = allocator_->GetRuntimeAddr(filter_tensor_);
  init_code.CodeArray("matrix_g", matrix_g_.data(), k64);
  init_code.CodeArray("matrix_gt", matrix_gt_.data(), k64);
  init_code.CodeFunction("WinogradWeightTransform", filter_str, reinterpret_cast<float *>(trans_weight_), "matrix_g",
                         "matrix_gt", C8NUM, input_unit_, kernel_unit_, conv_param_->input_channel_,
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

void ConvolutionWinogradFP32Coder::CollectFilesForFunc(CoderContext *const context) {
  if (target_ == kARM32) {
    Collect(context, {}, {},
            {
              "MatmulFp32.S",
              "MatmulFp32Opt.S",
              "MatVecMulFp32.S",
              "PreSum4x16Int8Peroc.S",
              "PreSum4x16Int8Pert.S",
              "IndirectGemmInt16to32_8x4.S",
              "MatmulInt8.S",
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
            });
  }
  Collect(context,
          {
            "nnacl/fp32/conv_winograd_fp32.h",
            "nnacl/common_func.h",
          },
          {
            "common_func.c",
            "conv_int8.c",
            "matmul_int8.c",
            "pack_fp32.c",
            "conv_winograd_fp32.c",
            "winograd_transform.c",
            "common_func_fp32.c",
            "fixed_point.c",
            "winograd_utils.c",
            "conv_common_base.c",
            "minimal_filtering_generator.c",
          });
  if (support_parallel_) {
    Collect(context, {"wrapper/fp32/conv_winograd_fp32_wrapper.h"}, {"conv_winograd_fp32_wrapper.c"});
  }
}

int ConvolutionWinogradFP32Coder::DoCode(CoderContext *const context) {
  CollectFilesForFunc(context);
  InitCodeOnline(context);
  NNaclFp32Serializer code;
  // call the op function
  code.CodeFunction("memset", trans_input_, "0", tile_buffer_size_);
  code.CodeFunction("memset", gemm_out_, "0", gemm_out_size_);
  code.CodeFunction("memset", tmp_data_, "0", tmp_data_size_);
  code.CodeFunction("memset", col_buffer_, "0", col_buffer_size_);
  code << "    float *tmp_buffer_address_list[4] = {" << allocator_->GetRuntimeAddr(static_cast<float *>(trans_input_))
       << ", " << allocator_->GetRuntimeAddr(static_cast<float *>(gemm_out_)) << ", "
       << allocator_->GetRuntimeAddr(static_cast<float *>(tmp_data_)) << ", "
       << allocator_->GetRuntimeAddr(static_cast<float *>(col_buffer_)) << "};\n";
  code.CodeStruct("conv_parameter", *conv_param_);
  code.CodeStruct("trans_func", trans_func_str_);
  auto trans_weight_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float *>(trans_weight_));
  auto new_bias_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(static_cast<float *>(new_bias_));
  if (support_parallel_) {
    code.CodeBaseStruct("ConvWinogradFp32Args", kRunArgs, input_tensor_, trans_weight_str, new_bias_str, output_tensor_,
                        "tmp_buffer_address_list", "&conv_parameter", "trans_func");
    code.CodeFunction(kParallelLaunch, "ConvWinogradFp32Run", kRunArgsAddr, "conv_parameter.thread_num_");
  } else {
    // code operator func
    code.CodeFunction("ConvWinogardFp32", input_tensor_, trans_weight_str, new_bias_str, output_tensor_,
                      "tmp_buffer_address_list", kDefaultTaskId, "&conv_parameter", "trans_func");
  }
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
