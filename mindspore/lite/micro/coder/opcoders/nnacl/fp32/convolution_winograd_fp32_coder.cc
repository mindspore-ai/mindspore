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
#include "coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.h"
#include <array>
#include "nnacl/base/minimal_filtering_generator.h"
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro::nnacl {
const std::array<std::string, 9> InputTransFuncList = {
  "", "", "", "", "InputTransform4x4Unit", "", "InputTransform6x6Unit", "", "InputTransform8x8Unit"};

const std::array<std::string, 4> OutputTransFuncList4 = {"", "", "OutputTransform4x2Unit", "OutputTransform4x3Unit"};

const std::array<std::string, 4> OutputTransFuncReluList4 = {"", "", "OutputTransform4x2ReluUnit",
                                                             "OutputTransform4x3ReluUnit"};

const std::array<std::string, 4> OutputTransFuncRelu6List4 = {"", "", "OutputTransform4x2Relu6Unit",
                                                              "OutputTransform4x3Relu6Unit"};
const std::array<std::string, 6> OutputTransFuncList6 = {
  "", "", "OutputTransform6x2Unit", "OutputTransform6x3Unit", "OutputTransform6x4Unit", "OutputTransform6x5Unit"};

const std::array<std::string, 8> OutputTransFuncReluList6 = {"",
                                                             "",
                                                             "OutputTransform6x2ReluUnit",
                                                             "OutputTransform6x3ReluUnit",
                                                             "OutputTransform6x4ReluUnit",
                                                             "OutputTransform6x5ReluUnit"};

const std::array<std::string, 8> OutputTransFuncRelu6List6 = {"",
                                                              "",
                                                              "OutputTransform6x2Relu6Unit",
                                                              "OutputTransform6x3Relu6Unit",
                                                              "OutputTransform6x4Relu6Unit",
                                                              "OutputTransform6x5Relu6Unit"};

const std::array<std::string, 8> OutputTransFuncList8 = {"",
                                                         "",
                                                         "OutputTransform8x2Unit",
                                                         "OutputTransform8x3Unit",
                                                         "OutputTransform8x4Unit",
                                                         "OutputTransform8x5Unit",
                                                         "OutputTransform8x6Unit",
                                                         "OutputTransform8x7Unit"};

const std::array<std::string, 8> OutputTransFuncReluList8 = {"",
                                                             "",
                                                             "OutputTransform8x2ReluUnit",
                                                             "OutputTransform8x3ReluUnit",
                                                             "OutputTransform8x4ReluUnit",
                                                             "OutputTransform8x5ReluUnit",
                                                             "OutputTransform8x6ReluUnit",
                                                             "OutputTransform8x7ReluUnit"};
const std::array<std::string, 8> OutputTransFuncRelu6List8 = {"",
                                                              "",
                                                              "OutputTransform8x2Relu6Unit",
                                                              "OutputTransform8x3Relu6Unit",
                                                              "OutputTransform8x4Relu6Unit",
                                                              "OutputTransform8x5Relu6Unit",
                                                              "OutputTransform8x6Relu6Unit",
                                                              "OutputTransform8x7Relu6Unit"};

int ConvolutionWinogradFP32Coder::WinogradFilterTransform(const float *weight_data, float *matrix_g,
                                                          const float *matrix_gt, int oc_block) {
  MS_CHECK_TRUE(oc_block, "Divide by zero!");
  return WinogradWeightTransform(weight_data, trans_weight_, matrix_g, matrix_gt, oc_block, input_unit_, kernel_unit_,
                                 conv_param_->input_channel_, conv_param_->output_channel_, true);
}

int ConvolutionWinogradFP32Coder::InitTmpBuffer() {
  int channel_out = conv_param_->output_channel_;
  int oc8 = UP_DIV(channel_out, C8NUM);
  int tile_num = C12NUM;
  tile_buffer_size_ = thread_num_ * tile_num * input_unit_ * input_unit_ * conv_param_->input_channel_ * sizeof(float);
  trans_input_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, tile_buffer_size_, kWorkspace));
  gemm_out_size_ = thread_num_ * tile_num * input_unit_ * input_unit_ * oc8 * C8NUM * sizeof(float);
  gemm_out_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, gemm_out_size_, kWorkspace));
  tmp_data_size_ = thread_num_ * C4NUM * input_unit_ * input_unit_ * sizeof(float);
  tmp_data_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, tmp_data_size_, kWorkspace));
  col_buffer_size_ = thread_num_ * tile_num * conv_param_->input_channel_ * sizeof(float);
  col_buffer_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, col_buffer_size_, kWorkspace));
  return RET_OK;
}

int ConvolutionWinogradFP32Coder::ReSize() {
  // malloc tmp buffer
  int ret = ConfigInputOutput();
  MS_CHECK_RET_CODE(ret, "ConfigInputOutput failed.");
  ret = InitTmpBuffer();
  MS_CHECK_RET_CODE(ret, "Init tmp buffer failed.");
  return RET_OK;
}

int ConvolutionWinogradFP32Coder::Prepare(CoderContext *const context) {
  int ret = Conv2DBaseCoder::Init();
  MS_CHECK_RET_CODE(ret, "convolution base coder init failed.");
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;
  ret = InitWeightBias();
  MS_CHECK_RET_CODE(ret, "Init weight bias failed.");
  return ReSize();
}  // namespace micro

int ConvolutionWinogradFP32Coder::InitWeightBias() {
  int in_channel = filter_tensor_->Channel();
  int out_channel = filter_tensor_->Batch();
  MS_CHECK_TRUE(in_channel > 0, "invalid in channel size");
  MS_CHECK_TRUE(out_channel > 0, "invalid out channel size");
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;

  int oc4 = UP_DIV(out_channel, C4NUM);
  const int oc_block = C8NUM;
  int oc_block_num = UP_DIV(out_channel, C8NUM);
  // init weight
  int trans_matrix_data_size = input_unit_ * input_unit_ * in_channel * oc_block_num * oc_block;
  trans_weight_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat32, trans_matrix_data_size * sizeof(float), kOfflinePackWeight));
  MS_CHECK_PTR(trans_weight_);
  int ret = memset_s(trans_weight_, trans_matrix_data_size * sizeof(float), 0, trans_matrix_data_size * sizeof(float));
  MS_CHECK_RET_CODE(ret, "memset_s failed!");
  float matrix_g[64];
  float matrix_gt[64];
  float matrix_a[64];
  float matrix_at[64];
  float matrix_b[64];
  float matrix_bt[64];
  float coef = 1.0f;
  if (input_unit_ == 8) {
    coef = 0.5f;
  }
  CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, matrix_g, matrix_gt, coef, output_unit_, kernel_unit_);

  auto out_channel_size = static_cast<size_t>(out_channel);
  auto weight_data = reinterpret_cast<float *>(filter_tensor_->MutableData());
  ret = WinogradFilterTransform(weight_data, matrix_g, matrix_gt, oc_block);
  MS_CHECK_RET_CODE(ret, "winograd filter transform failed!");
  // init bias
  int new_bias_ele_num = oc4 * C4NUM;
  auto new_bias_ele_size = static_cast<size_t>(new_bias_ele_num * sizeof(float));
  new_bias_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, new_bias_ele_size, kOfflinePackWeight));
  MS_CHECK_PTR(new_bias_);
  ret = memset_s(new_bias_, new_bias_ele_size, 0, new_bias_ele_size);
  MS_CHECK_RET_CODE(ret, "memset_s failed!");
  if (input_tensors_.size() == kInputSize2) {
    auto ori_bias_addr = reinterpret_cast<float *>(bias_tensor_->data_c());
    MS_CHECK_RET_CODE(memcpy_s(new_bias_, new_bias_ele_size, ori_bias_addr, out_channel_size * sizeof(float)),
                      "memcpy_s failed!");
  } else {
    MS_CHECK_RET_CODE(memset_s(new_bias_, new_bias_ele_size, 0, new_bias_ele_size), "memset_s failed!");
  }
  return RET_OK;
}

int ConvolutionWinogradFP32Coder::ConfigInputOutput() {
  in_func_ = GetInputTransFunc(input_unit_);
  MS_CHECK_TRUE(!in_func_.empty(), "Get input_trans_func failed.");
  out_func_ = GetOutputTransFunc(input_unit_, output_unit_, conv_param_->act_type_);
  MS_CHECK_TRUE(!out_func_.empty(), "Get output_trans_func_ failed.");
  return RET_OK;
}

std::string ConvolutionWinogradFP32Coder::GetInputTransFunc(int input_unit) {
  return InputTransFuncList.at(input_unit);
}

std::string ConvolutionWinogradFP32Coder::GetOutputTransFunc(int input_unit, int output_unit, ActType act_type) {
  std::string res;
  if (input_unit == 4 && output_unit < 4) {
    if (act_type == ActType_Relu) {
      return OutputTransFuncReluList4.at(output_unit);
    } else if (act_type == ActType_Relu6) {
      return OutputTransFuncRelu6List4.at(output_unit);
    } else {
      return OutputTransFuncList4.at(output_unit);
    }
  } else if (input_unit == 6 && output_unit < 6) {
    if (act_type == ActType_Relu) {
      return OutputTransFuncReluList6.at(output_unit);
    } else if (act_type == ActType_Relu6) {
      return OutputTransFuncRelu6List6.at(output_unit);
    } else {
      return OutputTransFuncList6.at(output_unit);
    }
  } else if (input_unit == 8 && output_unit < 8) {
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

int ConvolutionWinogradFP32Coder::DoCode(CoderContext *const context) {
  std::vector<std::string> asmFiles;
  if (target_ == kARM32A) {
    asmFiles = {
      "MatmulFp32.S", "MatmulFp32Opt.S", "PreSum4x16Int8Peroc.S", "PreSum4x16Int8Pert.S", "IndirectGemmInt16to32_8x4.S",
      "MatmulInt8.S"};
  } else if (target_ == kARM64) {
    asmFiles = {"MatmulFp32.S",          "MatmulFp32Opt.S",      "PreSum4x16Int8Peroc.S",       "MatVecMulFp32.S",
                "PreSum4x16Int8Peroc.S", "PreSum4x16Int8Pert.S", "IndirectGemmInt16to32_8x4.S", "MatmulInt8.S"};
  }
  Collect(
    context, {"nnacl/fp32/conv_winograd_fp32.h", "nnacl/common_func.h"},
    {"common_func.c", "conv_int8.c", "matmul_int8.c", "pack_fp32.c", "conv_winograd_fp32.c", "winograd_transform.c",
     "common_func_fp32.c", "fixed_point.c", "winograd_utils.c", "minimal_filtering_generator.c"},
    asmFiles);

  NNaclFp32Serializer code;
  // call the op function
  code.CodeFunction("memset", trans_input_, "0", tile_buffer_size_);
  code.CodeFunction("memset", gemm_out_, "0", gemm_out_size_);
  code.CodeFunction("memset", tmp_data_, "0", tmp_data_size_);
  code.CodeFunction("memset", col_buffer_, "0", col_buffer_size_);
  code << "\t\tfloat *tmp_buffer_address_list[4] = {" << allocator_->GetRuntimeAddr(trans_input_) << ", "
       << allocator_->GetRuntimeAddr(gemm_out_) << ", " << allocator_->GetRuntimeAddr(tmp_data_) << ", "
       << allocator_->GetRuntimeAddr(col_buffer_) << "};\n";
  code.CodeStruct("conv_parameter", *conv_param_);
  // code operator func
  int task_id = 0;
  code.CodeFunction("ConvWinogardFp32", input_tensor_, trans_weight_, new_bias_, output_tensor_,
                    "tmp_buffer_address_list", task_id, "&conv_parameter", in_func_, out_func_);
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
