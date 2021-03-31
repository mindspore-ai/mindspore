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

#include "coder/opcoders/nnacl/int8/deconvolution_int8_coder.h"
#include <vector>
#include "nnacl/int8/deconv_int8.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore::lite::micro::nnacl {

int DeconvolutionInt8Coder::Init(CoderContext *const context) {
  CheckSupportOptimize();
  MS_CHECK_RET_CODE(SetQuantParam(), "deconv int8 SetQuantParam error!");
  MS_CHECK_RET_CODE(Conv2DBaseCoder::Init(), "Conv2DBaseCoder SetQuantParam error!");
  MS_CHECK_RET_CODE(InitParam(), "deconv int8 InitParam error!");
  MS_CHECK_RET_CODE(InitBiasWeight(context), "deconv int8 InitBiasWeight error!");
  MS_CHECK_RET_CODE(InitData(context), "deconv int8 InitData error!");
  return RET_OK;
}

int DeconvolutionInt8Coder::Prepare(CoderContext *const context) {
  conv_param_->thread_num_ = thread_num_;
  conv_param_->op_parameter_.thread_num_ = thread_num_;
  thread_count_ = thread_num_;
  MS_CHECK_RET_CODE(Init(context), "deconv int8 Init error!");
  MS_CHECK_RET_CODE(InitRunBuf(context), "deconv int8 InitRunBuf error!");
  return 0;
}

void DeconvolutionInt8Coder::CheckSupportOptimize() {
  support_optimize_ = false;
  matmul_func_str_ = "NULL";
}

int DeconvolutionInt8Coder::InitParam() {
  matmul_param_ = new (std::nothrow) MatMulParameter();
  MS_CHECK_PTR(matmul_param_);
  matmul_param_->row_ = conv_param_->input_h_ * conv_param_->input_w_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_ * conv_param_->kernel_h_ * conv_param_->kernel_w_;

  /* optimize normal -> same data layout */
  int oc4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  thread_count_ = MSMIN(conv_param_->op_parameter_.thread_num_, oc4);
  MS_CHECK_TRUE(thread_count_ > 0, "thread_count_ <= 0");
  thread_stride_ = UP_DIV(oc4, thread_count_);
  return RET_OK;
}

int DeconvolutionInt8Coder::InitBiasWeight(CoderContext *const context) {
  MS_CHECK_TRUE(conv_param_->output_channel_ > 0, "invalid output_channel");
  int size = UP_ROUND(conv_param_->output_channel_, C4NUM) * sizeof(int32_t);
  bias_data_ = reinterpret_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, size, kOfflinePackWeight));
  MS_CHECK_PTR(bias_data_);
  MS_CHECK_RET_CODE(memset_s(bias_data_, size, 0, size), "memset_s new_bias_addr_ failed.");
  if (input_tensors_.size() == kInputSize2) {
    auto *ori_bias_addr = reinterpret_cast<int32_t *>(bias_tensor_->data_c());
    MS_CHECK_RET_CODE(memcpy_s(bias_data_, size, ori_bias_addr, bias_tensor_->Size()),
                      "memcpy_s new_bias_addr_ failed.");
  }

  size = UP_ROUND(conv_param_->output_channel_, C4NUM) * UP_ROUND(conv_param_->input_channel_, C16NUM) *
         conv_param_->kernel_w_ * conv_param_->kernel_h_ * sizeof(int8_t);
  weight_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, size, kOfflinePackWeight));
  MS_CHECK_PTR(weight_ptr_);
  MS_CHECK_RET_CODE(
    memset_s(weight_ptr_, size, static_cast<int8_t>(conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_), size),
    "memset_s weight_ptr_ failed.");
  DeConvWeightTransInt8(reinterpret_cast<int8_t *>(filter_tensor_->data_c()), weight_ptr_, conv_param_->input_channel_,
                        conv_param_->output_channel_, conv_param_->kernel_h_ * conv_param_->kernel_w_,
                        support_optimize_);

  size = UP_ROUND(conv_param_->output_channel_, C4NUM) * conv_param_->kernel_h_ * conv_param_->kernel_w_;
  weight_sum_ =
    reinterpret_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, size * sizeof(int32_t), kOfflinePackWeight));
  MS_CHECK_PTR(weight_sum_);
  MS_CHECK_RET_CODE(memset_s(weight_sum_, size * sizeof(int32_t), 0, size * sizeof(int32_t)),
                    "memset_s weight_sum_ failed.");
  DeConvPackWeightSum(weight_ptr_, weight_sum_, conv_param_->conv_quant_arg_.input_quant_args_[0].zp_,
                      conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_, UP_ROUND(matmul_param_->deep_, C16NUM),
                      size, support_optimize_);

  return RET_OK;
}

int DeconvolutionInt8Coder::InitData(CoderContext *const context) {
  input_ptr_size_ = UP_ROUND(conv_param_->input_h_ * conv_param_->input_w_, C4NUM) *
                    UP_ROUND(conv_param_->input_channel_, C16NUM) * sizeof(int8_t);
  input_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, input_ptr_size_, kWorkspace));
  return RET_OK;
}

int DeconvolutionInt8Coder::InitRunBuf(CoderContext *const context) {
  tmp_buffer_size_ = UP_ROUND(conv_param_->input_h_ * conv_param_->input_w_, C4NUM) *
                     UP_ROUND(conv_param_->output_channel_, C4NUM) * conv_param_->kernel_w_ * conv_param_->kernel_h_ *
                     sizeof(int32_t);
  tmp_buffer_ = reinterpret_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, tmp_buffer_size_, kWorkspace));

  tmp_output_size_ =
    UP_ROUND(conv_param_->output_channel_, C4NUM) * conv_param_->output_h_ * conv_param_->output_w_ * sizeof(int32_t);
  tmp_output_ = reinterpret_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, tmp_output_size_, kWorkspace));

  input_sum_size_ = UP_ROUND(matmul_param_->row_, C4NUM) * sizeof(int32_t);
  input_sum_ = reinterpret_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, input_sum_size_, kWorkspace));
  return RET_OK;
}

int DeconvolutionInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/int8/deconv.h"}, {"int8/deconv.c", "pack_int8.c", "quantization/fixed_point.c"});

  nnacl::NNaclInt8Serializer code;
  code.CodeFunction("memset", input_ptr_, 0, input_ptr_size_);
  code.CodeFunction("memset", tmp_buffer_, 0, tmp_buffer_size_);
  code.CodeFunction("memset", tmp_output_, 0, tmp_output_size_);
  code.CodeFunction("memset", input_sum_, 0, input_sum_size_);

  // define conv params
  code.CodeStruct("conv_param_", *conv_param_);

  MS_CHECK_TRUE(conv_param_->input_batch_ == 1, "batch number should be 1.");

  code.CodeFunction("RowMajor2Row16x4MajorInt8", input_tensor_, input_ptr_, matmul_param_->row_, matmul_param_->deep_);
  code.CodeFunction("DeConvPackInputSum", input_ptr_, input_sum_,
                    conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_, UP_ROUND(matmul_param_->row_, C4NUM),
                    UP_ROUND(matmul_param_->deep_, C16NUM), support_optimize_);

  int kernel_plane = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  int cur_oc = MSMIN(thread_stride_, UP_DIV(conv_param_->output_channel_, C8NUM));
  int cur_oc_res = MSMIN(thread_stride_ * C4NUM, conv_param_->output_channel_);

  MS_CHECK_TRUE(cur_oc > 0, "cur_oc should be greater than 0.");

  code.CodeFunction("DeConvInt8", input_ptr_, weight_ptr_, tmp_buffer_, weight_sum_, input_sum_,
                    UP_ROUND(matmul_param_->row_, C4NUM), cur_oc * C4NUM * kernel_plane,
                    UP_ROUND(matmul_param_->deep_, C16NUM), "&conv_param_", matmul_func_str_);

  code.CodeFunction("DeConvPostInt8", tmp_buffer_, bias_data_, tmp_output_, output_tensor_, cur_oc_res, "&conv_param_",
                    support_optimize_);
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Conv2dTransposeFusion,
                   CPUOpCoderCreator<DeconvolutionInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
