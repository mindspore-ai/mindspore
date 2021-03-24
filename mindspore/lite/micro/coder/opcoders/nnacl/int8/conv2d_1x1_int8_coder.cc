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

#include "coder/opcoders/nnacl/int8/conv2d_1x1_int8_coder.h"
#include <string>
#include <vector>
#include "securec/include/securec.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

namespace mindspore::lite::micro::nnacl {

int Conv2D1x1Int8Coder::Prepare(CoderContext *const context) {
  matmul_param_ = new (std::nothrow) MatMulParameter();
  MS_CHECK_PTR(matmul_param_);
  MS_CHECK_RET_CODE(Conv2DBaseCoder::Init(), "Init failed");
  MS_CHECK_RET_CODE(Conv2DBaseCoder::SetQuantParam(), "SetQuantParam failed");
  filter_peroc_ = (conv_param_->conv_quant_arg_.filter_arg_num_ != kPerTensor);
  if (filter_peroc_) {
    MS_CHECK_RET_CODE(InitFilterPeroc(), "InitFilterPeroc failed.");
  }
  CheckSupportOptimize();
  MS_CHECK_RET_CODE(InitWeightBias(context), "InitWeightBias failed");
  MS_CHECK_RET_CODE(InitParam(), "InitParam failed");
  MS_CHECK_RET_CODE(InitRunBuf(), "InitRunBuf failed");
  return RET_OK;
}

int Conv2D1x1Int8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {"wrapper/int8/conv1x1_init_int8_wrapper.h", "wrapper/int8/conv1x1_run_int8_wrapper.h", "nnacl/common_func.h",
           "nnacl/base/conv1x1_base.h", "nnacl/int8/matmul_int8.h", "nnacl/int8/pack_int8.h",
           "nnacl/int8/conv1x1_int8.h", "nnacl/errorcode.h"},
          {"common_func.c", "pack_int8.c", "conv1x1_int8.c", "matmul_int8.c", "fixed_point.c",
           "conv1x1_init_int8_wrapper.c", "conv1x1_run_int8_wrapper.c", "conv1x1_base.c"},
          {"MatmulInt8Opt.S"});

  nnacl::NNaclInt8Serializer code;

  code.CodeStruct("conv_param", *conv_param_);
  code.CodeStruct("matmul_param", *matmul_param_);

  code.CodeBaseStruct<false>("Conv1x1Args", kRunArgs, input_sum_, filter_zp_ptr_, left_shift_, right_shift_,
                             multiplier_, packed_weight_, bias_data_, packed_input_, nullptr, nullptr, 0, 0, 0, 0,
                             "&conv_param", "&matmul_param", matmul_func_, pre_trans_input_, "GetSupportOptFlag()",
                             filter_peroc_, false);

  code.CodeFunction("Conv1x1PreRun", kRunArgsAddr, gThreadNum);
  code << "for (int batch_index = 0; batch_index < " << conv_param_->input_batch_ << "; batch_index++) {\n";
  std::string src_in = allocator_->GetRuntimeAddr(input_tensor_) + " + batch_index * " +
                       std::to_string(conv_param_->input_h_ * conv_param_->input_w_ * conv_param_->input_channel_);
  std::string src_out = allocator_->GetRuntimeAddr(output_tensor_) + " + batch_index * " +
                        std::to_string(matmul_param_->row_ * matmul_param_->col_);
  code.CodeFunction("Pre1x1Trans", kRunArgsAddr, src_in, src_out);
  code << "if (args.parallel_by_oc_) {\n";
  /* input transpose and input sum */
  code << "if (GetSupportOptFlag()) {\n";
  if (support_parallel_) {
    code.CodeFunction(kParallelLaunch, gThreadPool, "OcOptPre", kRunArgsAddr, "args.thread_count_hw");
  } else {
    code.CodeFunction("OcOptPre", kRunArgsAddr, kDefaultTaskId);
  }
  code << "} else {\n";
  code << "RowMajor2Row16x4MajorInt8(args.input_ptr_, args.packed_input_, args.matmul_param_->row_, "
          "args.matmul_param_->deep_);\n";
  if (filter_peroc_) {
    code << "PackInputSum16x4PerLayer(args.packed_input_, args.input_sum_, 1, args.matmul_param_->row_4_, "
            "args.matmul_param_->deep_16_);\n";
  } else {
    code << "PackInputSum16x4PerLayer(args.packed_input_, "
            "args.input_sum_,args.conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_, "
            "args.matmul_param_->row_4_, args.matmul_param_->deep_16_);\n";
  }
  code << "}\n";
  /* matmul parallel by oc */
  code << "if (GetSupportOptFlag()) {\n";
  if (support_parallel_) {
    code.CodeFunction(kParallelLaunch, gThreadPool, "RunArm64OptOc", kRunArgsAddr, "args.thread_count_oc");
  } else {
    code.CodeFunction("RunArm64OptOc", kRunArgsAddr, kDefaultTaskId);
  }
  code << "} else {\n";
  if (support_parallel_) {
    code.CodeFunction(kParallelLaunch, gThreadPool, "RunArmOc", kRunArgsAddr, "args.thread_count_oc");
  } else {
    code.CodeFunction("RunArmOc", kRunArgsAddr, kDefaultTaskId);
  }
  code << "}\n";
  code << "} else {\n";
  /* matmul parallel by hw */
  code << "if (GetSupportOptFlag()) {\n";
  if (support_parallel_) {
    code.CodeFunction(kParallelLaunch, gThreadPool, "RunArm64OptHw", kRunArgsAddr, "args.thread_count_hw");
  } else {
    code.CodeFunction("RunArm64OptHw", kRunArgsAddr, kDefaultTaskId);
  }
  code << "} else {\n";
  if (support_parallel_) {
    code.CodeFunction(kParallelLaunch, gThreadPool, "RunArmHw", kRunArgsAddr, "args.thread_count_hw");
  } else {
    code.CodeFunction("RunArmHw", kRunArgsAddr, kDefaultTaskId);
  }
  code << "}\n";
  code << "}\n";
  code << "}\n";

  context->AppendCode(code.str());
  return RET_OK;
}

void Conv2D1x1Int8Coder::CheckSupportOptimize() {
  support_optimize_ = false;
  matmul_func_ = "MatMulInt8_4x16_r";
  if (target_ == kARM64) {
    matmul_func_ = "MatMulDpInt8_optimize_handler";
  }
}

int Conv2D1x1Int8Coder::InitWeightBias(CoderContext *const context) {
  int32_t input_channel = filter_tensor_->Channel();
  int32_t output_channel = filter_tensor_->Batch();
  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;

  nnacl::NNaclInt8Serializer code;

  packed_weight_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(packed_weight_);
  bias_data_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(bias_data_);

  std::string packed_weight_str = "(int8_t **)&" + allocator_->GetRuntimeAddr(packed_weight_);
  std::string bias_data_str = "(int32_t **)&" + allocator_->GetRuntimeAddr(bias_data_);
  std::string filter_zp_str = "";
  if (filter_peroc_) {
    filter_zp_str = allocator_->GetRuntimeAddr(filter_zp_ptr_);
  } else {
    MS_CHECK_PTR(conv_param_->conv_quant_arg_.filter_quant_args_);
    filter_zp_str = "filter_zp";
    code << "int32_t filter_zp[1] = {" << conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_ << "};\n";
  }

  if (target_ == kARM64) {
    code.CodeFunctionWithCheck("Conv1x1Init", filter_tensor_, bias_tensor_, filter_zp_str, input_channel,
                               output_channel, input_zp, "GetSupportOptFlag()", filter_peroc_, packed_weight_str,
                               bias_data_str);
  } else {
    code.CodeFunctionWithCheck("Conv1x1Init", filter_tensor_, bias_tensor_, filter_zp_str, input_channel,
                               output_channel, input_zp, support_optimize_, filter_peroc_, packed_weight_str,
                               bias_data_str);
  }

  context->AppendInitCode(code.str());
  return RET_OK;
}

int Conv2D1x1Int8Coder::InitFilterPeroc() {
  int32_t output_channel = filter_tensor_->Batch();
  int round_oc;
  if (target_ == kARM32A) {
    round_oc = UP_ROUND(output_channel, C2NUM);
  } else {
    round_oc = MSMAX(UP_ROUND(output_channel, C16NUM), UP_ROUND(output_channel, C4NUM));
  }

  MS_CHECK_TRUE(conv_quant_arg_->filter_arg_num_ == static_cast<size_t>(output_channel),
                "weight per channel quant param length is not equal to filter num, filter is not PerChannel");
  size_t output_size = output_channel * sizeof(int32_t);
  size_t oc_size = round_oc * sizeof(int32_t);

  /* filter zp */
  filter_zp_ptr_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, output_size, kOfflinePackWeight));
  MS_CHECK_PTR(filter_zp_ptr_);
  MS_CHECK_PTR(conv_param_->conv_quant_arg_.filter_quant_args_);
  for (int fi = 0; fi < output_channel; fi++) {
    filter_zp_ptr_[fi] = conv_param_->conv_quant_arg_.filter_quant_args_[fi].zp_;
  }

  /* left shift */
  left_shift_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, oc_size, kOfflinePackWeight));
  MS_CHECK_PTR(left_shift_);
  MS_CHECK_RET_CODE(memset_s(left_shift_, oc_size, 0, oc_size), "memset left_shift_ failed");
  MS_CHECK_RET_CODE(memcpy_s(left_shift_, oc_size, conv_param_->conv_quant_arg_.left_shift_, output_size),
                    "memcpy_s left_shift_ failed");

  /* right shift */
  right_shift_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, oc_size, kOfflinePackWeight));
  MS_CHECK_PTR(right_shift_);
  MS_CHECK_RET_CODE(memset_s(right_shift_, oc_size, 0, oc_size), "memset right_shift_ failed");
  MS_CHECK_RET_CODE(memcpy_s(right_shift_, oc_size, conv_param_->conv_quant_arg_.right_shift_, output_size),
                    "memcpy_s right_shift_ failed");
  /* multiplier */
  multiplier_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, oc_size, kOfflinePackWeight));
  MS_CHECK_PTR(multiplier_);
  MS_CHECK_RET_CODE(memset_s(multiplier_, oc_size, 0, oc_size), "memset multiplier_ failed");
  MS_CHECK_RET_CODE(memcpy_s(multiplier_, oc_size, conv_param_->conv_quant_arg_.quant_multiplier_, output_size),
                    "memcpy_s multiplier_ failed");

  return RET_OK;
}

int Conv2D1x1Int8Coder::InitParam() {
  pre_trans_input_ = (conv_param_->pad_u_ != 0 || conv_param_->pad_l_ != 0 || conv_param_->stride_h_ != 1 ||
                      conv_param_->stride_w_ != 1);

  matmul_param_->row_ = conv_param_->output_h_ * conv_param_->output_w_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_;
  matmul_param_->row_4_ = UP_ROUND(matmul_param_->row_, C4NUM);
  matmul_param_->deep_4_ = UP_ROUND(matmul_param_->deep_, C4NUM);
  matmul_param_->deep_16_ = UP_ROUND(matmul_param_->deep_, C16NUM);

  int row_pack_count = C4NUM;
  /* init input sum size */
  input_sum_size_ = UP_ROUND(matmul_param_->row_, row_pack_count);

  if (pre_trans_input_) {
    input_ptr_ = reinterpret_cast<int8_t *>(
      allocator_->Malloc(kNumberTypeInt8, matmul_param_->row_ * matmul_param_->deep_ * sizeof(int8_t), kWorkspace));
    MS_CHECK_PTR(input_ptr_);
  }

  return RET_OK;
}

int Conv2D1x1Int8Coder::InitRunBuf() {
  input_sum_ =
    reinterpret_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, input_sum_size_ * sizeof(int32_t), kWorkspace));
  MS_CHECK_PTR(input_sum_);

  size_t size = MSMAX(UP_ROUND(matmul_param_->row_, C8NUM) * UP_ROUND(matmul_param_->deep_, C4NUM),
                      UP_ROUND(matmul_param_->row_, C4NUM) * UP_ROUND(matmul_param_->deep_, C16NUM));

  packed_input_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, size * sizeof(int8_t), kWorkspace));
  MS_CHECK_PTR(packed_input_);
  return RET_OK;
}

}  // namespace mindspore::lite::micro::nnacl
