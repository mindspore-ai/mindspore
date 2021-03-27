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

#include "coder/opcoders/nnacl/int8/conv2d_int8_coder.h"
#include <memory>
#include <string>
#include <vector>
#include "securec/include/securec.h"
#include "coder/opcoders/nnacl/int8/conv2d_1x1_int8_coder.h"
#include "coder/opcoders/nnacl/int8/conv2d_3x3_int8_coder.h"
#include "coder/opcoders/nnacl/int8/convolution_depthwise_int8_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/ops/populate/populate_register.h"
#include "src/common/version_manager.h"
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore::lite::micro::nnacl {

int Conv2DINT8Coder::InitTmpBuffer(CoderContext *const context) {
  int kernel_plane = conv_param_->kernel_h_ * conv_param_->kernel_w_;
  int tmp_size;
  if (target_ == kARM64) {
    tmp_size = MSMAX(UP_ROUND(kernel_plane * conv_param_->input_channel_, C4NUM),
                     UP_ROUND(kernel_plane * conv_param_->input_channel_, C16NUM));
  } else {
    if (support_optimize_) {
      tmp_size = UP_ROUND(kernel_plane * conv_param_->input_channel_, C4NUM);
    } else {
      tmp_size = UP_ROUND(kernel_plane * conv_param_->input_channel_, C16NUM);
    }
  }
  // malloc packed input
  packed_input_size_ = tmp_size * thread_num_ * tile_num_ * sizeof(int8_t);
  packed_input_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, packed_input_size_, kWorkspace));
  MS_CHECK_PTR(packed_input_);
  matmul_packed_input_size_ = thread_num_ * tile_num_ * kernel_plane * conv_param_->input_channel_ * sizeof(int8_t);
  matmul_packed_input_ =
    static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, matmul_packed_input_size_, kWorkspace));
  MS_CHECK_PTR(matmul_packed_input_);
  return RET_OK;
}

void Conv2DINT8Coder::CheckSupportOptimize() {
  tile_num_ = 8;
  matmul_func_ = "NULL";

  switch (target_) {
    case kARM32A:
      support_optimize_ = false;
      tile_num_ = 4;
      matmul_func_ = "NULL";
      break;
    case kARM64:
      // check support_optimize at runtime
      matmul_func_ = "MatMulRInt8_optimize_handler";
      tile_num_ = 8;
      break;
    case kX86:
      support_optimize_ = true;
      tile_num_ = 8;
      break;
    default:
      MS_LOG(ERROR) << "target not supported";
      return;
  }
  conv_param_->tile_num_ = tile_num_;
}

int Conv2DINT8Coder::InitWeightBias(CoderContext *const context) {
  int32_t input_channel = filter_tensor_->Channel();
  int32_t output_channel = filter_tensor_->Batch();
  int32_t kernel_h = filter_tensor_->Height();
  int32_t kernel_w = filter_tensor_->Width();
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  auto output_channel_size = static_cast<size_t>(output_channel);
  auto output_channel_data_size = static_cast<size_t>(output_channel_size * sizeof(int32_t));

  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;
  filter_peroc_ = conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL;

  if (filter_peroc_) {
    filter_zp_ptr_ =
      static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, output_channel_data_size, kOfflinePackWeight));
    MS_CHECK_PTR(filter_zp_ptr_);
    MS_CHECK_RET_CODE(memset_s(filter_zp_ptr_, output_channel_data_size, 0, output_channel_data_size),
                      "memset_s filter_zp_ptr_addr failed.");
    for (int oc = 0; oc < output_channel; oc++) {
      filter_zp_ptr_[oc] = conv_param_->conv_quant_arg_.filter_quant_args_[oc].zp_;
    }
  }

  int up_round_oc;
  switch (target_) {
    case kARM32A:
      up_round_oc = UP_ROUND(output_channel, C2NUM);
      break;
    case kARM64:
      up_round_oc = MSMAX(UP_ROUND(output_channel, C8NUM), UP_ROUND(output_channel, C4NUM));
      break;
    case kX86:
      up_round_oc = UP_ROUND(output_channel, C8NUM);
      break;
    default:
      MS_LOG(ERROR) << "target not supported";
      return RET_ERROR;
  }

  if (filter_peroc_) {
    input_sum_size_ = up_round_oc * tile_num_ * thread_num_ * sizeof(int32_t);
  } else {
    input_sum_size_ = tile_num_ * thread_num_ * sizeof(int32_t);
  }
  input_sum_ =
    static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, static_cast<size_t>(input_sum_size_), kWorkspace));
  MS_CHECK_PTR(input_sum_);

  packed_weight_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(packed_weight_);
  bias_data_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, kOnlineSize, kOnlinePackWeight));
  MS_CHECK_PTR(bias_data_);
  std::string filter_zp_str = "";
  std::string packed_weight_str = "(int8_t **)&" + allocator_->GetRuntimeAddr(packed_weight_);
  std::string bias_data_str = "(int32_t **)&" + allocator_->GetRuntimeAddr(bias_data_);

  nnacl::NNaclInt8Serializer code;

  if (filter_peroc_) {
    filter_zp_str = allocator_->GetRuntimeAddr(filter_zp_ptr_);
  } else {
    filter_zp_str = "filter_zp";
    code << "int32_t filter_zp[1] = {" << conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_ << "};\n";
  }

  if (target_ == kARM64) {
    code.CodeFunctionWithCheck("ConvInit", filter_tensor_, bias_tensor_, filter_zp_str, kernel_h, kernel_w,
                               input_channel, output_channel, input_zp, filter_peroc_, "GetSupportOptFlag()",
                               packed_weight_str, bias_data_str);
  } else {
    code.CodeFunctionWithCheck("ConvInit", filter_tensor_, bias_tensor_, filter_zp_str, kernel_h, kernel_w,
                               input_channel, output_channel, input_zp, filter_peroc_, support_optimize_,
                               packed_weight_str, bias_data_str);
  }

  context->AppendInitCode(code.str());

  return RET_OK;
}

int Conv2DINT8Coder::Prepare(CoderContext *const context) {
  Conv2DBaseCoder::Init();
  CheckSupportOptimize();
  MS_CHECK_RET_CODE(SetQuantParam(), "Set quant param failed!");
  MS_CHECK_RET_CODE(InitWeightBias(context), "Init weight bias failed.");
  MS_CHECK_RET_CODE(Resize(), "Resize failed.");
  MS_CHECK_RET_CODE(InitTmpBuffer(context), "InitTmpBuffer failed.");
  return RET_OK;
}

int Conv2DINT8Coder::Resize() {
  MS_CHECK_RET_CODE(Conv2DBaseCoder::CheckResizeValid(), "Resize is invalid.");
  MS_CHECK_RET_CODE(Conv2DBaseCoder::Init(), "Conv2DBaseCoder init failed.");
  return RET_OK;
}

int Conv2DINT8Coder::DoCode(CoderContext *const context) {
  std::vector<std::string> asm_files;
  if (target_ == kARM32A) {
    asm_files = {"PreSum4x16Int8Peroc.S", "PreSum4x16Int8Pert.S", "MatmulInt8.S"};
  } else if (target_ == kARM64) {
    asm_files = {"PreSum4x16Int8Peroc.S", "PreSum4x16Int8Pert.S", "MatmulInt8.S", "MatmulDpInt8.S"};
  }
  Collect(context,
          {"nnacl/int8/conv_int8.h", "nnacl/common_func.h", "wrapper/int8/convolution_int8_wrapper.h",
           "wrapper/int8/conv_init_int8_wrapper.h", "wrapper/base/common_wrapper.h",
           "wrapper/base/optimize_handler_wrapper.h"},
          {"common_func.c", "pack_int8.c", "conv_int8.c", "winograd_transform.c", "matmul_int8.c", "fixed_point.c",
           "convolution_int8_wrapper.c", "conv_init_int8_wrapper.c", "common_wrapper.c", "optimize_handler_wrapper.c"},
          asm_files);
  // call the op function
  nnacl::NNaclInt8Serializer code;
  code.precision(kPrecision);
  code.CodeFunction("memset", packed_input_, 0, packed_input_size_);
  code.CodeFunction("memset", input_sum_, 0, input_sum_size_);
  code.CodeFunction("memset", matmul_packed_input_, 0, matmul_packed_input_size_);
  code.CodeStruct("conv_param", *conv_param_);

  if (target_ == kARM64) {
    code.CodeBaseStruct("ConvolutionInt8Args", kRunArgs, input_tensor_, packed_input_, matmul_packed_input_,
                        packed_weight_, bias_data_, output_tensor_, filter_zp_ptr_, input_sum_,
                        "(ConvParameter *)&conv_param", matmul_func_, "GetSupportOptFlag()");
  } else {
    code.CodeBaseStruct("ConvolutionInt8Args", kRunArgs, input_tensor_, packed_input_, matmul_packed_input_,
                        packed_weight_, bias_data_, output_tensor_, filter_zp_ptr_, input_sum_,
                        "(ConvParameter *)&conv_param", matmul_func_, support_optimize_);
  }

  if (support_parallel_) {
    code.CodeFunction(kParallelLaunch, gThreadPool, "ConvolutionInt8Run", kRunArgsAddr, gThreadNum);
  } else {
    code.CodeFunction("ConvolutionInt8Run", kRunArgsAddr, kDefaultTaskId);
  }
  context->AppendCode(code.str());
  return RET_OK;
}

std::unique_ptr<OperatorCoder> CPUConv2DINT8CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                         const std::vector<Tensor *> &out_tensors,
                                                         const Model::Node *node, size_t node_index, Target target) {
  const void *primitive = node->primitive_;
  if (primitive == nullptr) {
    return nullptr;
  }
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  ParameterGen paramGen =
    PopulateRegistry::GetInstance()->GetParameterCreator(GetPrimitiveType(node->primitive_), schema_version);
  if (paramGen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is null";
    return nullptr;
  }
  auto conv_param = reinterpret_cast<ConvParameter *>(paramGen(node->primitive_));
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  free(conv_param);
  std::unique_ptr<OperatorCoder> coder;
  if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    coder = CPUOpCoderCreator<Conv2D3x3Int8Coder>(in_tensors, out_tensors, node, node_index, target);
  } else if (kernel_h == 1 && kernel_w == 1) {
    coder = CPUOpCoderCreator<Conv2D1x1Int8Coder>(in_tensors, out_tensors, node, node_index, target);
  } else {
    coder = CPUOpCoderCreator<Conv2DINT8Coder>(in_tensors, out_tensors, node, node_index, target);
  }
  if (coder == nullptr) {
    MS_LOG(ERROR) << "create conv2d int8 coder failed";
    return nullptr;
  }
  return coder;
}

std::unique_ptr<OperatorCoder> CPUConv2DFusionINT8CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                               const std::vector<Tensor *> &out_tensors,
                                                               const Model::Node *node, size_t node_index,
                                                               Target target) {
  const void *primitive = node->primitive_;
  if (primitive == nullptr) {
    return nullptr;
  }
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  ParameterGen paramGen =
    PopulateRegistry::GetInstance()->GetParameterCreator(GetPrimitiveType(node->primitive_), schema_version);
  if (paramGen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is null";
    return nullptr;
  }
  auto conv_param = reinterpret_cast<ConvParameter *>(paramGen(node->primitive_));
  std::unique_ptr<OperatorCoder> coder;
  if (conv_param->group_ == 1) {
    coder = CPUConv2DINT8CoderCreator(in_tensors, out_tensors, node, node_index, target);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    coder = CPUOpCoderCreator<ConvolutionDepthwiseINT8Coder>(in_tensors, out_tensors, node, node_index, target);
  } else {
    // group conv
  }
  free(conv_param);
  if (coder == nullptr) {
    MS_LOG(ERROR) << "create conv2d int8 coder failed";
    return nullptr;
  }
  return coder;
}

REG_OPERATOR_CODER(kX86, kNumberTypeInt8, PrimitiveType_Conv2DFusion, CPUConv2DFusionINT8CoderCreator)
REG_OPERATOR_CODER(kARM32A, kNumberTypeInt8, PrimitiveType_Conv2DFusion, CPUConv2DFusionINT8CoderCreator)
REG_OPERATOR_CODER(kARM64, kNumberTypeInt8, PrimitiveType_Conv2DFusion, CPUConv2DFusionINT8CoderCreator)
}  // namespace mindspore::lite::micro::nnacl
