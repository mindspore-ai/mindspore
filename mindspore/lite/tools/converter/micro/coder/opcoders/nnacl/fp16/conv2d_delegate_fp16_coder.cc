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

#include "coder/opcoders/nnacl/fp16/conv2d_delegate_fp16_coder.h"
#include "src/common/version_manager.h"
#include "src/common/tensor_util.h"
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/fp32/winograd_utils.h"
#include "nnacl/base/conv_common_base.h"
#include "nnacl/infer/conv2d_infer.h"
#include "coder/opcoders/nnacl/fp16/convolution_fp16_coder.h"
#include "coder/opcoders/nnacl/fp16/conv_depthwise_fp16_coder.h"
#include "coder/opcoders/nnacl/fp16/convolution_winograd_fp16_coder.h"
#include "coder/opcoders/nnacl/fp16/convolution_1x1_fp16_coder.h"
#include "coder/opcoders/nnacl/fp16/conv_depthwise_3x3_fp16_coder.h"
#include "coder/opcoders/nnacl/fp16/conv_depthwise_sw_fp16_coder.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore::lite::micro::nnacl {
int ConvDelegateFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  // Update shape info of input and output
  SetInputOutputShapeInfo(reinterpret_cast<ConvParameter *>(parameter_), input_tensor_, output_tensor_);
  if (conv_coder_ == nullptr) {
    // need to select actual execute coder here
    conv_coder_ =
      CPUConvolutionFP16CoderSelect(input_tensors_, output_tensors_, node_, node_index(), target_, schema_version_);
    MS_CHECK_PTR(conv_coder_);
    ConvParameter *op_parameter = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
    if (op_parameter == nullptr) {
      MS_LOG(ERROR) << "malloc ConvParameter failed.";
      return RET_ERROR;
    }
    if (memcpy_s(op_parameter, sizeof(ConvParameter), parameter_, sizeof(ConvParameter)) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      free(op_parameter);
      return RET_ERROR;
    }
    conv_coder_->set_type(GetPrimitiveType(node_->primitive_, schema_version_));
    conv_coder_->set_thread_num(thread_num_);
    conv_coder_->set_parameter(reinterpret_cast<OpParameter *>(op_parameter));
  }
  return conv_coder_->Prepare(context);
}

int ConvDelegateFP16Coder::DoCode(CoderContext *const context) { return conv_coder_->DoCode(context); }

int SelectOutUnit(const ConvParameter *conv_param) {
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_c = conv_param->input_channel_;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_c = conv_param->output_channel_;
  if (conv_param->op_parameter_.thread_num_ == 0) {
    return NNACL_PARAM_INVALID;
  }
  int unit2 = UP_DIV(out_w * out_h, C12NUM * conv_param->op_parameter_.thread_num_);
  int max_out_unit = static_cast<int>(sqrtf(static_cast<float>(unit2)));
  max_out_unit = max_out_unit < C8NUM ? max_out_unit : C8NUM;
  max_out_unit = max_out_unit > C2NUM ? max_out_unit : C2NUM;

  int unit = 0;
  float max_rate = 0.0f;
  float common_cost = static_cast<float>(out_h * out_w * in_c * out_c * kernel_h * kernel_w);

  for (int i = C2NUM; i <= max_out_unit; ++i) {
    int input_unit = i + kernel_w - 1;
    if (input_unit != C4NUM && input_unit != C6NUM && input_unit != C8NUM) {
      continue;
    }
    if ((i >= input_unit) || (i < C2NUM)) {
      continue;
    }
    float penalty = (static_cast<float>(input_unit) * input_unit) / (static_cast<float>(kernel_h) * kernel_w) * 0.12f;
    float wino_cost = ((2 + out_c) * static_cast<float>(input_unit) * input_unit * in_c +
                       (static_cast<float>(input_unit) + i) * i * out_c) *
                      UP_DIV(out_w, i) * UP_DIV(out_h, i);
    float reduce_rate = common_cost / wino_cost - penalty;
    if (reduce_rate > max_rate) {
      max_rate = reduce_rate;
      unit = i;
    }
  }
  if (max_rate < 1.0f) {
    unit = 1;
  }
  // If output_unit is 1, then it is conventional convolution
  return unit;
}

std::unique_ptr<OperatorCoder> CPUConvolutionFP16CoderSelect(const std::vector<Tensor *> &in_tensors,
                                                             const std::vector<Tensor *> &out_tensors,
                                                             const LiteGraph::Node *node, size_t node_index,
                                                             Target target, int schema_version) {
  const void *primitive = node->primitive_;
  if (primitive == nullptr) {
    return nullptr;
  }
  ParameterGen paramGen = PopulateRegistry::GetInstance()->GetParameterCreator(
    GetPrimitiveType(node->primitive_, schema_version), schema_version);
  MS_CHECK_PTR_RET_NULL(paramGen);
  auto conv_param = reinterpret_cast<ConvParameter *>(paramGen(node->primitive_));
  MS_CHECK_PTR_RET_NULL(conv_param);
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  conv_param->input_h_ = in_tensors.at(kInputIndex)->Height();
  conv_param->input_w_ = in_tensors.at(kInputIndex)->Width();
  conv_param->input_channel_ = in_tensors.at(kInputIndex)->Channel();
  conv_param->output_h_ = out_tensors.at(kOutputIndex)->Height();
  conv_param->output_w_ = out_tensors.at(kOutputIndex)->Width();
  conv_param->output_channel_ = out_tensors.at(kOutputIndex)->Channel();
  conv_param->op_parameter_.thread_num_ = 1;
  int out_unit = 0;
  bool use_winograd = (conv_param->kernel_h_ != C1NUM && conv_param->kernel_w_ != C1NUM &&
                       conv_param->kernel_w_ == conv_param->kernel_h_ && conv_param->dilation_h_ == C1NUM &&
                       conv_param->dilation_w_ == C1NUM && conv_param->stride_h_ == C1NUM &&
                       conv_param->stride_w_ == C1NUM && conv_param->input_channel_ != C1NUM);
  if (use_winograd) {
    out_unit = SelectOutUnit(conv_param);
    use_winograd = (out_unit > 1);
  }
  free(conv_param);
  std::unique_ptr<OperatorCoder> coder;
  if (kernel_h == 1 && kernel_w == 1) {
    MS_LOG(DEBUG) << "create Convolution1x1FP16CPUKernel";
    coder =
      CPUOpCoderCreator<Convolution1x1FP16Coder>(in_tensors, out_tensors, node, node_index, target, schema_version);
  } else if (use_winograd) {
    MS_LOG(DEBUG) << "create Conv2DWinogradFP16Coder";
    coder = std::make_unique<ConvolutionWinogradFP16Coder>(in_tensors, out_tensors, node, node_index, target, out_unit);
  } else {
    MS_LOG(DEBUG) << "create ConvolutionFP16Coder";
    coder = CPUOpCoderCreator<ConvolutionFP16Coder>(in_tensors, out_tensors, node, node_index, target, schema_version);
  }
  return coder;
}

std::unique_ptr<OperatorCoder> CreateDelegateFp16Conv(const std::vector<Tensor *> &in_tensors,
                                                      const std::vector<Tensor *> &out_tensors,
                                                      const LiteGraph::Node *node, size_t node_index, Target target,
                                                      int schema_version) {
  return CPUOpCoderCreator<ConvDelegateFP16Coder>(in_tensors, out_tensors, node, node_index, target, schema_version);
}

std::unique_ptr<OperatorCoder> CPUConvDwFp16CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                         const std::vector<Tensor *> &out_tensors,
                                                         const LiteGraph::Node *node, size_t node_index, Target target,
                                                         int schema_version) {
  const void *primitive = node->primitive_;
  if (primitive == nullptr) {
    return nullptr;
  }
  ParameterGen paramGen = PopulateRegistry::GetInstance()->GetParameterCreator(
    GetPrimitiveType(node->primitive_, schema_version), schema_version);
  MS_CHECK_PTR_RET_NULL(paramGen);
  auto conv_param = reinterpret_cast<ConvParameter *>(paramGen(node->primitive_));
  MS_CHECK_PTR_RET_NULL(conv_param);
  std::unique_ptr<OperatorCoder> coder;
  conv_param->input_h_ = in_tensors.at(kInputIndex)->Height();
  conv_param->input_w_ = in_tensors.at(kInputIndex)->Width();
  conv_param->input_channel_ = in_tensors.at(kInputIndex)->Channel();
  conv_param->output_h_ = out_tensors.at(kOutputIndex)->Height();
  conv_param->output_w_ = out_tensors.at(kOutputIndex)->Width();
  conv_param->output_channel_ = out_tensors.at(kOutputIndex)->Channel();
  conv_param->op_parameter_.thread_num_ = 1;

  if (target == kARM64 || target == kARM32) {
    std::vector<TensorC *> in_tensor_c;
    std::vector<TensorC *> out_tensor_c;
    GenerateInTensorC(in_tensors, &in_tensor_c, NULL);
    GenerateOutTensorC(reinterpret_cast<const OpParameter *const>(conv_param), out_tensors, &out_tensor_c);
    Conv2dInferShape(reinterpret_cast<const TensorC *const *>(in_tensor_c.data()), in_tensor_c.size(),
                     reinterpret_cast<TensorC **>(out_tensor_c.data()), out_tensor_c.size(),
                     reinterpret_cast<OpParameter *>(conv_param));
    bool use_winograd =
      (conv_param->kernel_h_ == C3NUM && conv_param->kernel_w_ == C3NUM && conv_param->stride_w_ == C1NUM &&
       conv_param->stride_h_ == C1NUM && conv_param->dilation_h_ == C1NUM && conv_param->dilation_w_ == C1NUM &&
       conv_param->pad_u_ == C1NUM && conv_param->pad_d_ <= C1NUM && conv_param->pad_l_ == C1NUM &&
       conv_param->pad_r_ == C1NUM && conv_param->input_channel_ == conv_param->output_channel_ &&
       conv_param->output_w_ >= C4NUM && conv_param->output_h_ >= C4NUM);
    if (use_winograd) {
      MS_LOG(DEBUG) << "create ConvolutionDepthwise3x3FP16CPUKernel";
      coder = CPUOpCoderCreator<ConvolutionDepthwise3x3FP16Coder>(in_tensors, out_tensors, node, node_index, target,
                                                                  schema_version);
    }
    if (coder != nullptr) {
      free(conv_param);
      return coder;
    }
  }
  if (conv_param->input_channel_ < C32NUM) {
    MS_LOG(DEBUG) << "create ConvolutionDepthwiseSWFP16CPUKernel";
    coder = CPUOpCoderCreator<ConvolutionDepthwiseSWFP16Coder>(in_tensors, out_tensors, node, node_index, target,
                                                               schema_version);
  } else {
    MS_LOG(DEBUG) << "create ConvolutionDepthwiseFp16CPUKernel";
    coder = CPUOpCoderCreator<ConvolutionDepthwiseFP16Coder>(in_tensors, out_tensors, node, node_index, target,
                                                             schema_version);
  }
  free(conv_param);
  return coder;
}

std::unique_ptr<OperatorCoder> CPUConv2DFusionFP16CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                               const std::vector<Tensor *> &out_tensors,
                                                               const LiteGraph::Node *node, size_t node_index,
                                                               Target target, int schema_version) {
  const void *primitive = node->primitive_;
  if (primitive == nullptr) {
    return nullptr;
  }
  ParameterGen param_gen = PopulateRegistry::GetInstance()->GetParameterCreator(
    GetPrimitiveType(node->primitive_, schema_version), schema_version);
  if (param_gen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is null";
    return nullptr;
  }
  auto conv_param = reinterpret_cast<ConvParameter *>(param_gen(node->primitive_));
  std::unique_ptr<OperatorCoder> coder;
  if (conv_param->group_ == 1) {
    coder = CreateDelegateFp16Conv(in_tensors, out_tensors, node, node_index, target, schema_version);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    coder = CPUConvDwFp16CoderCreator(in_tensors, out_tensors, node, node_index, target, schema_version);
  } else {
    // GroupConv
    return nullptr;
  }
  return coder;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Conv2DFusion, CPUConv2DFusionFP16CoderCreator)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Conv2DFusion, CPUConv2DFusionFP16CoderCreator)
}  // namespace mindspore::lite::micro::nnacl
