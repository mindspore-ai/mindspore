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

#include "coder/opcoders/nnacl/fp32/conv2d_delegate_fp32_coder.h"
#include "src/common/version_manager.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/winograd_utils.h"
#include "coder/opcoders/nnacl/fp32/convolution_fp32_coder.h"
#include "coder/opcoders/nnacl/fp32/convolution_depthwise_fp32_coder.h"
#include "coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.h"
using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore::lite::micro::nnacl {

int ConvDelegateCoder::Prepare(CoderContext *const context) {
  // Update shape info of input and output
  SetInputOutputShapeInfo(reinterpret_cast<ConvParameter *>(parameter_), input_tensor_, output_tensor_);
  if (conv_coder_ == nullptr) {
    // need to select actual execute coder here
    conv_coder_ = CPUConvolutionFP32CoderSelect(input_tensors_, output_tensors_, node_, node_index(), target_);
    MS_CHECK_PTR(conv_coder_);
    const void *primitive = node_->primitive_;
    MS_CHECK_PTR(primitive);
    int primitive_type = GetPrimitiveType(node_->primitive_);
    int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
    ParameterGen parameter_gen =
      PopulateRegistry::GetInstance()->GetParameterCreator(GetPrimitiveType(node_->primitive_), schema_version);
    MS_CHECK_PTR(parameter_gen);
    OpParameter *op_parameter = parameter_gen(node_->primitive_);
    op_parameter->thread_num_ = thread_num_;
    conv_coder_->set_type(primitive_type);
    conv_coder_->set_thread_num(thread_num_);
    conv_coder_->set_parameter(op_parameter);
  }
  return conv_coder_->Prepare(context);
}

int ConvDelegateCoder::DoCode(CoderContext *const context) { return conv_coder_->DoCode(context); }

void SetInputOutputShapeInfo(ConvParameter *conv_param, const lite::Tensor *input, const lite::Tensor *output) {
  conv_param->input_batch_ = input->Batch();
  conv_param->input_h_ = input->Height();
  conv_param->input_w_ = input->Width();
  conv_param->input_channel_ = input->Channel();
  conv_param->output_batch_ = output->Batch();
  conv_param->output_h_ = output->Height();
  conv_param->output_w_ = output->Width();
  conv_param->output_channel_ = output->Channel();
}

std::unique_ptr<OperatorCoder> CPUConvolutionFP32CoderSelect(const std::vector<Tensor *> &in_tensors,
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
  bool use_winograd = CheckIfUseWinograd(&out_unit, conv_param);
  free(conv_param);
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

std::unique_ptr<OperatorCoder> CreateDelegateConv(const std::vector<Tensor *> &in_tensors,
                                                  const std::vector<Tensor *> &out_tensors, const Model::Node *node,
                                                  size_t node_index, Target target) {
  return CPUOpCoderCreator<ConvDelegateCoder>(in_tensors, out_tensors, node, node_index, target);
}

std::unique_ptr<OperatorCoder> CPUConvDwFp32CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                         const std::vector<Tensor *> &out_tensors,
                                                         const Model::Node *node, size_t node_index, Target target) {
  return CPUOpCoderCreator<ConvolutionDepthwiseFP32Coder>(in_tensors, out_tensors, node, node_index, target);
}

std::unique_ptr<OperatorCoder> CPUConv2DFusionFP32CoderCreator(const std::vector<Tensor *> &in_tensors,
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
    coder = CreateDelegateConv(in_tensors, out_tensors, node, node_index, target);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    coder = CPUConvDwFp32CoderCreator(in_tensors, out_tensors, node, node_index, target);
  } else {
    // GroupConv
    return nullptr;
  }
  return coder;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Conv2DFusion, CPUConv2DFusionFP32CoderCreator)
}  // namespace mindspore::lite::micro::nnacl
