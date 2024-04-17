/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/fp16/conv2d_delegate_dynamic_fp16_coder.h"
#include "src/common/version_manager.h"
#include "src/common/tensor_util.h"
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/fp32/winograd_utils.h"
#include "nnacl/base/conv_common_base.h"
#include "nnacl/infer/conv2d_infer.h"
#include "coder/shape_info_container.h"
#include "coder/opcoders/nnacl/fp16/convolution_dynamic_fp16_coder.h"
#include "coder/opcoders/nnacl/fp16/convolution_1x1_dynamic_fp16_coder.h"

using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore::lite::micro::nnacl {
int ConvDelegateDynamicFP16Coder::Prepare(CoderContext *const context) {
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(input_tensors_[i]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "Input tensor data type is invalid");
  }
  for (size_t i = 0; i < output_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(output_tensors_[i]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "Output tensor data type is invalid");
  }
  // Update shape info of input and output
  SetInputOutputShapeInfo(reinterpret_cast<ConvParameter *>(parameter_), input_tensor_, output_tensor_);
  if (conv_coder_ == nullptr) {
    // need to select actual execute coder here
    conv_coder_ =
      CPUConvFP16DynamicCoderSelect(input_tensors_, output_tensors_, node_, node_index(), target_, schema_version_);
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
    conv_coder_->set_shape_info_container(shape_info_container_);
    conv_coder_->set_dynamic_mem_manager(dynamic_mem_manager_);
  }
  return conv_coder_->Prepare(context);
}

int ConvDelegateDynamicFP16Coder::DoCode(CoderContext *const context) { return conv_coder_->DoCode(context); }

void ConvDelegateDynamicFP16Coder::SetInputOutputShapeInfo(ConvParameter *conv_param, const lite::Tensor *input,
                                                           const lite::Tensor *output) {
  dynamic_param_.input_batch_ = shape_info_container_->GetTemplateShape(input_tensor_).at(0);
  conv_param->input_h_ = input->Height();
  conv_param->input_w_ = input->Width();
  conv_param->input_channel_ = input->Channel();
  dynamic_param_.output_batch_ = shape_info_container_->GetTemplateShape(output_tensor_).at(0);
  conv_param->output_h_ = output->Height();
  conv_param->output_w_ = output->Width();
  conv_param->output_channel_ = output->Channel();
}

std::unique_ptr<OperatorCoder> CPUConvFP16DynamicCoderSelect(const std::vector<lite::Tensor *> &in_tensors,
                                                             const std::vector<lite::Tensor *> &out_tensors,
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
  free(conv_param);
  std::unique_ptr<OperatorCoder> coder;
  if (kernel_h == 1 && kernel_w == 1) {
    MS_LOG(DEBUG) << "create Convolution1x1DynamicFP16CPUKernel";
    coder = CPUOpCoderCreator<Convolution1x1DynamicFP16Coder>(in_tensors, out_tensors, node, node_index, target,
                                                              schema_version);
  } else {
    MS_LOG(DEBUG) << "create ConvolutionDynamicFP16Coder";
    coder =
      CPUOpCoderCreator<ConvolutionDynamicFP16Coder>(in_tensors, out_tensors, node, node_index, target, schema_version);
  }
  return coder;
}

std::unique_ptr<OperatorCoder> CreateConvDelegateFp16(const std::vector<lite::Tensor *> &in_tensors,
                                                      const std::vector<lite::Tensor *> &out_tensors,
                                                      const LiteGraph::Node *node, size_t node_index, Target target,
                                                      int schema_version) {
  return CPUOpCoderCreator<ConvDelegateDynamicFP16Coder>(in_tensors, out_tensors, node, node_index, target,
                                                         schema_version);
}

std::unique_ptr<OperatorCoder> CPUConv2DFusionDynamicFP16CoderCreator(const std::vector<lite::Tensor *> &in_tensors,
                                                                      const std::vector<lite::Tensor *> &out_tensors,
                                                                      const LiteGraph::Node *node, size_t node_index,
                                                                      Target target, int schema_version) {
  const void *primitive = node->primitive_;
  if (primitive == nullptr) {
    return nullptr;
  }
  ParameterGen param_gen = PopulateRegistry::GetInstance()->GetParameterCreator(
    GetPrimitiveType(node->primitive_, schema_version), schema_version);
  MS_CHECK_PTR_RET_NULL(param_gen);
  auto conv_param = reinterpret_cast<ConvParameter *>(param_gen(node->primitive_));
  MS_CHECK_PTR_RET_NULL(conv_param);
  std::unique_ptr<OperatorCoder> coder;
  if (conv_param->group_ == 1) {
    coder = CreateConvDelegateFp16(in_tensors, out_tensors, node, node_index, target, schema_version);
  } else {
    // GroupConv
    MS_LOG(ERROR) << "currently, only support conv_param->group_ == 1 in dynamic coder scene";
    return nullptr;
  }
  return coder;
}

REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Conv2DFusion,
                           CPUConv2DFusionDynamicFP16CoderCreator)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Conv2DFusion,
                           CPUConv2DFusionDynamicFP16CoderCreator)
}  // namespace mindspore::lite::micro::nnacl
