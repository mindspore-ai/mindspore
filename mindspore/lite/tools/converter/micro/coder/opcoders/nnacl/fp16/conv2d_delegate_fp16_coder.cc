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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/fp32/winograd_utils.h"
#include "nnacl/base/conv_common_base.h"
#include "coder/opcoders/nnacl/fp16/convolution_fp16_coder.h"
#include "coder/opcoders/nnacl/fp16/conv_depthwise_fp16_coder.h"

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

std::unique_ptr<OperatorCoder> CPUConvolutionFP16CoderSelect(const std::vector<Tensor *> &in_tensors,
                                                             const std::vector<Tensor *> &out_tensors,
                                                             const LiteGraph::Node *node, size_t node_index,
                                                             Target target, int schema_version) {
  return CPUOpCoderCreator<ConvolutionFP16Coder>(in_tensors, out_tensors, node, node_index, target, schema_version);
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
  return CPUOpCoderCreator<ConvolutionDepthwiseFP16Coder>(in_tensors, out_tensors, node, node_index, target,
                                                          schema_version);
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
