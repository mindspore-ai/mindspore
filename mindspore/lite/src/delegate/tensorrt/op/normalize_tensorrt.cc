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

#include "src/delegate/tensorrt/op/normalize_tensorrt.h"
#include <memory>
#include "src/delegate/tensorrt/op/normalize_opt_plugin.h"

namespace mindspore::lite {
int NormalizeTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                                 const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE3 && in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != INPUT_SIZE3 && out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  auto norm_op = primitive->value_as_LayerNormFusion();
  CHECK_NULL_RETURN(norm_op);
  int being_norm_axis = norm_op->begin_norm_axis();
  being_norm_axis = being_norm_axis >= 0 ? being_norm_axis : in_tensors[0].Shape().size() + being_norm_axis;
  int begin_params_axis = norm_op->begin_params_axis();
  begin_params_axis = begin_params_axis >= 0 ? begin_params_axis : in_tensors[0].Shape().size() + begin_params_axis;
  if (begin_params_axis != being_norm_axis || begin_params_axis != in_tensors[0].Shape().size() - 1) {
    MS_LOG(ERROR) << "only support normalize on last one dim, being_norm_axis is " << being_norm_axis << " for "
                  << op_name_;
    return RET_ERROR;
  }
  axis_ = begin_params_axis;
  epsilon_ = norm_op->epsilon();
  return RET_OK;
}

int NormalizeTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  CHECK_NULL_RETURN(network);
  int ret = PreprocessInputs(network);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "preprocess input failed for " << op_name_;
    return ret;
  }
  return RunOptPlugin() ? RunAsOptPlugin(network) : RunAsTrtOps(network);
}

int NormalizeTensorRT::PreprocessInputs(nvinfer1::INetworkDefinition *network) {
  int ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], &norm_input_);
  if (ret != RET_OK || norm_input_.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim norm_input failed for " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors_.size() == BETA_INDEX + 1) {
    size_t expand_shape_size = in_tensors_[0].Shape().size();
    gamma_ = ConvertTensorWithExpandDims(network, in_tensors_[1], expand_shape_size, in_tensors_[1].Name());
    CHECK_NULL_RETURN(gamma_);
    beta_ =
      ConvertTensorWithExpandDims(network, in_tensors_[BETA_INDEX], expand_shape_size, in_tensors_[BETA_INDEX].Name());
    CHECK_NULL_RETURN(beta_);
  }
  return RET_OK;
}

int NormalizeTensorRT::RunAsOptPlugin(nvinfer1::INetworkDefinition *network) {
  auto plugin = std::make_shared<NormalizeOptPlugin>(op_name_, axis_, epsilon_);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "create NormalizeOptPlugin failed for " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *inputTensors[] = {norm_input_.trt_tensor_, gamma_, beta_};
  nvinfer1::IPluginV2Layer *norm_layer = network->addPluginV2(inputTensors, INPUT_SIZE3, *plugin);
  if (norm_layer == nullptr) {
    MS_LOG(ERROR) << "add norm opt plugin layer failed for " << op_name_;
    return RET_ERROR;
  }
  layer_ = norm_layer;
  AddInnerOutTensors(ITensorHelper{norm_layer->getOutput(0), norm_input_.format_, norm_input_.same_format_});
  return RET_OK;
}

int NormalizeTensorRT::RunAsTrtOps(nvinfer1::INetworkDefinition *network) {
  size_t axis = 1u << axis_;
  // first output, add later
  AddInnerOutTensors(ITensorHelper{nullptr, norm_input_.format_, norm_input_.same_format_});

  // mean
  auto mean = network->addReduce(*(norm_input_.trt_tensor_), nvinfer1::ReduceOperation::kAVG, axis, true)->getOutput(0);
  CHECK_NULL_RETURN(mean);
  if (out_tensors_.size() == INPUT_SIZE3) {
    AddInnerOutTensors(ITensorHelper{mean, norm_input_.format_, norm_input_.same_format_});
  }
  // x - mean
  auto sub_mean =
    network->addElementWise(*(norm_input_.trt_tensor_), *mean, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
  CHECK_NULL_RETURN(sub_mean);
  // (x - mean)^2
  auto const_two = ConvertScalarToITensor(network, in_tensors_[0].Shape().size(), &two_, DataType::kNumberTypeFloat32,
                                          op_name_ + "_two");
  CHECK_NULL_RETURN(const_two);
  auto pow = network->addElementWise(*sub_mean, *const_two, nvinfer1::ElementWiseOperation::kPOW)->getOutput(0);
  CHECK_NULL_RETURN(pow);
  // mean of (x - mean)^2
  auto var = network->addReduce(*pow, nvinfer1::ReduceOperation::kAVG, axis, true)->getOutput(0);
  CHECK_NULL_RETURN(var);
  if (out_tensors_.size() == INPUT_SIZE3) {
    AddInnerOutTensors(ITensorHelper{var, norm_input_.format_, norm_input_.same_format_});
  }

  // var + min epsilon
  auto const_epsilon = ConvertScalarToITensor(network, in_tensors_[0].Shape().size(), &epsilon_,
                                              DataType::kNumberTypeFloat32, op_name_ + "_epsilion");
  CHECK_NULL_RETURN(const_epsilon);
  auto var_epsilon = network->addElementWise(*var, *const_epsilon, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  CHECK_NULL_RETURN(var_epsilon);

  // standard deviation
  auto std_dev = network->addUnary(*var_epsilon, nvinfer1::UnaryOperation::kSQRT)->getOutput(0);
  CHECK_NULL_RETURN(std_dev);

  // sub_mean / std_dev
  auto norm_layer = network->addElementWise(*sub_mean, *std_dev, nvinfer1::ElementWiseOperation::kDIV);
  CHECK_NULL_RETURN(norm_layer);
  this->layer_ = norm_layer;
  auto norm = norm_layer->getOutput(0);
  CHECK_NULL_RETURN(norm);

  // scale with gamma and beta
  if (gamma_ != nullptr && beta_ != nullptr) {
    auto gamma_out = network->addElementWise(*norm, *gamma_, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    CHECK_NULL_RETURN(gamma_out);
    auto beta_out = network->addElementWise(*gamma_out, *beta_, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    CHECK_NULL_RETURN(beta_out);
    tensorrt_out_tensors_[0].trt_tensor_ = beta_out;
  } else {
    tensorrt_out_tensors_[0].trt_tensor_ = norm;
  }
  return RET_OK;
}

bool NormalizeTensorRT::RunOptPlugin() {
  if (out_tensors_.size() == 1 && in_tensors_.size() == INPUT_SIZE3 && axis_ == in_tensors_[0].Shape().size() - 1 &&
      in_tensors_[0].Shape()[axis_] < GET_THREADS) {
    MS_LOG(INFO) << op_name_ << " use opt plugin";
    return true;
  }
  return false;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_LayerNormFusion, NormalizeTensorRT)
}  // namespace mindspore::lite
