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

#include "src/extendrt/delegate/tensorrt/op/fullyconnected_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/extendrt/delegate/tensorrt/op/activation_tensorrt.h"
#include "ops/fusion/full_connection.h"

namespace mindspore::lite {
constexpr int BIAS_INDEX = 2;
int FullyConnectedTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                      const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int FullyConnectedTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto primitive = AsOps<ops::FullConnection>();
  CHECK_NULL_RETURN(primitive);
  if (primitive->HasAttr(ops::kActivationType)) {
    activation_ = primitive->get_activation_type();
  }
  int axis = primitive->get_axis();
  ITensorHelper fc_input;
  auto ret = PreprocessInputs(ctx, &fc_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreprocessInputs failed for " << op_name_;
    return ret;
  }
  auto kernel_weight = ConvertWeight(!in_tensors_[1].IsConst() ? in_tensors_[0] : in_tensors_[1]);
  nvinfer1::Weights bias_weight{};
  if (primitive->get_has_bias()) {
    bias_weight = ConvertWeight(in_tensors_[BIAS_INDEX]);
  }
  nvinfer1::IFullyConnectedLayer *fc_layer = ctx->network()->addFullyConnected(
    *(fc_input.trt_tensor_), in_tensors_[1].Shape()[1 - axis], kernel_weight, bias_weight);
  if (fc_layer == nullptr) {
    MS_LOG(ERROR) << "addFullyConnected failed for " << op_name_;
    return RET_ERROR;
  }
  this->layer_ = fc_layer;
  fc_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = fc_layer->getOutput(0);

  int origin_input_dims = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  if (out_tensor->getDimensions().nbDims != origin_input_dims) {
    std::vector<int64_t> squeeze_dim;
    for (int i = 0; i != origin_input_dims; ++i) {
      squeeze_dim.push_back(out_tensor->getDimensions().d[i]);
    }
    out_tensor = Reshape(ctx, out_tensor, squeeze_dim);
  }
  // add activation
  if (activation_ != ActivationType::NO_ACTIVATION) {
    nvinfer1::ILayer *activation_layer =
      ActivationTensorRT::AddActivation(ctx, activation_, 0, 0, 0, out_tensor, op_name_, device_id_);
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for matmul failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
    out_tensor = activation_layer->getOutput(0);
  }

  ctx->RegisterTensor(ITensorHelper{out_tensor, fc_input.format_}, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(out_tensor);
  return RET_OK;
}

int FullyConnectedTensorRT::PreprocessInputs(TensorRTContext *ctx, ITensorHelper *fc_input) {
  auto ret = PreprocessInputs2SameDim(ctx, input(ctx, in_tensors_[1].IsConst() ? 0 : 1), fc_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim failed for " << op_name_;
    return ret;
  }
  auto origin_dims = fc_input->trt_tensor_->getDimensions();
  if (origin_dims.nbDims != DIMENSION_4D) {
    std::vector<int64_t> expand_dim(origin_dims.d, origin_dims.d + origin_dims.nbDims);
    for (int i = 0; i < DIMENSION_4D - origin_dims.nbDims; i++) {
      expand_dim.push_back(1);
    }
    fc_input->trt_tensor_ = Reshape(ctx, fc_input->trt_tensor_, expand_dim);
  }
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameFullConnection, FullyConnectedTensorRT)
}  // namespace mindspore::lite
