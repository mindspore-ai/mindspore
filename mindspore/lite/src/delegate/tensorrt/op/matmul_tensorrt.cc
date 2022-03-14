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

#include "src/delegate/tensorrt/op/matmul_tensorrt.h"
#include <memory>
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "src/delegate/tensorrt/op/activation_tensorrt.h"
#include "src/delegate/tensorrt/op/matmul_opt_plugin.h"
#include "src/delegate/tensorrt/tensorrt_runtime.h"

namespace mindspore::lite {
MatMulTensorRT::~MatMulTensorRT() {
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
}
int MatMulTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                              const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (type_ == schema::PrimitiveType_MatMulFusion) {
    auto primitive = this->GetPrimitive()->value_as_MatMulFusion();
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "convert to primitive matmul failed for " << op_name_;
      return RET_ERROR;
    }
    transpose_a_ = primitive->transpose_a();
    transpose_b_ = primitive->transpose_b();
    activation_ = primitive->activation_type();
  }
  nvinfer1::ITensor *out_tensor = nullptr;
  if (RunOptPlugin()) {
    // dynamic input batch size opt for 2d matrix
    out_tensor = AddAsOptPlugin(network);
    MS_LOG(INFO) << "use optimize matmul plugin for " << op_name_;
  } else if (in_tensors_.size() == INPUT_SIZE3 && in_tensors_[1].Data() != nullptr &&
             in_tensors_[kBiasIndex].Data() != nullptr && !transpose_a_ &&
             in_tensors_[1].Shape().size() == DIMENSION_2D &&
             (in_tensors_[0].Shape().size() == DIMENSION_2D || in_tensors_[0].Shape().size() == DIMENSION_4D)) {
    MS_LOG(DEBUG) << "use fully connected instead of matmul for " << op_name_;
    out_tensor = AddAsFullConnect(network);
  } else {
    MS_LOG(DEBUG) << "use origin tensorrt matmul for " << op_name_;
    out_tensor = AddAsMatmul(network);
  }
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "add matmul failed for " << op_name_;
    return RET_ERROR;
  }

  // add activation
  if (activation_ != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    nvinfer1::ILayer *activation_layer = ActivationTensorRT::AddActivation(network, activation_, 0, 0, 0, out_tensor);
    if (activation_layer == nullptr) {
      MS_LOG(ERROR) << "addActivation for matmul failed";
      return RET_ERROR;
    }
    activation_layer->setName((op_name_ + "_activation").c_str());
    out_tensor = activation_layer->getOutput(0);
  }

  out_tensor->setName((op_name_ + "_output").c_str());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(out_tensor);
  this->AddInnerOutTensors(ITensorHelper{out_tensor, out_format_});
  return RET_OK;
}

int MatMulTensorRT::PreprocessMatMulInputs(nvinfer1::INetworkDefinition *network, ITensorHelper *matmul_a,
                                           ITensorHelper *matmul_b) {
  int ret;
  if (tensorrt_in_tensors_.size() == INPUT_SIZE2) {
    int a_index =
      GetDimsVolume(tensorrt_in_tensors_[0].trt_tensor_->getDimensions()) == GetDimsVolume(in_tensors_[0].Shape()) ? 0
                                                                                                                   : 1;
    ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[a_index], matmul_a);
    if (ret != RET_OK || matmul_a->trt_tensor_ == nullptr) {
      MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input a failed for " << op_name_;
      return RET_ERROR;
    }
    ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[1 - a_index], matmul_b);
    if (ret != RET_OK || matmul_b->trt_tensor_ == nullptr) {
      MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input b failed for " << op_name_;
      return RET_ERROR;
    }
    out_format_ = matmul_a->format_;
    if (matmul_a->format_ != matmul_b->format_) {
      MS_LOG(WARNING) << "matmul input tensor has different format " << op_name_;
    }
  } else if (tensorrt_in_tensors_.size() == 1) {
    nvinfer1::ITensor *weight = nullptr;
    int weight_index = in_tensors_[1].Data() != nullptr ? 1 : 0;
    if (in_tensors_[weight_index].Shape().size() <
        static_cast<size_t>(tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims)) {
      weight = ConvertTensorWithExpandDims(network, in_tensors_[weight_index],
                                           in_tensors_[1 - weight_index].Shape().size(), op_name_);
    } else if (in_tensors_[weight_index].Shape().size() ==
               static_cast<size_t>(tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims)) {
      weight = ConvertConstantTensor(network, in_tensors_[weight_index], op_name_);
    } else {
      MS_LOG(ERROR) << "input tensor shape is invalid for " << op_name_;
      return RET_ERROR;
    }
    if (weight == nullptr) {
      MS_LOG(ERROR) << "create constant weight tensor failed for " << op_name_;
      return RET_ERROR;
    }
    if (weight_index == 1) {
      matmul_b->trt_tensor_ = weight;
      ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], matmul_a);
      if (ret != RET_OK || matmul_a->trt_tensor_ == nullptr) {
        MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input a failed for " << op_name_;
        return RET_ERROR;
      }
      out_format_ = matmul_a->format_;
    } else {
      matmul_a->trt_tensor_ = weight;
      ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], matmul_b);
      if (ret != RET_OK || matmul_b->trt_tensor_ == nullptr) {
        MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input b failed for " << op_name_;
        return RET_ERROR;
      }
      out_format_ = matmul_b->format_;
    }
  } else {
    MS_LOG(ERROR) << op_name_ << " tensorrt in tensor size is invalid " << tensorrt_in_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

nvinfer1::ITensor *MatMulTensorRT::AddAsMatmul(nvinfer1::INetworkDefinition *network) {
  ITensorHelper matmul_a;
  ITensorHelper matmul_b;

  int ret = PreprocessMatMulInputs(network, &matmul_a, &matmul_b);
  if (ret != RET_OK || matmul_a.trt_tensor_ == nullptr || matmul_b.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessMatMulInputs matmul failed for " << op_name_;
    return nullptr;
  }

  MS_LOG(DEBUG) << "matmul input a " << GetTensorFormat(matmul_a);
  MS_LOG(DEBUG) << "matmul input b " << GetTensorFormat(matmul_b);

  auto matmul_layer = network->addMatrixMultiply(
    *matmul_a.trt_tensor_, transpose_a_ ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE,
    *matmul_b.trt_tensor_, transpose_b_ ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE);
  if (matmul_layer == nullptr) {
    MS_LOG(ERROR) << "addMatrixMultiply failed for " << op_name_;
    return nullptr;
  }
  matmul_layer->setName(op_name_.c_str());
  return AddBias(network, matmul_layer->getOutput(0));
}

nvinfer1::ITensor *MatMulTensorRT::AddAsFullConnect(nvinfer1::INetworkDefinition *network) {
  nvinfer1::Weights weight;
  nvinfer1::Weights bias = ConvertWeight(in_tensors_[kBiasIndex]);
  nvinfer1::ITensor *input_a = tensorrt_in_tensors_[0].trt_tensor_;
  out_format_ = tensorrt_in_tensors_[0].format_;
  if (input_a->getDimensions().nbDims != DIMENSION_4D) {
    nvinfer1::Dims in_dims(input_a->getDimensions());
    in_dims.nbDims = DIMENSION_4D;
    for (int i = input_a->getDimensions().nbDims; i < DIMENSION_4D; i++) {
      in_dims.d[i] = 1;
    }
    input_a = Reshape(network, input_a, in_dims);
    if (input_a == nullptr) {
      MS_LOG(ERROR) << "reshape input failed for " << op_name_;
      return nullptr;
    }
    MS_LOG(DEBUG) << "full connect expand input a to " << GetTensorFormat(input_a);
  } else {
    ITensorHelper tmp_input;
    int ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], &tmp_input);
    if (ret != RET_OK || tmp_input.trt_tensor_ == nullptr) {
      MS_LOG(ERROR) << "rPreprocessInputs2SameDim failed for " << op_name_;
      return nullptr;
    }
    input_a = tmp_input.trt_tensor_;
    out_format_ = tmp_input.format_;
    MS_LOG(DEBUG) << "full connect preprocess input a to " << GetTensorFormat(tmp_input);
  }
  if (!transpose_b_) {
    // transpose weight
    weight = TransposeWeight2D(in_tensors_[1], &weight_ptr_);
    if (weight.values == nullptr || weight_ptr_ == nullptr) {
      MS_LOG(ERROR) << "TransposeWeight2D input weight failed for " << op_name_;
      return nullptr;
    }
  } else {
    weight = ConvertWeight(in_tensors_[1]);
  }

  int output_cnt = in_tensors_[kBiasIndex].Shape()[0];

  auto fc_layer = network->addFullyConnected(*input_a, output_cnt, weight, bias);
  if (fc_layer == nullptr) {
    MS_LOG(ERROR) << "add fully connected layer failed for " << op_name_;
    return nullptr;
  }
  fc_layer->setName((op_name_ + "_fullyconnected").c_str());
  nvinfer1::ITensor *out_tensor = fc_layer->getOutput(0);
  if (out_tensor->getDimensions().nbDims != out_tensors_[0].Shape().size()) {
    std::vector<int64_t> out_dims(out_tensors_[0].Shape());
    out_dims[0] = out_tensor->getDimensions().d[0];
    out_tensor = Reshape(network, out_tensor, out_dims);
  }
  return out_tensor;
}
nvinfer1::ITensor *MatMulTensorRT::AddAsOptPlugin(nvinfer1::INetworkDefinition *network) {
  nvinfer1::ITensor *weight_tensor = nullptr;
  if (tensorrt_in_tensors_.size() >= INPUT_SIZE2) {
    weight_tensor = tensorrt_in_tensors_[1].trt_tensor_;
  } else {
    weight_tensor = ConvertConstantTensor(network, in_tensors_[1], in_tensors_[1].Name());
  }

  auto plugin = std::make_shared<MatmulOptPlugin>(op_name_, transpose_a_, transpose_b_);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "create MatmulOptPlugin failed for " << op_name_;
    return nullptr;
  }
  nvinfer1::ITensor *inputTensors[] = {tensorrt_in_tensors_[0].trt_tensor_, weight_tensor};
  nvinfer1::IPluginV2Layer *matmul_layer = network->addPluginV2(inputTensors, INPUT_SIZE2, *plugin);
  if (matmul_layer == nullptr) {
    MS_LOG(ERROR) << "add matmul opt plugin layer failed for " << op_name_;
    return nullptr;
  }
  return AddBias(network, matmul_layer->getOutput(0));
}
nvinfer1::ITensor *MatMulTensorRT::AddBias(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input_tensor) {
  nvinfer1::ITensor *out_tensor = input_tensor;
  if (in_tensors_.size() == kBiasIndex + 1) {
    nvinfer1::ITensor *bias = nullptr;
    if (in_tensors_[kBiasIndex].Shape().size() < static_cast<size_t>(out_tensor->getDimensions().nbDims)) {
      bias =
        ConvertTensorWithExpandDims(network, in_tensors_[kBiasIndex], out_tensor->getDimensions().nbDims, op_name_);
    } else if (in_tensors_[kBiasIndex].Shape().size() == static_cast<size_t>(out_tensor->getDimensions().nbDims)) {
      bias = ConvertConstantTensor(network, in_tensors_[kBiasIndex], op_name_);
    } else {
      MS_LOG(ERROR) << "input tensor shape is invalid for " << op_name_;
      return nullptr;
    }
    if (bias == nullptr) {
      MS_LOG(ERROR) << "create constant bias tensor failed for " << op_name_;
      return nullptr;
    }
    auto bias_layer = network->addElementWise(*out_tensor, *bias, nvinfer1::ElementWiseOperation::kSUM);
    if (bias_layer == nullptr) {
      MS_LOG(ERROR) << "add bias add layer failed for " << op_name_;
      return nullptr;
    }
    auto bias_layer_name = op_name_ + "_bias";
    bias_layer->setName(bias_layer_name.c_str());
    out_tensor = bias_layer->getOutput(0);
  }
  return out_tensor;
}

bool MatMulTensorRT::RunOptPlugin() {
  if (in_tensors_[0].Shape().size() == DIMENSION_2D && in_tensors_[1].Shape().size() == DIMENSION_2D &&
      in_tensors_[0].Shape()[0] > 1 && tensorrt_in_tensors_[0].trt_tensor_->getDimensions().d[0] == -1 &&
      runtime_->GetRuntimePrecisionMode() == RuntimePrecisionMode::RuntimePrecisionMode_FP32) {
    return true;
  }
  return false;
}
}  // namespace mindspore::lite
