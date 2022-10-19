/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/matmul_tensorrt.h"
#include <memory>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "src/extendrt/delegate/tensorrt/op/activation_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_runtime.h"
#include "ops/fusion/mat_mul_fusion.h"

namespace mindspore::lite {
MatMulTensorRT::~MatMulTensorRT() {
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
}
int MatMulTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                              const std::vector<TensorInfo> &out_tensors) {
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

int MatMulTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (type_ == ops::kNameMatMulFusion) {
    auto primitive = AsOps<ops::MatMulFusion>();
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "convert to primitive matmul failed for " << op_name_;
      return RET_ERROR;
    }
    transpose_a_ = primitive->get_transpose_a();
    transpose_b_ = primitive->get_transpose_b();
    if (primitive->HasAttr(ops::kActivationType)) {
      activation_ = primitive->get_activation_type();
    }
  }
  nvinfer1::ITensor *out_tensor = nullptr;
  if (RunFullConnect(ctx)) {
    MS_LOG(DEBUG) << "use fully connected instead of matmul for " << op_name_;
    out_tensor = AddAsFullConnect(ctx);
  } else {
    MS_LOG(DEBUG) << "use origin tensorrt matmul for " << op_name_;
    out_tensor = AddAsMatmul(ctx);
  }
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "add matmul failed for " << op_name_;
    return RET_ERROR;
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

  ctx->RegisterTensor(ITensorHelper{out_tensor, out_format_}, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(out_tensor, out_format_, true);
  return RET_OK;
}

int MatMulTensorRT::PreprocessMatMulInputs(TensorRTContext *ctx, ITensorHelper *matmul_a, ITensorHelper *matmul_b) {
  if (!HasConst()) {
    *matmul_a = input(ctx, 0);
    *matmul_b = input(ctx, 1);
    int ret = PreprocessInputs2SameDim(ctx, *matmul_a, matmul_a);
    ret += PreprocessInputs2SameDim(ctx, *matmul_b, matmul_b);
    if (ret != RET_OK || matmul_a->trt_tensor_ == nullptr || matmul_b->trt_tensor_ == nullptr) {
      MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul inputs failed for " << op_name_;
      return ret;
    }
    out_format_ = matmul_a->format_;
    if (matmul_a->format_ != matmul_b->format_) {
      MS_LOG(WARNING) << "matmul input tensor has different format " << op_name_;
      out_format_ = Format::NCHW;
    }
  } else {
    for (size_t i = 0; i < in_tensors_.size(); i++) {
      auto in_tensor = input(ctx, i);
      if (in_tensors_[i].IsConst() || in_tensor.trt_tensor_ == nullptr) {
        in_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx, in_tensors_[i], op_name_);
        in_tensor.format_ = Format::NCHW;
        ctx->RegisterTensor(in_tensor, in_tensors_[i].Name());
      }
    }

    auto weight = ProcessWeightTensor(ctx);
    *matmul_a = input(ctx, 0);
    *matmul_b = input(ctx, 1);
    if (weight == nullptr) {
      MS_LOG(ERROR) << "create constant weight tensor failed for " << op_name_;
      return RET_ERROR;
    }
    int weight_index = in_tensors_[1].IsConst() ? 1 : 0;
    ITensorHelper *weight_helper = (weight_index == 1) ? matmul_b : matmul_a;
    ITensorHelper *var_helper = (weight_index == 1) ? matmul_a : matmul_b;
    weight_helper->trt_tensor_ = weight;
    int ret = PreprocessInputs2SameDim(ctx, *var_helper, var_helper);
    if (ret != RET_OK || var_helper->trt_tensor_ == nullptr) {
      MS_LOG(ERROR) << "PreprocessInputs2SameDim of matmul input var_helper failed for " << op_name_;
      return ret;
    }
    out_format_ = var_helper->format_;
  }
  return RET_OK;
}

nvinfer1::ITensor *MatMulTensorRT::ProcessWeightTensor(TensorRTContext *ctx) {
  nvinfer1::ITensor *weight = nullptr;
  int weight_index = in_tensors_[1].IsConst() ? 1 : 0;
  if (in_tensors_[weight_index].Shape().size() <
      static_cast<size_t>(input(ctx, 0).trt_tensor_->getDimensions().nbDims)) {
    std::vector<int64_t> expect_shape(input(ctx, 1 - weight_index).trt_tensor_->getDimensions().nbDims, 1);
    auto origin_shape = in_tensors_[weight_index].Shape();
    for (size_t i = 0; i < origin_shape.size(); i++) {
      expect_shape[expect_shape.size() - 1 - i] = origin_shape[origin_shape.size() - 1 - i];
    }
    weight = ConvertTensorWithExpandDims(ctx, in_tensors_[weight_index], expect_shape, op_name_);
  } else if (in_tensors_[weight_index].Shape().size() ==
             static_cast<size_t>(input(ctx, 0).trt_tensor_->getDimensions().nbDims)) {
    weight = ConvertConstantTensor(ctx, in_tensors_[weight_index], op_name_);
  } else {
    MS_LOG(ERROR) << "input tensor shape is invalid for " << op_name_;
    return nullptr;
  }
  return weight;
}

nvinfer1::ITensor *MatMulTensorRT::AddAsMatmul(TensorRTContext *ctx) {
  ITensorHelper matmul_a;
  ITensorHelper matmul_b;

  int ret = PreprocessMatMulInputs(ctx, &matmul_a, &matmul_b);
  if (ret != RET_OK || matmul_a.trt_tensor_ == nullptr || matmul_b.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessMatMulInputs matmul failed for " << op_name_;
    return nullptr;
  }

  MS_LOG(DEBUG) << "matmul input a " << GetTensorFormat(matmul_a);
  MS_LOG(DEBUG) << "matmul input b " << GetTensorFormat(matmul_b);

  auto matmul_layer = ctx->network()->addMatrixMultiply(
    *matmul_a.trt_tensor_, transpose_a_ ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE,
    *matmul_b.trt_tensor_, transpose_b_ ? nvinfer1::MatrixOperation::kTRANSPOSE : nvinfer1::MatrixOperation::kNONE);
  if (matmul_layer == nullptr) {
    MS_LOG(ERROR) << "addMatrixMultiply failed for " << op_name_;
    return nullptr;
  }
  this->layer_ = matmul_layer;
  matmul_layer->setName(op_name_.c_str());
  return AddBias(ctx, matmul_layer->getOutput(0));
}

nvinfer1::ITensor *MatMulTensorRT::AddAsFullConnect(TensorRTContext *ctx) {
  nvinfer1::Weights weight;
  nvinfer1::Weights bias = ConvertWeight(in_tensors_[kBiasIndex]);
  nvinfer1::ITensor *input_a = input(ctx, 0).trt_tensor_;
  out_format_ = input(ctx, 0).format_;
  if (input_a->getDimensions().nbDims != DIMENSION_4D) {
    nvinfer1::Dims in_dims(input_a->getDimensions());
    in_dims.nbDims = DIMENSION_4D;
    for (int i = input_a->getDimensions().nbDims; i < DIMENSION_4D; i++) {
      in_dims.d[i] = 1;
    }
    input_a = Reshape(ctx, input_a, in_dims);
    if (input_a == nullptr) {
      MS_LOG(ERROR) << "reshape input failed for " << op_name_;
      return nullptr;
    }
    MS_LOG(DEBUG) << "full connect expand input a to " << GetTensorFormat(input_a);
  } else {
    ITensorHelper tmp_input;
    int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &tmp_input);
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

  auto fc_layer = ctx->network()->addFullyConnected(*input_a, output_cnt, weight, bias);
  if (fc_layer == nullptr) {
    MS_LOG(ERROR) << "add fully connected layer failed for " << op_name_;
    return nullptr;
  }
  this->layer_ = fc_layer;
  fc_layer->setName((op_name_ + "_fullyconnected").c_str());
  nvinfer1::ITensor *out_tensor = fc_layer->getOutput(0);
  int origin_input_dims = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  if (out_tensor->getDimensions().nbDims != origin_input_dims) {
    std::vector<int64_t> squeeze_dim;
    for (int i = 0; i != origin_input_dims; ++i) {
      squeeze_dim.push_back(out_tensor->getDimensions().d[i]);
    }
    out_tensor = Reshape(ctx, out_tensor, squeeze_dim);
  }
  return out_tensor;
}
nvinfer1::ITensor *MatMulTensorRT::AddBias(TensorRTContext *ctx, nvinfer1::ITensor *input_tensor) {
  nvinfer1::ITensor *out_tensor = input_tensor;
  if (in_tensors_.size() == kBiasIndex + 1) {
    nvinfer1::ITensor *bias = nullptr;
    if (in_tensors_[kBiasIndex].Shape().size() < static_cast<size_t>(out_tensor->getDimensions().nbDims)) {
      std::vector<int64_t> expect_dims(input_tensor->getDimensions().nbDims, 1);
      expect_dims[expect_dims.size() - 1] = in_tensors_[kBiasIndex].Shape().back();
      bias = ConvertTensorWithExpandDims(ctx, in_tensors_[kBiasIndex], expect_dims, op_name_);
    } else if (in_tensors_[kBiasIndex].Shape().size() == static_cast<size_t>(out_tensor->getDimensions().nbDims)) {
      bias = ConvertConstantTensor(ctx, in_tensors_[kBiasIndex], op_name_);
    } else {
      MS_LOG(ERROR) << "input tensor shape is invalid for " << op_name_;
      return nullptr;
    }
    if (bias == nullptr) {
      MS_LOG(ERROR) << "create constant bias tensor failed for " << op_name_;
      return nullptr;
    }
    auto bias_layer = ctx->network()->addElementWise(*out_tensor, *bias, nvinfer1::ElementWiseOperation::kSUM);
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

bool MatMulTensorRT::RunFullConnect(TensorRTContext *ctx) {
  if (in_tensors_.size() == INPUT_SIZE3 && in_tensors_[1].IsConst() && in_tensors_[kBiasIndex].IsConst() &&
      !transpose_a_ && in_tensors_[1].Shape().size() == DIMENSION_2D &&
      (input(ctx, 0).trt_tensor_->getDimensions().nbDims == DIMENSION_2D ||
       input(ctx, 0).trt_tensor_->getDimensions().nbDims == DIMENSION_4D)) {
    return true;
  }
  return false;
}
REGISTER_TENSORRT_CREATOR(ops::kNameMatMulFusion, MatMulTensorRT)
}  // namespace mindspore::lite
