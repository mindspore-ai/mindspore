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

#include "src/delegate/tensorrt/op/shuffle_tensorrt.h"
#include <vector>
#include <numeric>
#include <functional>

namespace mindspore::lite {
int ShuffleTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  switch (type_) {
    case schema::PrimitiveType_Flatten:
    case schema::PrimitiveType_Squeeze:
    case schema::PrimitiveType_Unsqueeze: {
      if (in_tensors.size() != 1) {
        MS_LOG(ERROR) << "Unsupported in_tensors size " << in_tensors.size() << " of "
                      << schema::EnumNamePrimitiveType(type_);
        return RET_ERROR;
      }
      break;
    }
    case schema::PrimitiveType_Reshape: {
      if (in_tensors.size() != INPUT_SIZE2) {
        MS_LOG(ERROR) << "PrimitiveType_Transpose Unsupported in_tensors size: " << in_tensors.size();
        return RET_ERROR;
      }
      break;
    }
    case schema::PrimitiveType_Transpose: {
      if (in_tensors.size() != INPUT_SIZE2) {
        MS_LOG(ERROR) << "PrimitiveType_Transpose Unsupported in_tensors size: " << in_tensors.size();
        return RET_ERROR;
      }
      if (in_tensors[1].Data() == nullptr) {
        MS_LOG(ERROR) << "Unsupported shape tensor of " << schema::EnumNamePrimitiveType(type_);
        return RET_ERROR;
      }
      break;
    }
    default: {
      MS_LOG(ERROR) << "Unsupported op type:" << schema::EnumNamePrimitiveType(type_);
      return RET_ERROR;
    }
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensort size: " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int ShuffleTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  nvinfer1::ITensor *shuffler_input = tensorrt_in_tensors_[0].trt_tensor_;
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(shuffler_input, tensorrt_in_tensors_[0].format_);
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      !SameDims(tensorrt_in_tensors_[0].trt_tensor_->getDimensions(), in_tensors_[0].Shape())) {
    // only valid for nchw or nhwc
    if (tensorrt_in_tensors_[0].format_ == Format::NCHW) {
      nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(network, *tensorrt_in_tensors_[0].trt_tensor_);
      if (transpose_layer == nullptr) {
        MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
      }
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      shuffler_input = transpose_layer->getOutput(0);
      out_format_ = Format::NHWC;
    } else if (tensorrt_in_tensors_[0].format_ == Format::NHWC) {
      nvinfer1::IShuffleLayer *transpose_layer = NHWC2NCHW(network, *tensorrt_in_tensors_[0].trt_tensor_);
      if (transpose_layer == nullptr) {
        MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
      }
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      shuffler_input = transpose_layer->getOutput(0);
      out_format_ = Format::NCHW;
    } else {
      MS_LOG(ERROR) << "invalid input format for " << op_name_;
      return RET_ERROR;
    }
  } else {
    out_format_ = tensorrt_in_tensors_[0].format_;
  }
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(shuffler_input, out_format_);

  nvinfer1::IShuffleLayer *shuffle_layer = network->addShuffle(*shuffler_input);
  if (shuffle_layer == nullptr) {
    MS_LOG(ERROR) << "add Shuffle op failed for TensorRT.";
    return RET_ERROR;
  }
  shuffle_layer->setName(op_name_.c_str());

  int ret = RET_OK;
  switch (type_) {
    case schema::PrimitiveType_Unsqueeze: {
      ret = AddUnsqueezeOp(shuffle_layer);
      break;
    }
    case schema::PrimitiveType_Squeeze: {
      ret = AddSqueezeOp(shuffle_layer);
      break;
    }
    case schema::PrimitiveType_Transpose: {
      ret = AddTransposeOp(shuffle_layer);
      break;
    }
    case schema::PrimitiveType_Reshape: {
      ret = AddReshapeOp(shuffle_layer);
      break;
    }
    case schema::PrimitiveType_Flatten: {
      ret = AddFlattenOp(shuffle_layer);
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported op type for " << op_name_;
      return RET_ERROR;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AddOp failed for " << op_name_;
    return ret;
  }

  nvinfer1::ITensor *out_tensor = shuffle_layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor create failed";
    return RET_ERROR;
  }
  out_tensor->setName((op_name_ + "_output").c_str());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(out_tensor, out_format_);
  this->AddInnerOutTensors(ITensorHelper{out_tensor, out_format_});
  return RET_OK;
}

int ShuffleTensorRT::AddSqueezeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  // squeeze
  auto squeeze_op = this->op_primitive_->value_as_Squeeze();
  if (squeeze_op == nullptr) {
    MS_LOG(ERROR) << "SqueezeOp convert failed";
    return RET_ERROR;
  }

  // axis
  auto squeeze_shape = in_tensors_[0].Shape();
  auto begin = std::begin(squeeze_shape);
  auto axis = squeeze_op->axis();
  if (axis == nullptr) {
    MS_LOG(ERROR) << "AddSqueezeOp has invalid axis";
    return RET_ERROR;
  }

  for (size_t i = 0; i < axis->size(); i++) {
    if (squeeze_shape[axis->Get(i)] != 1) {
      MS_LOG(WARNING) << "squeeze_shape value is not 1, need check";
    }
    squeeze_shape.erase(begin + axis->Get(i));
  }

  nvinfer1::Dims squeeze_dims = lite::ConvertCudaDims(squeeze_shape);
  MS_LOG(DEBUG) << "AddSqueezeOp: " << op_name_ << " squeeze_dims.nbDims: " << squeeze_dims.nbDims;

  shuffle_layer->setReshapeDimensions(squeeze_dims);
  return shuffle_layer->getOutput(0) == nullptr ? RET_ERROR : RET_OK;
}

int ShuffleTensorRT::AddUnsqueezeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  // Unsqueeze
  auto unsqueeze_op = this->op_primitive_->value_as_Unsqueeze();
  if (unsqueeze_op == nullptr) {
    MS_LOG(ERROR) << "AddUnsqueezeOp convert failed";
    return RET_ERROR;
  }
  if (in_tensors_.size() != 1) {
    MS_LOG(WARNING) << "AddUnsqueezeOp size of in tensort needs check: " << in_tensors_.size();
  }
  // axis
  auto unsqueeze_shape = tensorrt_in_tensors_[0].trt_tensor_->getDimensions();
  std::vector<int64_t> new_shape(unsqueeze_shape.d, unsqueeze_shape.d + unsqueeze_shape.nbDims);
  auto axis = unsqueeze_op->axis();

  for (size_t i = 0; i < axis->size(); i++) {
    new_shape.insert(new_shape.begin() + axis->Get(i), 1);
  }

  nvinfer1::Dims unsqueeze_dims = lite::ConvertCudaDims(new_shape);

  shuffle_layer->setReshapeDimensions(unsqueeze_dims);
  return shuffle_layer->getOutput(0) == nullptr ? RET_ERROR : RET_OK;
}

int ShuffleTensorRT::AddTransposeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  auto transpose_op = this->op_primitive_->value_as_Transpose();
  if (transpose_op == nullptr) {
    MS_LOG(ERROR) << "AddTransposeOp convert failed";
    return RET_ERROR;
  }
  if (in_tensors_.size() != 2) {
    MS_LOG(ERROR) << "AddTransposeOp size of in tensort needs check: " << in_tensors_.size();
    return RET_ERROR;
  }
  // perm
  mindspore::MSTensor perm_ternsor = in_tensors_[1];
  if (perm_ternsor.Data() == nullptr) {
    MS_LOG(ERROR) << "AddTransposeOp perm_ternsor data is invalid: " << op_name_;
    return RET_ERROR;
  }
  int *perm_data = reinterpret_cast<int *>(perm_ternsor.MutableData());

  nvinfer1::Permutation perm{};
  for (int i = 0; i < perm_ternsor.ElementNum(); i++) {
    perm.order[i] = *perm_data;
    perm_data++;
  }
  shuffle_layer->setFirstTranspose(perm);
  if (perm_ternsor.ElementNum() == DIMENSION_4D) {
    if (perm.order[1] == 3 && perm.order[2] == 1 && perm.order[3] == 2) {
      out_format_ = Format::NCHW;
    } else if (perm.order[1] == 2 && perm.order[2] == 3 && perm.order[3] == 1) {
      out_format_ = Format::NHWC;
    } else {
      MS_LOG(WARNING) << "input format and perm order is invalid: " << op_name_;
    }
  }
  return RET_OK;
}

int ShuffleTensorRT::AddReshapeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  mindspore::MSTensor &shape_tensor = in_tensors_[1];
  if (shape_tensor.Data() != nullptr) {
    // static shuffle layer
    nvinfer1::Dims reshape_dims = lite::ConvertCudaDims(shape_tensor.Data().get(), shape_tensor.ElementNum());
    shuffle_layer->setReshapeDimensions(reshape_dims);
  } else {
    if (tensorrt_in_tensors_.size() != INPUT_SIZE2) {
      MS_LOG(ERROR) << "invalid shape tensor for reshape " << op_name_;
      return RET_ERROR;
    }
    shuffle_layer->setInput(1, *tensorrt_in_tensors_[1].trt_tensor_);
  }
  return RET_OK;
}

int ShuffleTensorRT::AddFlattenOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  nvinfer1::Dims flatten_dims;
  const std::vector<int64_t> &input_shape = in_tensors_[0].Shape();
  flatten_dims.nbDims = DIMENSION_2D;
  flatten_dims.d[0] = input_shape[0];
  flatten_dims.d[1] = std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<int>());
  shuffle_layer->setReshapeDimensions(flatten_dims);
  return RET_OK;
}

int ShuffleTensorRT::InferReshapeDims(nvinfer1::Dims input_dims, nvinfer1::Dims *reshape_dims) {
  // tensorrt support infer shape of 0 and -1
  int infer_index = -1;
  int known_cnt = 1;
  for (int i = 0; i < reshape_dims->nbDims; i++) {
    if (reshape_dims->d[i] == 0) {
      reshape_dims->d[i] = input_dims.d[i];
      known_cnt *= (input_dims.d[i] == -1 ? 1 : input_dims.d[i]);
    } else if (reshape_dims->d[i] == -1) {
      if (infer_index != -1) {
        MS_LOG(ERROR) << "invalid dims (more than one infer dim) for reshape " << op_name_;
        return RET_ERROR;
      }
      infer_index = i;
    } else {
      known_cnt *= input_dims.d[i];
    }
  }
  if (infer_index != -1) {
    size_t tot_cnt = 1;
    for (int i = 0; i < input_dims.nbDims; i++) {
      tot_cnt *= (input_dims.d[i] == -1 ? 1 : input_dims.d[i]);
    }
    if (known_cnt == 0) {
      MS_LOG(ERROR) << "invalid known cnt for " << op_name_;
      return RET_ERROR;
    }
    reshape_dims->d[infer_index] = tot_cnt / known_cnt;
    MS_LOG(DEBUG) << "reshape infer_index: " << infer_index
                  << ", reshape infer value: " << reshape_dims->d[infer_index];
  }
  return RET_OK;
}
}  // namespace mindspore::lite
