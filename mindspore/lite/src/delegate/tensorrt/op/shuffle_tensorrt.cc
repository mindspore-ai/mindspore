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

namespace mindspore::lite {
int ShuffleTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  if ((type_ == schema::PrimitiveType::PrimitiveType_Squeeze ||
       type_ == schema::PrimitiveType::PrimitiveType_Unsqueeze) &&
      in_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid input tensort size: " << in_tensors.size();
    return RET_ERROR;
  }
  if ((type_ == schema::PrimitiveType::PrimitiveType_Transpose) && in_tensors.size() != 2) {
    MS_LOG(ERROR) << "invalid input tensort size: " << in_tensors.size();
    return RET_ERROR;
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
  nvinfer1::IShuffleLayer *shuffle_layer = network->addShuffle(*tensorrt_in_tensors_[0]);
  if (shuffle_layer == nullptr) {
    MS_LOG(ERROR) << "add Shuffle op failed for TensorRT.";
    return RET_ERROR;
  }
  shuffle_layer->setName(op_name_.c_str());

  switch (this->type()) {
    case schema::PrimitiveType_Unsqueeze: {
      int ret = AddUnsqueezeOp(shuffle_layer);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "AddUnSqueezeOp failed.";
        return ret;
      }
      break;
    }
    case schema::PrimitiveType_Squeeze: {
      int ret = AddSqueezeOp(shuffle_layer);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "AddSqueezeOp failed.";
        return ret;
      }
      break;
    }
    case schema::PrimitiveType_Transpose: {
      int ret = AddTransposeOp(shuffle_layer);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "AddTransposeOpss failed.";
        return ret;
      }
      break;
    }
    case schema::PrimitiveType_Reshape: {
      int ret = AddReshapeOp(shuffle_layer);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "AddReshapeOp failed.";
        return ret;
      }
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported op type.";
      return RET_ERROR;
  }

  nvinfer1::ITensor *out_tensor = shuffle_layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor create failed";
    return RET_ERROR;
  }
  out_tensor->setName(out_tensors_[0].Name().c_str());
  this->AddInnerOutTensors(out_tensor);
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
  MS_LOG(INFO) << "AddSqueezeOp: " << op_name_ << " squeeze_dims.nbDims: " << squeeze_dims.nbDims;

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
  auto unsqueeze_shape = in_tensors_[0].Shape();
  auto begin = std::begin(unsqueeze_shape);
  auto axis = unsqueeze_op->axis();

  for (size_t i = 0; i < axis->size(); i++) {
    unsqueeze_shape.insert(begin + axis->Get(i), 1);
  }

  nvinfer1::Dims unsqueeze_dims = lite::ConvertCudaDims(unsqueeze_shape);
  MS_LOG(INFO) << "AddUnsqueezeOp: " << op_name_ << " unsqueeze_dims.nbDims: " << unsqueeze_dims.nbDims;

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
  if (perm_ternsor.Data() == nullptr || perm_ternsor.ElementNum() != tensorrt_in_tensors_[0]->getDimensions().nbDims) {
    MS_LOG(ERROR) << "AddTransposeOp perm_ternsor data is invalid.";
    return RET_ERROR;
  }
  int *perm_data = reinterpret_cast<int *>(perm_ternsor.MutableData());

  nvinfer1::Permutation perm{};
  for (int i = 0; i < perm_ternsor.ElementNum(); i++) {
    perm.order[i] = *perm_data;
    perm_data++;
  }
  shuffle_layer->setFirstTranspose(perm);
  return RET_OK;
}
int ShuffleTensorRT::AddReshapeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  auto reshape_op = this->op_primitive_->value_as_Reshape();
  if (reshape_op == nullptr) {
    MS_LOG(ERROR) << "AddReshapeOp convert failed";
    return RET_ERROR;
  }
  if (in_tensors_.size() != 2) {
    MS_LOG(ERROR) << "AddReshapeOp size of in tensort needs check: " << in_tensors_.size();
    return RET_ERROR;
  }
  mindspore::MSTensor &shape_tensor = in_tensors_[1];
  nvinfer1::Dims reshape_dims = ConvertCudaDims(shape_tensor.Data().get(), shape_tensor.ElementNum());
  int ret = InferReshapeDims(tensorrt_in_tensors_[0]->getDimensions(), &reshape_dims);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "invalid dims for reshape " << op_name_;
    return ret;
  }
  shuffle_layer->setReshapeDimensions(reshape_dims);
  return RET_OK;
}
int ShuffleTensorRT::InferReshapeDims(nvinfer1::Dims input_dims, nvinfer1::Dims *reshape_dims) {
  int infer_index = -1;
  int known_cnt = 1;
  for (int i = 0; i < reshape_dims->nbDims; i++) {
    if (reshape_dims->d[i] == 0) {
      reshape_dims->d[i] = input_dims.d[i];
      known_cnt *= input_dims.d[i];
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
      tot_cnt *= input_dims.d[i];
    }
    reshape_dims->d[infer_index] = tot_cnt / known_cnt;
    MS_LOG(INFO) << "reshape infer_index: " << infer_index << ", reshape infer value: " << reshape_dims->d[infer_index];
  }
  return RET_OK;
}
}  // namespace mindspore::lite
