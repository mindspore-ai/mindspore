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
    case schema::PrimitiveType_Transpose:
    case schema::PrimitiveType_ExpandDims: {
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
  network_ = network;

  int ret = InputTensorPreprocess();
  if (ret != RET_OK || shuffler_input_ == nullptr) {
    MS_LOG(ERROR) << "InputTensorPreprocess failed for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::IShuffleLayer *shuffle_layer = network->addShuffle(*shuffler_input_);
  if (shuffle_layer == nullptr) {
    MS_LOG(ERROR) << "add Shuffle op failed for TensorRT.";
    return RET_ERROR;
  }
  shuffle_layer->setName(op_name_.c_str());

  ret = RET_OK;
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
    case schema::PrimitiveType_ExpandDims: {
      ret = AddExpandDimsOp(shuffle_layer);
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

  if (shuffler_output_ == nullptr) {
    MS_LOG(ERROR) << "output tensor create failed for " << op_name_;
    return RET_ERROR;
  }
  shuffler_output_->setName((op_name_ + "_output").c_str());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(shuffler_output_, out_format_);
  this->AddInnerOutTensors(ITensorHelper{shuffler_output_, out_format_, true});
  return RET_OK;
}

int ShuffleTensorRT::InputTensorPreprocess() {
  shuffler_input_ = tensorrt_in_tensors_[0].trt_tensor_;
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(shuffler_input_, tensorrt_in_tensors_[0].format_);
  out_format_ = tensorrt_in_tensors_[0].format_;
  if (shuffler_input_->getDimensions().nbDims == DIMENSION_4D && !tensorrt_in_tensors_[0].same_format_) {
    // input tensor support NCHW format input
    if (tensorrt_in_tensors_[0].format_ == Format::NCHW) {
      // for transpose op, if tensor has same dim with ms tensor, keep origin dims
      nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(network_, *shuffler_input_);
      if (transpose_layer == nullptr) {
        MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
        return RET_ERROR;
      }
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      shuffler_input_ = transpose_layer->getOutput(0);
      out_format_ = Format::NHWC;
    } else if (tensorrt_in_tensors_[0].format_ == Format::NHWC) {
      // infer format may error, correct here
      nvinfer1::IShuffleLayer *transpose_layer = NHWC2NCHW(network_, *shuffler_input_);
      if (transpose_layer == nullptr) {
        MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
        return RET_ERROR;
      }
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      shuffler_input_ = transpose_layer->getOutput(0);
      out_format_ = Format::NCHW;
    }
  }
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(shuffler_input_, out_format_);
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
  auto squeeze_shape = shuffler_input_->getDimensions();
  std::vector<int64_t> new_shape(squeeze_shape.d, squeeze_shape.d + squeeze_shape.nbDims);
  auto axis = squeeze_op->axis();
  if (axis == nullptr) {
    MS_LOG(WARNING) << op_name_ << " has invalid axis, output shape is totally depends on ms tensor.";
    new_shape = out_tensors_[0].Shape();
  } else {
    for (int i = axis->size() - 1; i >= 0; i--) {
      if (new_shape[axis->Get(i)] != 1) {
        MS_LOG(WARNING) << "squeeze_shape value at " << i << " is " << axis->Get(i) << ", need check " << op_name_;
      }
      new_shape.erase(new_shape.begin() + axis->Get(i));
    }
  }

  nvinfer1::Dims squeeze_dims = lite::ConvertCudaDims(new_shape);
  if (squeeze_dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return RET_ERROR;
  }
  shuffle_layer->setReshapeDimensions(squeeze_dims);
  shuffler_output_ = shuffle_layer->getOutput(0);
  return shuffler_output_ == nullptr ? RET_ERROR : RET_OK;
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
  auto unsqueeze_shape = shuffler_input_->getDimensions();
  std::vector<int64_t> new_shape(unsqueeze_shape.d, unsqueeze_shape.d + unsqueeze_shape.nbDims);
  auto axis = unsqueeze_op->axis();

  for (size_t i = 0; i < axis->size(); i++) {
    new_shape.insert(new_shape.begin() + axis->Get(i), 1);
  }

  nvinfer1::Dims unsqueeze_dims = lite::ConvertCudaDims(new_shape);
  if (unsqueeze_dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return RET_ERROR;
  }

  shuffle_layer->setReshapeDimensions(unsqueeze_dims);
  shuffler_output_ = shuffle_layer->getOutput(0);
  return shuffler_output_ == nullptr ? RET_ERROR : RET_OK;
}

int ShuffleTensorRT::AddTransposeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  auto transpose_op = this->op_primitive_->value_as_Transpose();
  if (transpose_op == nullptr) {
    MS_LOG(ERROR) << "AddTransposeOp convert failed";
    return RET_ERROR;
  }
  if (in_tensors_.size() != INPUT_SIZE2) {
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
    if (perm.order[kNCHW_C] == kNHWC_C && perm.order[kNCHW_H] == kNHWC_H && perm.order[kNCHW_W] == kNHWC_W) {
      out_format_ = Format::NCHW;
    } else if (perm.order[kNHWC_H] == kNCHW_H && perm.order[kNHWC_W] == kNCHW_W && perm.order[kNHWC_C] == kNCHW_C) {
      out_format_ = Format::NHWC;
    } else {
      MS_LOG(WARNING) << "input format and perm order is not NHWC or NCHW: " << op_name_;
    }
  }
  shuffler_output_ = shuffle_layer->getOutput(0);
  return RET_OK;
}

int ShuffleTensorRT::AddReshapeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  mindspore::MSTensor &shape_tensor = in_tensors_[1];
  if (shape_tensor.Data() != nullptr) {
    // static shuffle layer
    shuffle_layer->setReshapeDimensions(
      InferReshapeDims(shuffler_input_->getDimensions(), in_tensors_[0].Shape(), out_tensors_[0].Shape()));
  } else {
    if (tensorrt_in_tensors_.size() != INPUT_SIZE2) {
      MS_LOG(ERROR) << "invalid shape tensor for reshape " << op_name_;
      return RET_ERROR;
    }
    shuffle_layer->setInput(1, *tensorrt_in_tensors_[1].trt_tensor_);
  }
  shuffler_output_ = shuffle_layer->getOutput(0);
  return RET_OK;
}

int ShuffleTensorRT::AddFlattenOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  nvinfer1::Dims flatten_dims;
  const std::vector<int64_t> &input_shape = in_tensors_[0].Shape();
  flatten_dims.nbDims = DIMENSION_2D;
  flatten_dims.d[0] = tensorrt_in_tensors_[0].trt_tensor_->getDimensions().d[0] == -1
                        ? 0
                        : tensorrt_in_tensors_[0].trt_tensor_->getDimensions().d[0];
  flatten_dims.d[1] = std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<int>());
  if (flatten_dims.d[1] <= 0) {
    MS_LOG(ERROR) << op_name_ << "infer shape failed";
  }
  shuffle_layer->setReshapeDimensions(flatten_dims);
  shuffler_output_ = shuffle_layer->getOutput(0);
  return RET_OK;
}

int ShuffleTensorRT::AddExpandDimsOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  if (in_tensors_[1].DataType() != DataType::kNumberTypeInt32) {
    MS_LOG(WARNING) << op_name_ << " axis tensor data type is " << static_cast<int>(in_tensors_[1].DataType());
  }
  auto axis_data = static_cast<const int *>(in_tensors_[1].Data().get());
  int axis = axis_data[0];
  auto input_dims = shuffler_input_->getDimensions();
  // if expand dim not at last dim and shape is dynamic, change to expanddim at last dim and transpose
  bool special_expand = false;
  for (int i = 0; i < input_dims.nbDims; i++) {
    special_expand = special_expand || input_dims.d[i] == -1;
  }
  special_expand = special_expand && (axis != -1 && axis != input_dims.nbDims - 1);

  if (special_expand) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < input_dims.nbDims; i++) {
      new_shape.push_back(input_dims.d[i] == -1 ? 0 : input_dims.d[i]);
    }
    new_shape.push_back(1);
    nvinfer1::Dims new_dims = ConvertCudaDims(new_shape);
    if (new_dims.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return RET_ERROR;
    }
    shuffle_layer->setReshapeDimensions(new_dims);
    // transpose
    nvinfer1::Permutation perm{};
    for (int i = 0; i < new_dims.nbDims; i++) {
      if (i < axis) {
        perm.order[i] = i;
      } else if (i == axis) {
        perm.order[i] = new_dims.nbDims - 1;
      } else {
        perm.order[i] = i - 1;
      }
    }
    nvinfer1::IShuffleLayer *trans_layer = network_->addShuffle(*shuffle_layer->getOutput(0));
    if (trans_layer == nullptr) {
      MS_LOG(ERROR) << "add transpose layer failed for special expand dims op " << op_name_;
      return RET_ERROR;
    }
    trans_layer->setFirstTranspose(perm);
    shuffler_output_ = trans_layer->getOutput(0);
  } else {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < input_dims.nbDims; i++) {
      if (axis == i) {
        new_shape.push_back(1);
      }
      new_shape.push_back(input_dims.d[i] == -1 ? 0 : input_dims.d[i]);
    }
    if (axis == -1) {
      new_shape.push_back(1);
    }
    nvinfer1::Dims new_dims = ConvertCudaDims(new_shape);
    if (new_dims.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return RET_ERROR;
    }
    shuffle_layer->setReshapeDimensions(new_dims);
    shuffler_output_ = shuffle_layer->getOutput(0);
  }
  return RET_OK;
}

nvinfer1::Dims ShuffleTensorRT::InferReshapeDims(const nvinfer1::Dims &input_dims,
                                                 const std::vector<int64_t> &ms_input_shape,
                                                 const std::vector<int64_t> &ms_output_shape) {
  // tensorrt support infer shape of 0 and -1
  nvinfer1::Dims reshape_dims = ConvertCudaDims(ms_output_shape);
  if (reshape_dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return reshape_dims;
  }
  for (int i = 0; i < reshape_dims.nbDims; i++) {
    if (input_dims.d[i] == -1) {
      if (ms_input_shape[i] == ms_output_shape[i]) {
        reshape_dims.d[i] = 0;
      } else {
        reshape_dims.d[i] = -1;
      }
    }
    MS_LOG(DEBUG) << "reshape infer_index " << i << " value: " << reshape_dims.d[i];
  }
  return reshape_dims;
}
}  // namespace mindspore::lite
