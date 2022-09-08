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

#include "src/extendrt/delegate/tensorrt/op/shuffle_tensorrt.h"
#include <vector>
#include <numeric>
#include <functional>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int ShuffleTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    // return RET_ERROR;
  }
  switch (type_) {
    case schema::PrimitiveType_Flatten:
    case schema::PrimitiveType_Unsqueeze: {
      if (in_tensors.size() != 1) {
        MS_LOG(ERROR) << "Unsupported in_tensors size " << in_tensors.size() << " of "
                      << schema::EnumNamePrimitiveType(type_);
        return RET_ERROR;
      }
      break;
    }
    case schema::PrimitiveType_Squeeze: {
      if (in_tensors.size() != 1) {
        MS_LOG(ERROR) << "Unsupported in_tensors size " << in_tensors.size() << " of "
                      << schema::EnumNamePrimitiveType(type_);
        return RET_ERROR;
      }
      auto squeeze_op = this->op_primitive_->value_as_Squeeze();
      if (squeeze_op == nullptr) {
        MS_LOG(ERROR) << "SqueezeOp convert failed";
        return RET_ERROR;
      }
      param_axis_ = squeeze_op->axis();
      if (param_axis_ == nullptr) {
        MS_LOG(WARNING) << op_name_ << " is a full dim squeeze, don't support dynamic input shape.";
        dynamic_shape_params_.support_dynamic_ = false;
        dynamic_shape_params_.support_hw_dynamic_ = false;
      }
      break;
    }
    case schema::PrimitiveType_Reshape: {
      if (in_tensors.size() != INPUT_SIZE2) {
        MS_LOG(ERROR) << "PrimitiveType_Transpose Unsupported in_tensors size: " << in_tensors.size();
        return RET_ERROR;
      }
      dynamic_shape_params_.support_hw_dynamic_ = false;
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
    case schema::PrimitiveType_BroadcastTo: {
      if (in_tensors.size() != INPUT_SIZE2) {
        MS_LOG(ERROR) << "PrimitiveType_Transpose Unsupported in_tensors size: " << in_tensors.size();
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

int ShuffleTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  ctx_ = ctx;

  int ret = InputTensorPreprocess(ctx);
  if (ret != RET_OK || shuffler_input_ == nullptr) {
    MS_LOG(ERROR) << "InputTensorPreprocess failed for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::IShuffleLayer *shuffle_layer = ctx->network()->addShuffle(*shuffler_input_);
  if (shuffle_layer == nullptr) {
    MS_LOG(ERROR) << "add Shuffle op failed for TensorRT.";
    return RET_ERROR;
  }
  shuffle_layer->setName(op_name_.c_str());
  this->layer_ = shuffle_layer;

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
    case schema::PrimitiveType_BroadcastTo: {
      ret = AddBroadcastToOp(shuffle_layer);
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
  auto output_helper = ITensorHelper{shuffler_output_, out_format_, true};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

int ShuffleTensorRT::InputTensorPreprocess(TensorRTContext *ctx) {
  shuffler_input_ = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(input(ctx, 0));
  out_format_ = input(ctx, 0).format_;
  if (shuffler_input_->getDimensions().nbDims == DIMENSION_4D && !input(ctx, 0).same_format_) {
    // input tensor support NCHW format input
    if (input(ctx, 0).format_ == Format::NCHW) {
      // for transpose op, if tensor has same dim with ms tensor, keep origin dims
      nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(ctx_, *shuffler_input_);
      if (transpose_layer == nullptr) {
        MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
        return RET_ERROR;
      }
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      shuffler_input_ = transpose_layer->getOutput(0);
      out_format_ = Format::NHWC;
    } else if (input(ctx, 0).format_ == Format::NHWC) {
      // infer format may error, correct here
      nvinfer1::IShuffleLayer *transpose_layer = NHWC2NCHW(ctx_, *shuffler_input_);
      if (transpose_layer == nullptr) {
        MS_LOG(ERROR) << "create transpose layer failed for " << op_name_;
        return RET_ERROR;
      }
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      shuffler_input_ = transpose_layer->getOutput(0);
      out_format_ = Format::NCHW;
    }
  }
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(shuffler_input_, out_format_, true);
  return RET_OK;
}

int ShuffleTensorRT::AddSqueezeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  // axis
  auto squeeze_shape = shuffler_input_->getDimensions();
  std::vector<int64_t> new_shape(squeeze_shape.d, squeeze_shape.d + squeeze_shape.nbDims);
  if (param_axis_ == nullptr) {
    MS_LOG(WARNING) << op_name_ << " has null axis.";
    for (int i = new_shape.size() - 1; i >= 0; i--) {
      if (new_shape[i] == 1) {
        new_shape.erase(new_shape.begin() + i);
      }
    }
  } else {
    for (int i = param_axis_->size() - 1; i >= 0; i--) {
      if (new_shape[param_axis_->Get(i)] != 1) {
        MS_LOG(WARNING) << "squeeze_shape value at " << i << " is " << param_axis_->Get(i) << ", need check "
                        << op_name_;
      }
      new_shape.erase(new_shape.begin() + param_axis_->Get(i));
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
  // axis
  param_axis_ = unsqueeze_op->axis();
  if (param_axis_ == nullptr) {
    MS_LOG(ERROR) << "axis is invalid for " << op_name_;
    return RET_ERROR;
  }
  if (param_axis_->size() != 1) {
    MS_LOG(WARNING) << op_name_ << " has unsqueeze axis size: " << param_axis_->size();
  }
  nvinfer1::ITensor *expand_input = shuffler_input_;
  if (input(ctx_, 0).is_tensor_ == true) {
    for (size_t i = 0; i < param_axis_->size(); i++) {
      expand_input = ExpandDim(ctx_, expand_input, param_axis_->Get(i));
    }
  }
  shuffler_output_ = expand_input;
  return shuffler_output_ == nullptr ? RET_ERROR : RET_OK;
}

int ShuffleTensorRT::AddTransposeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  if (shuffler_input_->getDimensions().nbDims != in_tensors_[1].ElementNum()) {
    MS_LOG(WARNING) << "transpose perm is invalid for input, ignore " << op_name_;
    shuffler_output_ = shuffler_input_;
    return RET_OK;
  }
  auto transpose_op = this->op_primitive_->value_as_Transpose();
  if (transpose_op == nullptr) {
    MS_LOG(ERROR) << "AddTransposeOp convert failed";
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
      MS_LOG(INFO) << "input format and perm order is not NHWC or NCHW: " << op_name_;
    }
  }
  shuffler_output_ = shuffle_layer->getOutput(0);
  return RET_OK;
}

int ShuffleTensorRT::AddReshapeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  mindspore::MSTensor &shape_tensor = in_tensors_[1];
  if (shape_tensor.Data() != nullptr) {
    // static shuffle layer
    nvinfer1::Dims reshape_dims{shape_tensor.ElementNum()};
    const int *shape_ptr = reinterpret_cast<const int *>(shape_tensor.Data().get());
    for (int i = 0; i != shape_tensor.ElementNum(); ++i) {
      reshape_dims.d[i] = *(shape_ptr + i);
    }
    shuffle_layer->setReshapeDimensions(reshape_dims);
  } else {
    if (in_tensors_.size() != INPUT_SIZE2) {
      MS_LOG(ERROR) << "invalid shape tensor for reshape " << op_name_;
      return RET_ERROR;
    }
    shuffle_layer->setInput(1, *input(ctx_, 1).trt_tensor_);
  }
  shuffler_output_ = shuffle_layer->getOutput(0);
  return RET_OK;
}

int ShuffleTensorRT::AddFlattenOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  nvinfer1::Dims flatten_dims;
  nvinfer1::Dims dims = input(ctx_, 0).trt_tensor_->getDimensions();
  flatten_dims.nbDims = DIMENSION_2D;
  flatten_dims.d[0] = dims.d[0] == -1 ? 0 : dims.d[0];
  flatten_dims.d[1] = std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<int32_t>());
  if (flatten_dims.d[1] <= 0) {
    MS_LOG(ERROR) << op_name_ << "infer shape failed";
  }
  shuffle_layer->setReshapeDimensions(flatten_dims);
  shuffler_output_ = shuffle_layer->getOutput(0);
  return RET_OK;
}

int ShuffleTensorRT::AddExpandDimsOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  if (!input(ctx_, 0).is_tensor_) {
    shuffler_output_ = shuffler_input_;
    return RET_OK;
  }
  if (in_tensors_[1].DataType() != DataType::kNumberTypeInt32) {
    MS_LOG(WARNING) << op_name_ << " axis tensor data type is " << static_cast<int>(in_tensors_[1].DataType());
  }
  auto axis_data = static_cast<const int *>(in_tensors_[1].Data().get());
  int axis = axis_data[0];
  shuffler_output_ = ExpandDim(ctx_, shuffler_input_, axis);
  return shuffler_output_ == nullptr ? RET_ERROR : RET_OK;
}

int ShuffleTensorRT::AddBroadcastToOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  if (in_tensors_[1].Data() == nullptr) {
    auto input_shape_tensor = input(ctx_, 1).trt_tensor_;
    shuffler_output_ = Broadcast(ctx_, shuffler_input_, input_shape_tensor);
  } else {
    std::vector<int> input_shape;
    const int *shape_ptr = reinterpret_cast<const int *>(in_tensors_[1].Data().get());
    for (int i = 0; i != in_tensors_[1].ElementNum(); ++i) {
      input_shape.push_back(*(shape_ptr + i));
    }

    nvinfer1::Dims in_tensor_dims = shuffler_input_->getDimensions();
    auto input_shape_tensor = ctx_->ConvertTo1DTensor(input_shape);

    while (in_tensor_dims.nbDims < input_shape.size()) {
      shuffler_input_ = ExpandDim(ctx_, shuffler_input_, 0);
      if (shuffler_input_->getDimensions().nbDims == -1) {
        MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
        return RET_ERROR;
      }
      shuffle_layer->setReshapeDimensions(shuffler_input_->getDimensions());
      shuffler_input_ = shuffle_layer->getOutput(0);
      in_tensor_dims = shuffler_input_->getDimensions();
    }

    auto size_tensor = ctx_->network()->addShape(*shuffler_input_)->getOutput(0);
    size_tensor = ctx_->network()
                    ->addElementWise(*input_shape_tensor, *size_tensor, nvinfer1::ElementWiseOperation::kMAX)
                    ->getOutput(0);
    shuffler_output_ = Broadcast(ctx_, shuffler_input_, size_tensor);
  }
  return shuffler_output_ == nullptr ? RET_ERROR : RET_OK;
}

REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Unsqueeze, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Squeeze, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Reshape, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Transpose, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Flatten, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_ExpandDims, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_BroadcastTo, ShuffleTensorRT)
}  // namespace mindspore::lite
