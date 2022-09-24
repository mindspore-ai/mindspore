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
#include "ops/unsqueeze.h"
#include "ops/squeeze.h"
#include "ops/reshape.h"
#include "ops/transpose.h"
#include "ops/flatten.h"
#include "ops/expand_dims.h"
#include "ops/broadcast_to.h"

namespace mindspore::lite {
int ShuffleTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                               const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
  }
  if (type_ == ops::kNameFlatten || type_ == ops::kNameUnsqueeze) {
    if (in_tensors.size() != 1) {
      MS_LOG(ERROR) << "Unsupported in_tensors size " << in_tensors.size() << " of " << type_;
      return RET_ERROR;
    }
  } else if (type_ == ops::kNameSqueeze) {
    if (in_tensors.size() != 1) {
      MS_LOG(ERROR) << "Unsupported in_tensors size " << in_tensors.size() << " of " << type_;
      return RET_ERROR;
    }
    auto squeeze_op = AsOps<ops::Squeeze>();
    if (squeeze_op == nullptr) {
      MS_LOG(ERROR) << "SqueezeOp convert failed";
      return RET_ERROR;
    }
    param_axis_ = squeeze_op->get_axis();
    if (param_axis_.empty()) {
      MS_LOG(WARNING) << op_name_ << " is a full dim squeeze, don't support dynamic input shape.";
      dynamic_shape_params_.support_dynamic_ = false;
      dynamic_shape_params_.support_hw_dynamic_ = false;
    }
  } else if (type_ == ops::kNameReshape) {
    if (in_tensors.size() != INPUT_SIZE2) {
      MS_LOG(ERROR) << "PrimitiveType_Transpose Unsupported in_tensors size: " << in_tensors.size();
      return RET_ERROR;
    }
    dynamic_shape_params_.support_hw_dynamic_ = false;
  } else if (type_ == ops::kNameTranspose || type_ == ops::kNameExpandDims || type_ == ops::kNameBroadcastTo) {
    if (in_tensors.size() != INPUT_SIZE2) {
      MS_LOG(ERROR) << "PrimitiveType_Transpose Unsupported in_tensors size: " << in_tensors.size();
      return RET_ERROR;
    }
    if (!in_tensors[1].IsConst()) {
      MS_LOG(ERROR) << "Unsupported shape tensor of " << type_;
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported op type:" << type_;
    return RET_ERROR;
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
  if (type_ == ops::kNameUnsqueeze) {
    ret = AddUnsqueezeOp(shuffle_layer);
  } else if (type_ == ops::kNameSqueeze) {
    ret = AddSqueezeOp(shuffle_layer);
  } else if (type_ == ops::kNameTranspose) {
    ret = AddTransposeOp(shuffle_layer);
  } else if (type_ == ops::kNameReshape) {
    ret = AddReshapeOp(shuffle_layer);
  } else if (type_ == ops::kNameFlatten) {
    ret = AddFlattenOp(shuffle_layer);
  } else if (type_ == ops::kNameExpandDims) {
    ret = AddExpandDimsOp(shuffle_layer);
  } else if (type_ == ops::kNameBroadcastTo) {
    ret = AddBroadcastToOp(shuffle_layer);
  } else {
    MS_LOG(ERROR) << "Unsupported op type for " << op_name_;
    return RET_ERROR;
  }
  if (ret == RET_OK) {
    auto output_helper = ITensorHelper{shuffler_output_, out_format_, true};
    ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
    MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  }
  return ret;
}

int ShuffleTensorRT::InputTensorPreprocess(TensorRTContext *ctx) {
  shuffler_input_ = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(input(ctx, 0));
  out_format_ = input(ctx, 0).format_;

  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(shuffler_input_, out_format_, true);
  return RET_OK;
}

int ShuffleTensorRT::AddSqueezeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  // axis
  auto squeeze_shape = shuffler_input_->getDimensions();
  std::vector<int64_t> new_shape(squeeze_shape.d, squeeze_shape.d + squeeze_shape.nbDims);
  if (param_axis_.empty()) {
    MS_LOG(WARNING) << op_name_ << " has null axis.";
    for (int i = new_shape.size() - 1; i >= 0; i--) {
      if (new_shape[i] == 1) {
        new_shape.erase(new_shape.begin() + i);
      }
    }
  } else {
    for (int i = SizeToInt(param_axis_.size()) - 1; i >= 0; i--) {
      if (param_axis_[i] < 0 || param_axis_[i] >= SizeToInt(new_shape.size()) || new_shape[param_axis_[i]] != 1) {
        MS_LOG(WARNING) << "squeeze_shape value at " << i << " is " << param_axis_[i] << ", need check " << op_name_;
      }
      new_shape.erase(new_shape.begin() + param_axis_[i]);
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
  auto unsqueeze_op = AsOps<ops::Unsqueeze>();
  if (unsqueeze_op == nullptr) {
    MS_LOG(ERROR) << "AddUnsqueezeOp convert failed";
    return RET_ERROR;
  }
  // axis
  param_axis_ = unsqueeze_op->get_axis();
  if (param_axis_.empty()) {
    MS_LOG(ERROR) << "axis is invalid for " << op_name_;
    return RET_ERROR;
  }
  if (param_axis_.size() != 1) {
    MS_LOG(WARNING) << op_name_ << " has unsqueeze axis size: " << param_axis_.size();
  }
  nvinfer1::ITensor *expand_input = shuffler_input_;
  if (input(ctx_, 0).is_tensor == true) {
    for (size_t i = 0; i < param_axis_.size(); i++) {
      expand_input = ExpandDim(shuffle_layer, expand_input, param_axis_[i]);
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
  auto transpose_op = AsOps<ops::Transpose>();
  if (transpose_op == nullptr) {
    MS_LOG(ERROR) << "AddTransposeOp convert failed";
    return RET_ERROR;
  }
  // perm
  auto perm_ternsor = in_tensors_[1];
  if (!perm_ternsor.IsConst()) {
    MS_LOG(ERROR) << "AddTransposeOp perm_ternsor data is invalid: " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::Permutation perm{};
  if (perm_ternsor.DataType() == DataType::kNumberTypeInt64) {
    int64_t *perm_data = reinterpret_cast<int64_t *>(perm_ternsor.MutableData());
    for (int i = 0; i < perm_ternsor.ElementNum(); i++) {
      perm.order[i] = *perm_data;
      perm_data++;
    }
  } else if (perm_ternsor.DataType() == DataType::kNumberTypeInt32) {
    int *perm_data = reinterpret_cast<int *>(perm_ternsor.MutableData());
    for (int i = 0; i < perm_ternsor.ElementNum(); i++) {
      perm.order[i] = *perm_data;
      perm_data++;
    }
  } else {
    MS_LOG(ERROR) << op_name_ << " perm tensor data type is " << static_cast<int>(perm_ternsor.DataType());
    return RET_ERROR;
  }

  shuffle_layer->setFirstTranspose(perm);

  shuffler_output_ = shuffle_layer->getOutput(0);
  return RET_OK;
}

int ShuffleTensorRT::AddReshapeOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  auto &shape_tensor = in_tensors_[1];
  if (shape_tensor.IsConst()) {
    // static shuffle layer
    shuffle_layer->setReshapeDimensions(
      InferReshapeDims(shuffler_input_->getDimensions(), in_tensors_[0].Shape(), out_tensors_[0].Shape()));
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
  int axis;
  if (in_tensors_[1].DataType() == DataType::kNumberTypeInt64) {
    auto axis_data = static_cast<const int64_t *>(in_tensors_[1].Data());
    axis = axis_data[0];
  } else if (in_tensors_[1].DataType() == DataType::kNumberTypeInt32) {
    auto axis_data = static_cast<const int32_t *>(in_tensors_[1].Data());
    axis = axis_data[0];
  } else {
    MS_LOG(WARNING) << op_name_ << " axis tensor data type is " << static_cast<int>(in_tensors_[1].DataType());
    return RET_ERROR;
  }
  shuffler_output_ = ExpandDim(shuffle_layer, shuffler_input_, axis);
  return shuffler_output_ == nullptr ? RET_ERROR : RET_OK;
}

int ShuffleTensorRT::AddBroadcastToOp(nvinfer1::IShuffleLayer *shuffle_layer) {
  if (out_tensors_[0].ElementNum() != in_tensors_[0].ElementNum() &&
      out_tensors_[0].Shape().size() == in_tensors_[0].Shape().size()) {
    MS_LOG(WARNING) << "broadcast element cnt changes, ignore broadcast for " << op_name_;
    shuffle_layer->setReshapeDimensions(shuffler_input_->getDimensions());
    MS_LOG(WARNING) << "here " << op_name_;
  } else if (out_tensors_[0].ElementNum() == in_tensors_[0].ElementNum()) {
    nvinfer1::Dims new_dims = ConvertCudaDims(out_tensors_[0].Shape());
    if (new_dims.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return RET_ERROR;
    }
    new_dims.d[0] = shuffler_input_->getDimensions().d[0];
    shuffle_layer->setReshapeDimensions(new_dims);
    MS_LOG(WARNING) << "here " << op_name_;
  } else {
    MS_LOG(ERROR) << "broadcast needs check for " << op_name_;
  }
  shuffler_output_ = shuffle_layer->getOutput(0);
  return shuffler_output_ == nullptr ? RET_ERROR : RET_OK;
}

nvinfer1::ITensor *ShuffleTensorRT::ExpandDim(nvinfer1::IShuffleLayer *shuffle_layer, nvinfer1::ITensor *input_tensor,
                                              int axis) {
  auto input_dims = input_tensor->getDimensions();
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
      return nullptr;
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
    nvinfer1::IShuffleLayer *trans_layer = ctx_->network()->addShuffle(*shuffle_layer->getOutput(0));
    if (trans_layer == nullptr) {
      MS_LOG(ERROR) << "add transpose layer failed for special expand dims op " << op_name_;
      return nullptr;
    }
    trans_layer->setFirstTranspose(perm);
    return trans_layer->getOutput(0);
  } else {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < input_dims.nbDims; i++) {
      if (axis == i) {
        new_shape.push_back(1);
      }
      new_shape.push_back(input_dims.d[i] == -1 ? 0 : input_dims.d[i]);
    }
    if (axis == -1 || axis == input_dims.nbDims) {
      new_shape.push_back(1);
    }
    nvinfer1::Dims new_dims = ConvertCudaDims(new_shape);
    if (new_dims.nbDims == -1) {
      MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
      return nullptr;
    }
    shuffle_layer->setReshapeDimensions(new_dims);
    return shuffle_layer->getOutput(0);
  }
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
REGISTER_TENSORRT_CREATOR(ops::kNameUnsqueeze, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameSqueeze, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameReshape, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameTranspose, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameFlatten, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameExpandDims, ShuffleTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameBroadcastTo, ShuffleTensorRT)
}  // namespace mindspore::lite
