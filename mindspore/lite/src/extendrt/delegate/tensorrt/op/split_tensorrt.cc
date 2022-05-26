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

#include <numeric>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/op/split_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int SplitTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                             const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }

  if (out_tensors.size() < 1 || out_tensors.size() != output_num_) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (axis_ < 0 || axis_ >= in_tensors_[0].Shape().size()) {
    MS_LOG(ERROR) << "invalid axis : " << axis_;
    return RET_ERROR;
  }

  if (size_splits_.empty()) {
    if (in_tensors_[0].Shape().at(axis_) % output_num_ != 0) {
      MS_LOG(ERROR) << "axis dim can not be split into same subdim";
      return RET_ERROR;
    }
    int split_width = in_tensors_[0].Shape().at(axis_) / output_num_;
    size_splits_.resize(output_num_);
    std::fill(size_splits_.begin(), size_splits_.end(), split_width);
  }
  int split_sum = std::accumulate(size_splits_.begin(), size_splits_.end(), 0);
  int split_sum_expect = in_tensors_[0].Shape()[axis_];

  if (size_splits_[size_splits_.size() - 1] == -1) {
    size_splits_[size_splits_.size() - 1] = split_sum_expect - split_sum - 1;
    split_sum = split_sum_expect;
  }

  if (split_sum != split_sum_expect) {
    MS_LOG(ERROR) << "Sum of size splits not equal input tensor dim. ";
    return RET_ERROR;
  }

  dynamic_shape_params_.support_dynamic_ = false;
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

int SplitTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  ITensorHelper split_input;
  int ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], &split_input);
  if (ret != RET_OK || split_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return ret;
  }

  int axis_dim_index = 0;
  nvinfer1::Dims one_dims = lite::ConvertCudaDims(1, in_tensors_[0].Shape().size());
  nvinfer1::ISliceLayer *slice_layer = nullptr;

  for (int i = 0; i != output_num_; ++i) {
    nvinfer1::Dims start_dims = lite::ConvertCudaDims(0, in_tensors_[0].Shape().size());
    start_dims.d[axis_] = axis_dim_index;
    axis_dim_index += size_splits_[i];

    nvinfer1::Dims size_dims = lite::ConvertCudaDims(in_tensors_[0].Shape());
    size_dims.d[axis_] = size_splits_[i];

    slice_layer = network->addSlice(*split_input.trt_tensor_, start_dims, size_dims, one_dims);
    if (slice_layer == nullptr) {
      MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
      return RET_ERROR;
    }

    nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
    if (type_ == schema::PrimitiveType_Unstack) {
      auto shuffer_layer = network->addShuffle(*out_tensor);
      auto shuffer_dims_opt = SqueezeDims(out_tensor->getDimensions(), axis_);
      if (!shuffer_dims_opt) {
        MS_LOG(ERROR) << "SqueezeDims failed.";
        return RET_ERROR;
      }
      shuffer_layer->setReshapeDimensions(shuffer_dims_opt.value());
      out_tensor = shuffer_layer->getOutput(0);
    }
    out_tensor->setName((op_name_ + "_" + std::to_string(i)).c_str());
    this->AddInnerOutTensors(ITensorHelper{out_tensor, split_input.format_, split_input.same_format_});
  }
  this->layer_ = slice_layer;
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Split, SplitTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Unstack, SplitTensorRT)
}  // namespace mindspore::lite
