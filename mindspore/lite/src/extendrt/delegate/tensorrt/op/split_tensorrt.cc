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
#include "ops/split.h"
#include "ops/unstack.h"

namespace mindspore::lite {
int SplitTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                             const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1 && in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  dynamic_shape_params_.support_dynamic_ = false;
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

int SplitTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper split_input;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &split_input);
  if (ret != RET_OK || split_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return ret;
  }

  ret = ParseParams(split_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << op_name_ << " parse params failed.";
    return ret;
  }

  int input_nbdims = split_input.trt_tensor_->getDimensions().nbDims;
  axis_ = axis_ < 0 ? axis_ + input_nbdims : axis_;

  if (axis_ < 0 || axis_ >= input_nbdims) {
    MS_LOG(ERROR) << "invalid axis : " << axis_;
    return RET_ERROR;
  }
  int split_sum = std::accumulate(size_splits_.begin(), size_splits_.end(), 0);
  int split_sum_expect = split_input.trt_tensor_->getDimensions().d[axis_];

  if (size_splits_[size_splits_.size() - 1] == -1) {
    size_splits_[size_splits_.size() - 1] = split_sum_expect - split_sum - 1;
    split_sum = split_sum_expect;
  }

  if (split_sum != split_sum_expect) {
    MS_LOG(ERROR) << "Sum of size splits not equal input tensor dim. ";
    return RET_ERROR;
  }

  int axis_dim_index = 0;
  nvinfer1::Dims one_dims = lite::ConvertCudaDims(1, input_nbdims);
  nvinfer1::ISliceLayer *slice_layer = nullptr;

  for (int i = 0; i != output_num_; ++i) {
    nvinfer1::Dims start_dims = lite::ConvertCudaDims(0, input_nbdims);
    start_dims.d[axis_] = axis_dim_index;
    axis_dim_index += size_splits_[i];

    nvinfer1::Dims size_dims = split_input.trt_tensor_->getDimensions();
    size_dims.d[axis_] = size_splits_[i];

    slice_layer = ctx->network()->addSlice(*split_input.trt_tensor_, start_dims, size_dims, one_dims);
    if (slice_layer == nullptr) {
      MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
      return RET_ERROR;
    }

    nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
    if (type_ == ops::kNameUnstack) {
      auto shuffer_layer = ctx->network()->addShuffle(*out_tensor);
      auto shuffer_dims_opt = SqueezeDims(out_tensor->getDimensions(), axis_);
      if (!shuffer_dims_opt) {
        MS_LOG(ERROR) << "SqueezeDims failed.";
        return RET_ERROR;
      }
      shuffer_layer->setReshapeDimensions(shuffer_dims_opt.value());
      out_tensor = shuffer_layer->getOutput(0);
    }
    ctx->RegisterTensor(ITensorHelper{out_tensor, split_input.format_, split_input.same_format_},
                        out_tensors_[i].Name());
  }
  this->layer_ = slice_layer;
  return RET_OK;
}

int SplitTensorRT::ParseParams(const ITensorHelper &helper) {
  if (type_ == ops::kNameSplit) {
    auto split_op = AsOps<ops::Split>();
    CHECK_NULL_RETURN(split_op);
    axis_ = split_op->get_axis();
    output_num_ = split_op->get_output_num();
    auto size_splits_ptr = split_op->get_size_splits();
    if (!size_splits_ptr.empty()) {
      size_splits_.resize(size_splits_ptr.size());
      std::copy(size_splits_ptr.begin(), size_splits_ptr.end(), size_splits_.begin());
    } else if (in_tensors_.size() == INPUT_SIZE2 && in_tensors_[1].IsConst() &&
               in_tensors_[1].DataType() == DataType::kNumberTypeInt32) {
      size_splits_.resize(in_tensors_[1].ElementNum());
      auto split_out_ptr = static_cast<const int *>(in_tensors_[1].Data());
      for (int i = 0; i < in_tensors_[1].ElementNum(); i++) {
        size_splits_[i] = split_out_ptr[i];
      }
    } else {
      MS_LOG(INFO) << op_name_ << " has invalid input size and size_splits: " << in_tensors_.size();
    }
  } else if (type_ == ops::kNameUnstack) {
    auto unstack_op = AsOps<ops::Unstack>();
    CHECK_NULL_RETURN(unstack_op);
    axis_ = unstack_op->get_axis();
    output_num_ = out_tensors_.size();
  } else {
    MS_LOG(ERROR) << op_name_ << " has invalid type for split";
    return RET_ERROR;
  }
  int axis_dim = helper.trt_tensor_->getDimensions().d[axis_];
  if (size_splits_.empty()) {
    if (output_num_ == 0 || axis_dim % output_num_ != 0) {
      MS_LOG(ERROR) << "axis dim can not be split into same subdim output_num : " << output_num_
                    << " axis_dim: " << axis_dim;
      return RET_ERROR;
    }
    int split_width = axis_dim / output_num_;
    size_splits_.resize(output_num_);
    std::fill(size_splits_.begin(), size_splits_.end(), split_width);
  }
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameSplit, SplitTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameUnstack, SplitTensorRT)
}  // namespace mindspore::lite
