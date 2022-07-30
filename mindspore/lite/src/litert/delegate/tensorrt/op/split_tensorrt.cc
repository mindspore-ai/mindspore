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
#include "src/litert/delegate/tensorrt/op/split_tensorrt.h"
#include "src/litert/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int SplitTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                             const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1 && in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  int ret = ParseParams();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << op_name_ << " parse params failed.";
    return ret;
  }

  axis_ = axis_ < 0 ? axis_ + in_tensors_[0].Shape().size() : axis_;

  if (out_tensors.size() < 1 || out_tensors.size() != output_num_) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (axis_ < 0 || axis_ >= in_tensors_[0].Shape().size()) {
    MS_LOG(ERROR) << "invalid axis : " << axis_;
    return RET_ERROR;
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

int SplitTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper split_input;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &split_input);
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

    slice_layer = ctx->network()->addSlice(*split_input.trt_tensor_, start_dims, size_dims, one_dims);
    if (slice_layer == nullptr) {
      MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
      return RET_ERROR;
    }

    nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
    if (type_ == schema::PrimitiveType_Unstack) {
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
int SplitTensorRT::ParseParams() {
  switch (type_) {
    case schema::PrimitiveType_Split: {
      auto split_op = op_primitive_->value_as_Split();
      CHECK_NULL_RETURN(split_op);
      axis_ = split_op->axis();
      output_num_ = split_op->output_num();
      auto size_splits_ptr = split_op->size_splits();
      if (size_splits_ptr != nullptr) {
        size_splits_.resize(size_splits_ptr->size());
        std::copy(size_splits_ptr->begin(), size_splits_ptr->end(), size_splits_.begin());
      } else if (in_tensors_.size() == INPUT_SIZE2 && in_tensors_[1].Data() != nullptr &&
                 in_tensors_[1].DataType() == DataType::kNumberTypeInt32) {
        size_splits_.resize(in_tensors_[1].ElementNum());
        auto split_out_ptr = static_cast<const int *>(in_tensors_[1].Data().get());
        for (int i = 0; i < in_tensors_[1].ElementNum(); i++) {
          size_splits_[i] = split_out_ptr[i];
        }
      } else {
        MS_LOG(ERROR) << op_name_ << " has invalid input size and size_splits: " << in_tensors_.size();
        return RET_ERROR;
      }
      break;
    }
    case schema::PrimitiveType_Unstack: {
      auto unstack_op = op_primitive_->value_as_Unstack();
      CHECK_NULL_RETURN(unstack_op);
      axis_ = unstack_op->axis();
      output_num_ = out_tensors_.size();
      break;
    }
    default: {
      MS_LOG(ERROR) << op_name_ << " has invalid type for split";
      return RET_ERROR;
    }
  }
  if (size_splits_.empty()) {
    if (output_num_ == 0 || in_tensors_[0].Shape().at(axis_) % output_num_ != 0) {
      MS_LOG(ERROR) << "axis dim can not be split into same subdim";
      return RET_ERROR;
    }
    int split_width = in_tensors_[0].Shape().at(axis_) / output_num_;
    size_splits_.resize(output_num_);
    std::fill(size_splits_.begin(), size_splits_.end(), split_width);
  }
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Split, SplitTensorRT)
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Unstack, SplitTensorRT)
}  // namespace mindspore::lite
