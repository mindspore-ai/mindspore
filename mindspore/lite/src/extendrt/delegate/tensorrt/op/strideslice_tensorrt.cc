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

#include "src/extendrt/delegate/tensorrt/op/strideslice_tensorrt.h"
#include <algorithm>
#include <numeric>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/strided_slice.h"

namespace mindspore::lite {
nvinfer1::ITensor *StrideSliceTensorRT::GetDynamicAxisSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                                                int size_dim, int axis,
                                                                nvinfer1::ITensor *size_tensor) {
  auto in_tensor_shape = ctx->network()->addShape(*input)->getOutput(0);
  if (in_tensor_shape == nullptr) {
    MS_LOG(ERROR) << "add shape layer of input failed!";
    return nullptr;
  }
  auto len_tensor = (size_tensor == nullptr ? ctx->ConvertTo1DTensor(static_cast<int>(size_dim)) : size_tensor);
  if (len_tensor == nullptr) {
    MS_LOG(ERROR) << "convert 1d tensor failed!";
    return nullptr;
  }

  nvinfer1::ITensor *concat_input_tensors[INPUT_SIZE2];
  concat_input_tensors[0] = in_tensor_shape;
  concat_input_tensors[1] = len_tensor;
  auto concat_layer = ctx->network()->addConcatenation(concat_input_tensors, INPUT_SIZE2);
  if (concat_layer == nullptr) {
    MS_LOG(ERROR) << "add concat layer failed!";
    return nullptr;
  }
  concat_layer->setAxis(0);
  auto shape_and_len = concat_layer->getOutput(0);
  if (shape_and_len == nullptr) {
    MS_LOG(ERROR) << "get concat layer result failed!";
    return nullptr;
  }

  std::vector<int> gather_slices(input->getDimensions().nbDims);
  std::iota(gather_slices.begin(), gather_slices.end(), 0);
  gather_slices[axis] = gather_slices.size();
  auto gather_slices_tensor = ctx->ConvertTo1DTensor(gather_slices);
  nvinfer1::IGatherLayer *gather_layer = ctx->network()->addGather(*shape_and_len, *gather_slices_tensor, 0);
  if (gather_layer == nullptr) {
    MS_LOG(ERROR) << "add gather layer failed!";
    return nullptr;
  }

  return gather_layer->getOutput(0);
}

nvinfer1::ITensor *StrideSliceTensorRT::GetDynamicSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                                            const nvinfer1::Dims &size_dims) {
  auto in_tensor_shape = ctx->network()->addShape(*input)->getOutput(0);
  if (in_tensor_shape == nullptr) {
    MS_LOG(ERROR) << "add shape layer of input failed!";
    return nullptr;
  }
  std::vector<int> is_dynamic;
  std::vector<int> is_fix;
  std::vector<int> size_vec;
  for (int i = 0; i != size_dims.nbDims; ++i) {
    is_dynamic.push_back(size_dims.d[i] < 0);
    is_fix.push_back(size_dims.d[i] >= 0);
    size_vec.push_back(size_dims.d[i]);
  }
  auto is_dynamic_tensor = ctx->ConvertTo1DTensor(is_dynamic);
  auto is_fix_tensor = ctx->ConvertTo1DTensor(is_fix);
  auto size_tensor = ctx->ConvertTo1DTensor(size_vec);

  auto fix_tensor =
    ctx->network()->addElementWise(*is_fix_tensor, *size_tensor, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  auto dynamic_tensor = ctx->network()
                          ->addElementWise(*is_dynamic_tensor, *in_tensor_shape, nvinfer1::ElementWiseOperation::kPROD)
                          ->getOutput(0);
  size_tensor =
    ctx->network()->addElementWise(*dynamic_tensor, *fix_tensor, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  return size_tensor;
}

int StrideSliceTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                   const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() < HAS_AXIS - 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (!in_tensors.at(BEGINS_INDEX).IsConst()) {
    MS_LOG(ERROR) << "invalid input tensor for: " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

bool StrideSliceTensorRT::GetConstInputValue(int *axis_val, int *start_val, int *stride_val) {
  int64_t axis_index = in_tensors_.size() == HAS_AXIS ? AXIS_INDEX : -1;
  const auto &begin = in_tensors_.at(BEGINS_INDEX);
  const auto &stride = in_tensors_.back();

  if (begin.ElementNum() != 1 || stride.ElementNum() != 1) {
    MS_LOG(ERROR)
      << "Only support element number of begin and stride to be 1 when this number < input dims number, op: "
      << op_name_;
    return false;
  }
  *axis_val = 0;
  if (axis_index != -1) {
    auto axis_vec = ConvertTensorAsIntVector(in_tensors_[axis_index]);
    if (axis_vec.size() != 1) {
      MS_LOG(ERROR) << "Failed to get axis input, node: " << op_name_ << ", axis count: " << axis_vec.size();
      return false;
    }
    *axis_val = axis_vec[0];
  }
  auto start_vec = ConvertTensorAsIntVector(begin);
  auto stride_vec = ConvertTensorAsIntVector(stride);
  if (start_vec.size() != 1 || stride_vec.size() != 1) {
    MS_LOG(ERROR) << "Failed to get start or stride input, node: " << op_name_;
    return {};
  }
  *start_val = start_vec[0];
  *stride_val = stride_vec[0];
  return true;
}

int StrideSliceTensorRT::ComputeSliceDims(TensorRTContext *ctx, ITensorHelper *slice_input) {
  auto op = AsOps<ops::StridedSlice>();
  shrink_axis_ = op->get_shrink_axis_mask();
  size_t start_mask = op->get_begin_mask();
  size_t end_mask = op->get_end_mask();

  const auto &begin = in_tensors_.at(BEGINS_INDEX);
  const auto &stride = in_tensors_.back();
  const auto &end = in_tensors_.at(ENDS_INDEX);

  auto input_dims = slice_input->trt_tensor_->getDimensions();

  if (begin.ElementNum() == slice_input->trt_tensor_->getDimensions().nbDims) {
    start_dims_ = lite::ConvertCudaDims(begin);
    auto end_dims = lite::ConvertCudaDims(end);
    size_dims_.nbDims = input_dims.nbDims;
    for (int i = 0; i < size_dims_.nbDims; i++) {
      size_t mask = 1 << i;
      start_dims_.d[i] = ((start_mask & mask) == 0 ? start_dims_.d[i] : 0);
      if (!end.IsConst()) {
        continue;
      }
      end_dims.d[i] = ((end_mask & mask) == 0 ? end_dims.d[i] : slice_input->trt_tensor_->getDimensions().d[i]);
      size_dims_.d[i] = end_dims.d[i] - start_dims_.d[i];
    }
    stride_dims_ = lite::ConvertCudaDims(stride);
    if (IsDynamicInput(ctx, 0)) {
      size_tensor_ = GetDynamicSliceSize(ctx, slice_input->trt_tensor_, size_dims_);
      size_dims_ = nvinfer1::Dims{-1};
    }
  } else {
    int axis_value = 0;
    int start_value = 0;
    int stride_value = 0;
    if (!GetConstInputValue(&axis_value, &start_value, &stride_value)) {
      return RET_ERROR;
    }
    if (axis_value < 0) {
      axis_value += input_dims.nbDims;
    }
    start_dims_.nbDims = input_dims.nbDims;
    std::fill(start_dims_.d, start_dims_.d + start_dims_.nbDims, 0);
    stride_dims_.nbDims = input_dims.nbDims;
    std::fill(stride_dims_.d, stride_dims_.d + stride_dims_.nbDims, 1);
    size_dims_ = slice_input->trt_tensor_->getDimensions();
    if (start_value < 0) {
      start_value = input_dims.d[axis_value] + start_value;
    }
    for (int i = 0; i < start_dims_.nbDims; i++) {
      if (i == axis_value) {
        start_dims_.d[i] = start_value;
        stride_dims_.d[i] = stride_value;
        if (end.IsConst()) {
          auto end_vec = ConvertTensorAsIntVector(end);
          if (end_vec.size() != 1) {
            MS_LOG(ERROR) << "Failed to get constant end value, end size " << end_vec.size() << ", op: " << op_name_;
            return false;
          }
          int end_value = end_vec[0];
          if (end_value >= 0) {
            size_dims_.d[i] = std::min(end_value, input_dims.d[i]) - start_dims_.d[i];
          } else if (end_value >= -input_dims.d[i]) {
            size_dims_.d[i] = end_value + input_dims.d[i] - start_dims_.d[i];
          } else {
            size_dims_.d[i] = input_dims.d[i];
          }
        }
      }
    }
    if (IsDynamicInput(ctx, 0)) {
      size_tensor_ =
        GetDynamicAxisSliceSize(ctx, slice_input->trt_tensor_, size_dims_.d[axis_value], axis_value, nullptr);
      size_dims_ = nvinfer1::Dims{-1};
    }
    if (!end.IsConst()) {
      auto start_tensor = ctx->ConvertTo1DTensor(start_value);
      auto len_tensor =
        ctx->network()
          ->addElementWise(*input(ctx, INPUT_SIZE2).trt_tensor_, *start_tensor, nvinfer1::ElementWiseOperation::kSUB)
          ->getOutput(0);
      size_tensor_ = GetDynamicAxisSliceSize(ctx, slice_input->trt_tensor_, -1, axis_value, len_tensor);
      size_dims_ = nvinfer1::Dims{-1};
    }
  }
  return RET_OK;
}

int StrideSliceTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper slice_input;
  auto in_tensor = input(ctx, 0);
  if (in_tensors_[0].IsConst() && in_tensor.trt_tensor_ == nullptr) {
    in_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx, in_tensors_[0], op_name_);
    in_tensor.format_ = Format::NCHW;
    ctx->RegisterTensor(in_tensor, in_tensors_[0].Name());
  }

  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &slice_input);
  if (ret != RET_OK || slice_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return RET_ERROR;
  }

  if (ComputeSliceDims(ctx, &slice_input) != RET_OK) {
    return RET_ERROR;
  }
  nvinfer1::ISliceLayer *slice_layer =
    ctx->network()->addSlice(*slice_input.trt_tensor_, start_dims_, size_dims_, stride_dims_);
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
    return RET_ERROR;
  }
  if (size_tensor_ != nullptr) {
    slice_layer->setInput(INPUT_SIZE2, *size_tensor_);
  }
  this->layer_ = slice_layer;
  slice_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
  auto shape = ConvertMSShape(out_tensor->getDimensions());
  bool rank_0 = false;
  if (shrink_axis_ != 0) {
    for (int i = SizeToInt(shape.size()) - 1; i >= 0; --i) {
      int mask = 1 << i;
      if ((shrink_axis_ & mask) != 0) {
        shape.erase(shape.begin() + i);
      }
    }
    if (!shape.empty()) {
      out_tensor = Reshape(ctx, out_tensor, shape);
    } else {
      rank_0 = true;
    }
  }
  auto helper = ITensorHelper{out_tensor, slice_input.format_, slice_input.same_format_, !rank_0};
  ctx->RegisterTensor(helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "slice output : " << GetTensorFormat(helper);
  return RET_OK;
}

REGISTER_TENSORRT_CREATOR(ops::kNameStridedSlice, StrideSliceTensorRT)
}  // namespace mindspore::lite
