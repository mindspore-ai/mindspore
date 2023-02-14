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
nvinfer1::ITensor *StrideSliceTensorRT::GetDynamicAxisSliceStart(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                                                 int axis, int nbdims) {
  if (axis == 0 && nbdims == 1) {
    return input;
  }
  std::vector<nvinfer1::ITensor *> gather_inputs;
  if (axis == 0) {
    gather_inputs.push_back(input);
    gather_inputs.push_back(ctx->ConvertTo1DTensor(std::vector<int>(nbdims - 1, 0)));
  } else if (axis == nbdims - 1) {
    gather_inputs.push_back(ctx->ConvertTo1DTensor(std::vector<int>(nbdims - 1, 0)));
    gather_inputs.push_back(input);
  } else {
    gather_inputs.push_back(ctx->ConvertTo1DTensor(std::vector<int>(axis, 0)));
    gather_inputs.push_back(input);
    gather_inputs.push_back(ctx->ConvertTo1DTensor(std::vector<int>(nbdims - 1 - axis, 0)));
  }
  auto concat_layer = ctx->network()->addConcatenation(gather_inputs.data(), gather_inputs.size());
  if (concat_layer == nullptr) {
    MS_LOG(ERROR) << "add concat layer failed!";
    return nullptr;
  }
  concat_layer->setAxis(0);
  return concat_layer->getOutput(0);
}

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

nvinfer1::ITensor *StrideSliceTensorRT::GetDynamicSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *slice_input,
                                                            size_t end_mask) {
  std::vector<int> end_mask_axis;
  std::vector<int> end_unmask_axis;
  std::vector<int> start_vector;
  for (int i = 0; i < size_dims_.nbDims; i++) {
    start_vector.push_back(start_dims_.d[i]);
    size_t mask = 1 << i;
    if ((end_mask & mask) == 0) {
      end_mask_axis.push_back(1);
      end_unmask_axis.push_back(0);
    } else {
      end_mask_axis.push_back(0);
      end_unmask_axis.push_back(1);
    }
  }
  auto end_mask_tensor = ctx->ConvertTo1DTensor(end_mask_axis);
  auto end_tensor =
    ctx->network()
      ->addElementWise(*input(ctx, INPUT_SIZE2).trt_tensor_, *end_mask_tensor, nvinfer1::ElementWiseOperation::kPROD)
      ->getOutput(0);
  auto end_unmask_tensor = ctx->ConvertTo1DTensor(end_unmask_axis);
  auto input_shape = ctx->network()->addShape(*slice_input)->getOutput(0);
  auto unmask_tensor = ctx->network()
                         ->addElementWise(*input_shape, *end_unmask_tensor, nvinfer1::ElementWiseOperation::kPROD)
                         ->getOutput(0);
  auto real_end_tensor =
    ctx->network()->addElementWise(*end_tensor, *unmask_tensor, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  auto start_tensor = ctx->ConvertTo1DTensor(start_vector);
  auto size_tensor =
    ctx->network()->addElementWise(*real_end_tensor, *start_tensor, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
  return size_tensor;
}

nvinfer1::ITensor *StrideSliceTensorRT::GetDynamicSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                                            const nvinfer1::Dims &size_dims,
                                                            const nvinfer1::Dims &start_dims) {
  auto in_tensor_shape = ctx->network()->addShape(*input)->getOutput(0);
  if (in_tensor_shape == nullptr) {
    MS_LOG(ERROR) << "add shape layer of input failed!";
    return nullptr;
  }
  std::vector<int> is_dynamic;
  std::vector<int> is_fix;
  std::vector<int> size_vec;
  std::vector<int> start_vec;
  for (int i = 0; i != size_dims.nbDims; ++i) {
    is_dynamic.push_back(size_dims.d[i] <= 0);
    is_fix.push_back(size_dims.d[i] > 0);
    size_vec.push_back(size_dims.d[i]);
    start_vec.push_back(start_dims.d[i]);
  }
  auto is_dynamic_tensor = ctx->ConvertTo1DTensor(is_dynamic);
  auto is_fix_tensor = ctx->ConvertTo1DTensor(is_fix);
  auto size_tensor = ctx->ConvertTo1DTensor(size_vec);
  auto start_tensor = ctx->ConvertTo1DTensor(start_vec);
  auto dynamic_in_tensor =
    ctx->network()->addElementWise(*in_tensor_shape, *start_tensor, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
  auto fix_tensor =
    ctx->network()->addElementWise(*is_fix_tensor, *size_tensor, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  auto dynamic_tensor =
    ctx->network()
      ->addElementWise(*is_dynamic_tensor, *dynamic_in_tensor, nvinfer1::ElementWiseOperation::kPROD)
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
  return RET_OK;
}

bool StrideSliceTensorRT::GetConstInputValue(int *start_val, int *stride_val) {
  const auto &begin = in_tensors_.at(BEGINS_INDEX);
  if (begin.IsConst()) {
    if (begin.ElementNum() != 1) {
      MS_LOG(ERROR) << "Only support element number of begin to be 1 when this number < input dims number, op: "
                    << op_name_;
      return false;
    }
    auto start_vec = ConvertTensorAsIntVector(begin);
    if (start_vec.size() != 1) {
      MS_LOG(ERROR) << "Failed to get start or stride input, node: " << op_name_;
      return false;
    }
    *start_val = start_vec[0];
  }
  const auto &stride = in_tensors_.back();
  if (stride.ElementNum() != 1) {
    MS_LOG(ERROR) << "Only support element number of stride to be 1 when this number < input dims number, op: "
                  << op_name_;
    return false;
  }
  auto stride_vec = ConvertTensorAsIntVector(stride);
  if (stride_vec.size() != 1) {
    MS_LOG(ERROR) << "Failed to get start or stride input, node: " << op_name_;
    return false;
  }
  *stride_val = stride_vec[0];
  return true;
}

int StrideSliceTensorRT::ComputeDimsMulti(TensorRTContext *ctx, ITensorHelper *slice_input, const TensorInfo &begin,
                                          const TensorInfo &stride, const TensorInfo &end, size_t start_mask,
                                          size_t end_mask) {
  auto input_dims = slice_input->trt_tensor_->getDimensions();
  start_dims_ = lite::ConvertCudaDims(begin);
  stride_dims_ = lite::ConvertCudaDims(stride);
  auto end_dims = lite::ConvertCudaDims(end);
  size_dims_.nbDims = input_dims.nbDims;
  for (int i = 0; i < size_dims_.nbDims; i++) {
    size_t mask = 1 << i;
    start_dims_.d[i] = ((start_mask & mask) == 0 ? start_dims_.d[i] : 0);
    if (start_dims_.d[i] < 0) {
      start_dims_.d[i] += input_dims.d[i];
    }
    if (end.IsConst()) {
      if ((end_mask & mask) != 0 && input_dims.d[i] > 0) {
        end_dims.d[i] = input_dims.d[i];
      } else if ((end_mask & mask) != 0) {
        size_dims_.d[i] = -1;
        continue;
      }
      if (end_dims.d[i] >= 0) {
        if (input_dims.d[i] >= 0) {
          size_dims_.d[i] = std::min(end_dims.d[i], input_dims.d[i]) - start_dims_.d[i];
        } else {
          size_dims_.d[i] = end_dims.d[i] - start_dims_.d[i];
        }
      } else if (end_dims.d[i] >= -input_dims.d[i]) {
        size_dims_.d[i] = end_dims.d[i] + input_dims.d[i] - start_dims_.d[i];
      } else {
        size_dims_.d[i] = input_dims.d[i];
      }
      if (size_dims_.d[i] < 0) {
        size_dims_.d[i] += input_dims.d[i];
        stride_dims_.d[i] = -stride_dims_.d[i];
      }
    }
  }
  return RET_OK;
}

int StrideSliceTensorRT::ComputeDimsSingle(TensorRTContext *ctx, ITensorHelper *slice_input, const TensorInfo &begin,
                                           const TensorInfo &stride, const TensorInfo &end, size_t start_mask,
                                           size_t end_mask) {
  auto input_dims = slice_input->trt_tensor_->getDimensions();

  int axis_value = GetAxis(ctx);
  int start_value = 0;
  int stride_value = 0;
  if (!GetConstInputValue(&start_value, &stride_value)) {
    return RET_ERROR;
  }
  stride_dims_.nbDims = input_dims.nbDims;
  std::fill(stride_dims_.d, stride_dims_.d + stride_dims_.nbDims, 1);
  stride_dims_.d[axis_value] = stride_value;

  if (!begin.IsConst() && !end.IsConst()) {
    return RET_OK;
  }

  if (start_value < 0) {
    start_value = input_dims.d[axis_value] + start_value;
  }
  start_dims_.nbDims = input_dims.nbDims;
  std::fill(start_dims_.d, start_dims_.d + start_dims_.nbDims, 0);
  start_dims_.d[axis_value] = start_value;

  if (!end.IsConst()) {
    return RET_OK;
  }
  int end_value = ConvertTensorAsIntVector(end)[0];
  size_dims_ = slice_input->trt_tensor_->getDimensions();
  for (int i = 0; i < start_dims_.nbDims; i++) {
    if (i == axis_value) {
      if (end_value >= 0) {
        size_dims_.d[i] = std::min(end_value, input_dims.d[i]) - start_dims_.d[i];
      } else if (end_value >= -input_dims.d[i]) {
        size_dims_.d[i] = end_value + input_dims.d[i] - start_dims_.d[i];
      } else {
        size_dims_.d[i] = input_dims.d[i];
      }
    }
  }
  DebugDims("size : ", size_dims_);
  return RET_OK;
}

int StrideSliceTensorRT::ComputeDims(TensorRTContext *ctx, ITensorHelper *slice_input, const TensorInfo &begin,
                                     const TensorInfo &stride, const TensorInfo &end, size_t start_mask,
                                     size_t end_mask) {
  if (static_cast<int>(begin.ElementNum()) == slice_input->trt_tensor_->getDimensions().nbDims) {
    return ComputeDimsMulti(ctx, slice_input, begin, stride, end, start_mask, end_mask);
  }
  return ComputeDimsSingle(ctx, slice_input, begin, stride, end, start_mask, end_mask);
}

int StrideSliceTensorRT::GetAxis(TensorRTContext *ctx) {
  int64_t axis_index = in_tensors_.size() == HAS_AXIS ? AXIS_INDEX : -1;
  int axis_value = 0;
  if (axis_index != -1) {
    auto axis_vec = ConvertTensorAsIntVector(in_tensors_[axis_index]);
    if (axis_vec.size() != 1) {
      MS_LOG(ERROR) << "Failed to get axis input, node: " << op_name_ << ", axis count: " << axis_vec.size();
      return -1;
    }
    axis_value = axis_vec[0];
  }
  if (axis_value < 0) {
    axis_value += input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  }
  return axis_value;
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
  if (begin.ElementNum() == input_dims.nbDims) {
    int dims_ret = ComputeDims(ctx, slice_input, begin, stride, end, start_mask, end_mask);
    if (dims_ret) {
      MS_LOG(ERROR) << "comput start dims, stride dims, size dims filed for " << op_name_;
      return RET_ERROR;
    }
    if (IsDynamicInput(ctx, 0) && end.IsConst()) {
      size_tensor_ = GetDynamicSliceSize(ctx, slice_input->trt_tensor_, size_dims_, start_dims_);
      size_dims_ = nvinfer1::Dims{-1};
    }
    if (!end.IsConst()) {
      size_tensor_ = GetDynamicSliceSize(ctx, slice_input->trt_tensor_, end_mask);
      size_dims_ = nvinfer1::Dims{-1};
    }
  } else {
    int axis_value = GetAxis(ctx);
    int dims_ret = ComputeDims(ctx, slice_input, begin, stride, end, start_mask, end_mask);
    if (dims_ret) {
      MS_LOG(ERROR) << "comput start dims, stride dims, size dims filed for " << op_name_;
      return RET_ERROR;
    }
    if (IsDynamicInput(ctx, 0) && begin.IsConst() && end.IsConst()) {
      size_tensor_ =
        GetDynamicAxisSliceSize(ctx, slice_input->trt_tensor_, size_dims_.d[axis_value], axis_value, nullptr);
      size_dims_ = nvinfer1::Dims{-1};
    }
    if (!begin.IsConst()) {
      start_tensor_ = GetDynamicAxisSliceStart(ctx, input(ctx, 1).trt_tensor_, axis_value, input_dims.nbDims);
      start_dims_ = nvinfer1::Dims{-1};
    }
    if (!end.IsConst()) {
      auto start_tensor = input(ctx, 1).trt_tensor_;
      if (start_tensor == nullptr) {
        auto start_vec = ConvertTensorAsIntVector(begin);
        int start_value = start_vec[0];
        if (start_value < 0) {
          start_value = slice_input->trt_tensor_->getDimensions().d[axis_value] + start_value;
        }
        start_tensor = ctx->ConvertTo1DTensor(start_value);
      }
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
  auto in_tensor = input(ctx, 0);
  if (in_tensors_[0].IsConst() && in_tensor.trt_tensor_ == nullptr) {
    in_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx, in_tensors_[0], op_name_);
    in_tensor.format_ = Format::NCHW;
    ctx->RegisterTensor(in_tensor, in_tensors_[0].Name());
  }

  if (ComputeSliceDims(ctx, &in_tensor) != RET_OK) {
    return RET_ERROR;
  }
  nvinfer1::ISliceLayer *slice_layer =
    ctx->network()->addSlice(*in_tensor.trt_tensor_, start_dims_, size_dims_, stride_dims_);
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
    return RET_ERROR;
  }
  if (start_tensor_ != nullptr) {
    slice_layer->setInput(1, *start_tensor_);
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
  auto helper = ITensorHelper{out_tensor, in_tensor.format_, in_tensor.same_format_, !rank_0};
  ctx->RegisterTensor(helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "slice output : " << GetTensorFormat(helper);
  return RET_OK;
}

REGISTER_TENSORRT_CREATOR(ops::kNameStridedSlice, StrideSliceTensorRT)
}  // namespace mindspore::lite
