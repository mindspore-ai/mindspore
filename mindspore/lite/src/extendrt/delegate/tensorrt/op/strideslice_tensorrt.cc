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

#include "src/extendrt/delegate/tensorrt/op/strideslice_tensorrt.h"
#include <algorithm>
#include <numeric>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/strided_slice.h"

namespace mindspore::lite {
nvinfer1::ITensor *StrideSliceTensorRT::GetDynamicSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                                            int size_dim, int axis) {
  auto in_tensor_shape = ctx->network()->addShape(*input)->getOutput(0);
  if (in_tensor_shape == nullptr) {
    MS_LOG(ERROR) << "add shape layer of input failed!";
    return nullptr;
  }
  auto len_tensor = ctx->ConvertTo1DTensor(static_cast<int>(size_dim));
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
  if (!in_tensors.at(BEGINS_INDEX).IsConst() || !in_tensors.at(ENDS_INDEX).IsConst()) {
    MS_LOG(ERROR) << "invalid input tensor for: " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

nvinfer1::ILayer *StrideSliceTensorRT::MakeLayer(TensorRTContext *ctx, const ITensorHelper &slice_input) {
  const auto &begin = in_tensors_.at(BEGINS_INDEX);
  const auto &stride = in_tensors_.back();
  const auto &end = in_tensors_.at(ENDS_INDEX);

  auto input_dims = slice_input.trt_tensor_->getDimensions();
  nvinfer1::Dims start_dims{input_dims};
  nvinfer1::Dims size_dims{input_dims};
  nvinfer1::Dims stride_dims{input_dims};

  int64_t axis_index = in_tensors_.size() == HAS_AXIS ? AXIS_INDEX : -1;
  nvinfer1::ITensor *size_tensor = nullptr;
  if (begin.ElementNum() == input_dims.nbDims) {
    start_dims = lite::ConvertCudaDims(begin);
    auto end_dims = lite::ConvertCudaDims(end);
    for (int i = 0; i < size_dims.nbDims; i++) {
      size_dims.d[i] = end_dims.d[i] - start_dims.d[i];
    }
    stride_dims = lite::ConvertCudaDims(stride);
  } else {
    if (begin.ElementNum() != 1 || end.ElementNum() != 1 || stride.ElementNum() != 1) {
      MS_LOG(ERROR)
        << "Only support element number of begin, end and stride to be 1 when this number < input dims number, op: "
        << op_name_;
      return nullptr;
    }
    int axis_value = axis_index == -1 ? 0 : *(static_cast<const int *>(in_tensors_.at(axis_index).Data()));
    int start_value = *(static_cast<const int *>(begin.Data()));
    int end_value = *(static_cast<const int *>(end.Data()));
    int stride_value = *(static_cast<const int *>(stride.Data()));
    std::fill(start_dims.d, start_dims.d + start_dims.nbDims, 0);
    std::fill(stride_dims.d, stride_dims.d + stride_dims.nbDims, 1);
    size_dims = slice_input.trt_tensor_->getDimensions();
    if (start_value < 0) {
      start_value = input_dims.d[axis_value] + start_value;
    }
    for (int i = 0; i < start_dims.nbDims; i++) {
      if (i == axis_value) {
        start_dims.d[i] = start_value;
        stride_dims.d[i] = stride_value;
        if (end_value >= 0) {
          size_dims.d[i] = std::min(end_value, input_dims.d[i]) - start_dims.d[i];
        } else if (end_value >= -input_dims.d[i]) {
          size_dims.d[i] = end_value + input_dims.d[i] - start_dims.d[i];
        } else {
          size_dims.d[i] = input_dims.d[i];
        }
      }
    }
    if (IsDynamicInput(ctx, 0)) {
      size_tensor = GetDynamicSliceSize(ctx, slice_input.trt_tensor_, size_dims.d[axis_value], axis_value);
      size_dims = nvinfer1::Dims{-1};
    }
  }
  nvinfer1::ISliceLayer *slice_layer =
    ctx->network()->addSlice(*slice_input.trt_tensor_, start_dims, size_dims, stride_dims);
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
    return nullptr;
  }
  if (size_tensor != nullptr) {
    slice_layer->setInput(INPUT_SIZE2, *size_tensor);
  }
  return slice_layer;
}

int StrideSliceTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper slice_input;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &slice_input);
  if (ret != RET_OK || slice_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return RET_ERROR;
  }

  auto slice_layer = MakeLayer(ctx, slice_input);
  if (slice_layer == nullptr) {
    return RET_ERROR;
  }
  this->layer_ = slice_layer;
  slice_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
  auto shape = ConvertMSShape(out_tensor->getDimensions());
  bool rank_0 = false;

  shrink_axis_ = AsOps<ops::StridedSlice>()->get_shrink_axis_mask();
  if (shrink_axis_ != 0) {
    for (int i = shape.size() - 1; i >= 0; --i) {
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
