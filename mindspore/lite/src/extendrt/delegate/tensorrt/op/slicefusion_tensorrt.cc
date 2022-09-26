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

#include "src/extendrt/delegate/tensorrt/op/slicefusion_tensorrt.h"
#include <algorithm>
#include <utility>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/fusion/slice_fusion.h"

namespace mindspore::lite {
nvinfer1::ITensor *SliceFusionTensorRT::GetDynamicSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                                            nvinfer1::Dims start_dims, nvinfer1::Dims size_dims) {
  auto in_tensor_shape = ctx->network()->addShape(*input)->getOutput(0);
  if (in_tensor_shape == nullptr) {
    MS_LOG(ERROR) << "add shape layer of input failed!";
    return nullptr;
  }
  std::vector<nvinfer1::ITensor *> shape_tensors;
  auto input_dims = input->getDimensions();
  std::vector<int> input_shape_vec;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    if (input_dims.d[i] == -1) {
      if (!input_shape_vec.empty()) {
        shape_tensors.push_back(ctx->ConvertTo1DTensor(input_shape_vec));
        input_shape_vec.clear();
      }
      auto starts = nvinfer1::Dims{1, {i}};
      auto size = nvinfer1::Dims{1, {1}};
      auto strides = nvinfer1::Dims{1, {1}};
      auto slice_layer = ctx->network()->addSlice(*in_tensor_shape, starts, size, strides);
      if (slice_layer == nullptr) {
        MS_LOG(ERROR) << "add slice layer failed";
        return nullptr;
      }
      auto start_tensor = ctx->ConvertTo1DTensor(start_dims.d[i]);
      shape_tensors.push_back(
        ctx->network()
          ->addElementWise(*slice_layer->getOutput(0), *start_tensor, nvinfer1::ElementWiseOperation::kSUB)
          ->getOutput(0));
    } else {
      input_shape_vec.push_back(size_dims.d[i]);
    }
  }
  if (!input_shape_vec.empty()) {
    shape_tensors.push_back(ctx->ConvertTo1DTensor(input_shape_vec));
  }
  nvinfer1::ITensor *concat_tensors[shape_tensors.size()];
  for (size_t i = 0; i != shape_tensors.size(); ++i) {
    concat_tensors[i] = shape_tensors[i];
  }
  auto concat_layer = ctx->network()->addConcatenation(concat_tensors, shape_tensors.size());
  if (concat_layer == nullptr) {
    MS_LOG(ERROR) << "add concat layer failed!";
    return nullptr;
  }
  concat_layer->setAxis(0);

  return concat_layer->getOutput(0);
}

int SliceFusionTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                   const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != SLICE_INPUT_SIZE) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

int SliceFusionTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper slice_input;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &slice_input);
  if (ret != RET_OK || slice_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return RET_ERROR;
  }

  const auto &begin = in_tensors_.at(1);
  const auto &size = in_tensors_.at(SIZE_INDEX);

  auto start_dims = lite::ConvertCudaDims(begin);
  auto size_dims = lite::ConvertCudaDims(size);
  nvinfer1::ITensor *size_tensor = nullptr;
  for (int i = 0; i < size_dims.nbDims; ++i) {
    if (size_dims.d[i] == -1 && !IsDynamicInput(ctx, 0)) {
      size_dims.d[i] = slice_input.trt_tensor_->getDimensions().d[i];
    }
  }
  if (IsDynamicInput(ctx, 0)) {
    size_tensor = GetDynamicSliceSize(ctx, slice_input.trt_tensor_, start_dims, size_dims);
    size_dims = nvinfer1::Dims{-1};
  }
  auto stride_dims = lite::ConvertCudaDims(1, begin.ElementNum());

  nvinfer1::ISliceLayer *slice_layer =
    ctx->network()->addSlice(*slice_input.trt_tensor_, start_dims, size_dims, stride_dims);
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
    return RET_ERROR;
  }
  if (size_tensor != nullptr) {
    slice_layer->setInput(INPUT_SIZE2, *size_tensor);
  }
  this->layer_ = slice_layer;
  slice_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
  auto helper = ITensorHelper{out_tensor, slice_input.format_, slice_input.same_format_};
  ctx->RegisterTensor(helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "slice output : " << GetTensorFormat(helper);
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameSliceFusion, SliceFusionTensorRT)
}  // namespace mindspore::lite
