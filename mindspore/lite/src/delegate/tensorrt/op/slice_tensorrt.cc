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

#include "src/delegate/tensorrt/op/slice_tensorrt.h"
#include <algorithm>
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int SliceTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                             const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() < HAS_AXIS - 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (in_tensors_[BEGIN_INDEX].Data() == nullptr) {
    MS_LOG(ERROR) << "invalid pad or stride tensor for: " << op_name_;
    return RET_ERROR;
  }
  dynamic_shape_params_.support_dynamic_ = false;
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

int SliceTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  auto slice_primitive = this->GetPrimitive()->value_as_StridedSlice();
  if (slice_primitive == nullptr) {
    MS_LOG(ERROR) << "convert StridedSlice failed: " << op_name_;
    return RET_ERROR;
  }
  strides_index_ = in_tensors_.size() - 1;
  axis_index_ = in_tensors_.size() == HAS_AXIS ? AXIS_INDEX : -1;

  ITensorHelper slice_input;
  int ret = PreprocessInputs2SameDim(network, tensorrt_in_tensors_[0], &slice_input);
  if (ret != RET_OK || slice_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return RET_ERROR;
  }
  ret = ConvertParamsDims();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvertParamsDims failed for " << op_name_;
    return ret;
  }

  nvinfer1::ISliceLayer *slice_layer =
    network->addSlice(*slice_input.trt_tensor_, start_dims_, size_dims_, stride_dims_);
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "add Slice op failed for TensorRT: " << op_name_;
    return RET_ERROR;
  }
  slice_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = slice_layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "output tensor create failed";
    return RET_ERROR;
  }
  out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{out_tensor, slice_input.format_, slice_input.same_format_});
  return RET_OK;
}

int SliceTensorRT::ConvertParamsDims() {
  const mindspore::MSTensor &begin = in_tensors_[BEGIN_INDEX];
  const mindspore::MSTensor &stride = in_tensors_[strides_index_];
  if (static_cast<size_t>(begin.ElementNum()) == in_tensors_[0].Shape().size()) {
    start_dims_ = lite::ConvertCudaDims(begin.Data().get(), begin.ElementNum());
    size_dims_ = lite::ConvertCudaDims(out_tensors_[0].Shape());
    stride_dims_ = lite::ConvertCudaDims(stride.Data().get(), stride.ElementNum());
  } else {
    if (axis_index_ == -1 || in_tensors_[axis_index_].ElementNum() != 1) {
      MS_LOG(ERROR) << "invalid input params for " << op_name_;
      return RET_ERROR;
    }
    int axis_value = *(static_cast<int *>(in_tensors_[axis_index_].MutableData()));
    int start_value = *(static_cast<int *>(in_tensors_[BEGIN_INDEX].MutableData()));
    start_dims_.nbDims = tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims;
    for (int i = 0; i < start_dims_.nbDims; i++) {
      start_dims_.d[i] = (i == axis_value) ? start_value : 0;
    }

    size_dims_ = lite::ConvertCudaDims(out_tensors_[0].Shape());
    int stride_value = *(static_cast<const int *>(stride.Data().get()));
    stride_dims_ = nvinfer1::Dims{size_dims_.nbDims, {}};
    std::fill(stride_dims_.d, stride_dims_.d + stride_dims_.nbDims, stride_value);
  }
  if (start_dims_.nbDims == -1 || size_dims_.nbDims == -1 || stride_dims_.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
