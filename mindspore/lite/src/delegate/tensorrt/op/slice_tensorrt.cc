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
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int SliceTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                             const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() < STRIDE_INDEX + 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (in_tensors_[BEGIN_INDEX].Data() == nullptr || in_tensors_[STRIDE_INDEX].Data() == nullptr) {
    MS_LOG(ERROR) << "invalid pad or stride tensor for: " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int SliceTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  auto slice_primitive = this->GetPrimitive()->value_as_StridedSlice();
  if (slice_primitive == nullptr) {
    MS_LOG(ERROR) << "convert StridedSlice failed: " << op_name_;
    return RET_ERROR;
  }
  const mindspore::MSTensor &begin = in_tensors_[BEGIN_INDEX];
  const mindspore::MSTensor &stride = in_tensors_[STRIDE_INDEX];

  nvinfer1::Dims start_dims = lite::ConvertCudaDims(begin.Data().get(), begin.ElementNum());
  nvinfer1::Dims size_dims = lite::ConvertCudaDims(out_tensors_[0].Shape());
  nvinfer1::Dims stride_dims = lite::ConvertCudaDims(stride.Data().get(), stride.ElementNum());

  nvinfer1::ITensor *slice_input = tensorrt_in_tensors_[0].trt_tensor_;
  Format out_format = tensorrt_in_tensors_[0].format_;
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NCHW) {
    // transpose: NCHW->NHWC
    nvinfer1::IShuffleLayer *transpose_layer_in = NCHW2NHWC(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "op action convert failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NHWC").c_str());
    slice_input = transpose_layer_in->getOutput(0);
    out_format = Format::NHWC;
  }

  nvinfer1::ISliceLayer *slice_layer = network->addSlice(*slice_input, start_dims, size_dims, stride_dims);
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
  this->AddInnerOutTensors(ITensorHelper{out_tensor, out_format});
  return RET_OK;
}
}  // namespace mindspore::lite
