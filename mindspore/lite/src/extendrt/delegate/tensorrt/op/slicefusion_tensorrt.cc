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

#include <algorithm>
#include <utility>
#include "src/extendrt/delegate/tensorrt/op/slicefusion_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
nvinfer1::ITensor *SliceFusionTensorRT::GetDynamicSliceSize(TensorRTContext *ctx, nvinfer1::ITensor *input,
                                                            const nvinfer1::Dims &start_dims,
                                                            const nvinfer1::Dims &size_dims) {
  auto in_tensor_shape = ctx->network()->addShape(*input)->getOutput(0);
  if (in_tensor_shape == nullptr) {
    MS_LOG(ERROR) << "add shape layer of input failed!";
    return nullptr;
  }
  std::vector<nvinfer1::ITensor *> shape_tensors;
  auto input_dims = input->getDimensions();
  std::vector<int> input_shape_vec;
  for (int i = 0; i != input_dims.nbDims; ++i) {
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
  auto concat_layer = ctx->network()->addConcatenation(shape_tensors.data(), shape_tensors.size());
  if (concat_layer == nullptr) {
    MS_LOG(ERROR) << "add concat layer failed!";
    return nullptr;
  }
  concat_layer->setAxis(0);

  return concat_layer->getOutput(0);
}

int SliceFusionTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                                   const std::vector<mindspore::MSTensor> &in_tensors,
                                   const std::vector<mindspore::MSTensor> &out_tensors) {
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

  auto start_dims = lite::ConvertCudaDims(begin.Data().get(), begin.ElementNum());
  auto size_dims =
    size.Data() == nullptr ? nvinfer1::Dims{0} : lite::ConvertCudaDims(size.Data().get(), size.ElementNum());
  nvinfer1::ITensor *size_tensor = nullptr;
  for (int i = 0; i != size_dims.nbDims; ++i) {
    if (size_dims.d[i] == -1 && !IsDynamicInput(ctx, 0)) {
      size_dims.d[i] = slice_input.trt_tensor_->getDimensions().d[i];
    }
  }
  if (IsDynamicInput(ctx, 0)) {
    size_tensor = GetDynamicSliceSize(ctx, slice_input.trt_tensor_, start_dims, size_dims);
    size_dims = nvinfer1::Dims{-1};
  }
  if (size.Data() == nullptr) {
#if TRT_VERSION_GE(7, 2)
    size_tensor = input(ctx, INPUT_SIZE2).trt_tensor_;
    auto shape_vec_int64 = in_tensors_[0].Shape();
    slice_input.trt_tensor_ = ConvertConstantTensor(ctx, in_tensors_[0], op_name_ + "_input");
    CHECK_NULL_RETURN(slice_input.trt_tensor_);
    std::vector<int> shape_vec_int32;
    std::copy(shape_vec_int64.begin(), shape_vec_int64.end(), std::back_inserter(shape_vec_int32));
    auto input_shape = ctx->ConvertTo1DTensor(shape_vec_int32);
    CHECK_NULL_RETURN(input_shape);
    auto minus_one = ctx->ConvertTo1DTensor(-1);
    auto eq_minus_one =
      ctx->network()->addElementWise(*size_tensor, *minus_one, nvinfer1::ElementWiseOperation::kEQUAL)->getOutput(0);
    auto int_eq_minus_one =
      TRTTensorCast(ctx, eq_minus_one, nvinfer1::DataType::kINT32, op_name_ + "_cast_int_mines_one");
    auto gr_minus_one =
      ctx->network()->addElementWise(*size_tensor, *minus_one, nvinfer1::ElementWiseOperation::kGREATER)->getOutput(0);
    auto int_gr_minus_one =
      TRTTensorCast(ctx, gr_minus_one, nvinfer1::DataType::kINT32, op_name_ + "_cast_int_ge_mines_one");
    auto x = ctx->network()
               ->addElementWise(*int_gr_minus_one, *size_tensor, nvinfer1::ElementWiseOperation::kPROD)
               ->getOutput(0);
    auto y = ctx->network()
               ->addElementWise(*int_eq_minus_one, *input_shape, nvinfer1::ElementWiseOperation::kPROD)
               ->getOutput(0);
    size_tensor = ctx->network()->addElementWise(*x, *y, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    size_dims = nvinfer1::Dims{-1};
#else
    MS_LOG(ERROR) << "Low version tensorrt don't support dynamic size for slice!";
    return RET_ERROR;
#endif
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
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_SliceFusion, SliceFusionTensorRT)
}  // namespace mindspore::lite
