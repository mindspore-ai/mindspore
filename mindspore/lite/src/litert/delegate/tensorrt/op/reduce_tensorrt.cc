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

#include <valarray>
#include <numeric>
#include "src/litert/delegate/tensorrt/op/reduce_tensorrt.h"

namespace mindspore::lite {
int ReduceTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
  }
  return RET_OK;
}

std::vector<int64_t> ReduceTensorRT::KeepDimsShape(TensorRTContext *ctx, nvinfer1::ITensor *out_tensor,
                                                   mindspore::MSTensor axis_tensor) {
  int *axis_data = reinterpret_cast<int *>(axis_tensor.MutableData());
  std::vector<int> axes;
  for (int i = 0; i < axis_tensor.ElementNum(); i++) {
    int format_axis_data =
      (*axis_data < 0) ? input(ctx, 0).trt_tensor_->getDimensions().nbDims + *axis_data : *axis_data;
    axes.push_back(format_axis_data);
    axis_data++;
  }
  std::vector<int64_t> squeeze_shape;
  for (int i = 0; i != out_tensor->getDimensions().nbDims; i++) {
    auto it = find(axes.begin(), axes.end(), i);
    if (it == axes.end()) {
      squeeze_shape.push_back(out_tensor->getDimensions().d[i]);
    }
  }
  return squeeze_shape;
}

int ReduceTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto reduce_op = op_primitive_->value_as_ReduceFusion();
  CHECK_NULL_RETURN(reduce_op);
  bool keep_dims = true;
  out_format_ = input(ctx, 0).format_;
  nvinfer1::ITensor *reduce_input = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "origin input " << GetTensorFormat(input(ctx, 0));
  if (input(ctx, 0).trt_tensor_->getDimensions().nbDims == DIMENSION_4D && !input(ctx, 0).same_format_ &&
      !in_tensors_[0].Shape().empty() &&
      !SameDims(input(ctx, 0).trt_tensor_->getDimensions(), in_tensors_[0].Shape())) {
    if (input(ctx, 0).format_ == Format::NCHW) {
      // NCHW->NHWC
      nvinfer1::IShuffleLayer *transpose_layer = NCHW2NHWC(ctx, *input(ctx, 0).trt_tensor_);
      CHECK_NULL_RETURN(transpose_layer);
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      reduce_input = transpose_layer->getOutput(0);
      out_format_ = Format::NHWC;
      this->transpose_layer_ = transpose_layer;
    } else if (input(ctx, 0).format_ == Format::NHWC) {
      // NHWC->NCHW
      nvinfer1::IShuffleLayer *transpose_layer = NHWC2NCHW(ctx, *input(ctx, 0).trt_tensor_);
      CHECK_NULL_RETURN(transpose_layer);
      transpose_layer->setName((op_name_ + "_transpose_in").c_str());
      reduce_input = transpose_layer->getOutput(0);
      out_format_ = Format::NCHW;
      this->transpose_layer_ = transpose_layer;
    } else {
      MS_LOG(WARNING) << "input tensor format needs check: " << op_name_;
    }
    MS_LOG(DEBUG) << "after transpose input " << GetTensorFormat(reduce_input, out_format_, true);
  }
  if (reduce_op->mode() == schema::ReduceMode::ReduceMode_ReduceL2) {
    // x^2
    auto *pow2_layer =
      ctx->network()->addElementWise(*reduce_input, *reduce_input, nvinfer1::ElementWiseOperation::kPROD);
    CHECK_NULL_RETURN(pow2_layer);
    pow2_layer->setName((op_name_ + "_pow2").c_str());
    reduce_input = pow2_layer->getOutput(0);
    CHECK_NULL_RETURN(reduce_input);
  }

  uint32_t reduceAxis = GetAxis(ctx);
  auto reduce_operation_opt = TryConvertTRTReduceMode(reduce_op->mode());
  if (!reduce_operation_opt) {
    MS_LOG(WARNING) << "invalid reduce for TensorRT, need check: " << static_cast<int>(reduce_op->mode());
    return RET_ERROR;
  }
  nvinfer1::IReduceLayer *layer =
    ctx->network()->addReduce(*reduce_input, reduce_operation_opt.value(), reduceAxis, keep_dims);
  CHECK_NULL_RETURN(layer);
  layer->setName(op_name_.c_str());
  this->layer_ = layer;

  nvinfer1::ITensor *out_tensor = layer->getOutput(0);
  keep_dims = reduce_op->keep_dims();
  // keep dims
  auto old_shape = ConvertMSShape(out_tensor->getDimensions());
  if (!keep_dims && old_shape.size() > 1 && in_tensors_.size() == INPUT_SIZE2) {
    mindspore::MSTensor axis_tensor = this->in_tensors_[1];
    CHECK_NULL_RETURN(axis_tensor.Data());
    if (axis_tensor.DataType() != DataType::kNumberTypeInt32) {
      MS_LOG(WARNING) << "not int data type";
    }
    std::vector<int64_t> squeeze_shape = KeepDimsShape(ctx, out_tensor, axis_tensor);
    auto squeeze = ctx->network()->addShuffle(*out_tensor);
    CHECK_NULL_RETURN(squeeze);
    squeeze->setName((op_name_ + "_squeeze_out").c_str());
    squeeze->setReshapeDimensions(ConvertCudaDims(squeeze_shape));
    out_tensor = squeeze->getOutput(0);
  }
  CHECK_NULL_RETURN(out_tensor);

  if (reduce_op->mode() == schema::ReduceMode::ReduceMode_ReduceL2) {
    auto sqrt_layer = ctx->network()->addUnary(*out_tensor, nvinfer1::UnaryOperation::kSQRT);
    CHECK_NULL_RETURN(sqrt_layer);
    sqrt_layer->setName((op_name_ + "_sqrt").c_str());
    out_tensor = sqrt_layer->getOutput(0);
  }
  auto output_helper = ITensorHelper{out_tensor, out_format_, true};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "output " << GetTensorFormat(output_helper);
  return RET_OK;
}

uint32_t ReduceTensorRT::GetAxis(TensorRTContext *ctx) {
  // axis
  uint32_t reduceAxis = 0;
  if (in_tensors_.size() == 1) {
    std::vector<int> axis;
    axis.resize(input(ctx, 0).trt_tensor_->getDimensions().nbDims);
    std::iota(axis.begin(), axis.end(), 0);
    for (int ax : axis) {
      MS_LOG(DEBUG) << op_name_ << " reduceAxis at index : " << ax;
      reduceAxis |= 1 << ax;
    }
  } else {
    //    uint32_t reduceAxis = 0;
    mindspore::MSTensor axis_tensor = this->in_tensors_[1];
    if (axis_tensor.Data() == nullptr) {
      MS_LOG(ERROR) << "invalid axis_tensor";
      return reduceAxis;
    }
    if (axis_tensor.DataType() != DataType::kNumberTypeInt32) {
      MS_LOG(WARNING) << "not int data type";
    }
    int *axis_data = reinterpret_cast<int *>(axis_tensor.MutableData());
    CHECK_NULL_RETURN(axis_data);
    for (int i = 0; i < axis_tensor.ElementNum(); i++) {
      int format_axis_data =
        (*axis_data < 0) ? input(ctx, 0).trt_tensor_->getDimensions().nbDims + *axis_data : *axis_data;
      MS_LOG(DEBUG) << op_name_ << " reduceAxis at index : " << format_axis_data;
      reduceAxis |= 1u << format_axis_data;
      axis_data++;
    }
  }
  return reduceAxis;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_ReduceFusion, ReduceTensorRT)
}  // namespace mindspore::lite
