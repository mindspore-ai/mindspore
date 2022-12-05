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

#include "src/extendrt/delegate/tensorrt/op/glu_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/glu.h"

namespace mindspore::lite {
int GLUTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int GLUTensorRT::AddInnerOp(TensorRTContext *ctx) {
  dim_ = AsOps<ops::GLU>()->get_axis();
  auto rank = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  dim_ = (dim_ != -1) ? dim_ : rank - 1;
  // split
  std::vector<int> split_dims(rank, 1);
  split_dims[dim_] = SPLITE_NUM;
  auto split_dims_tensor = ctx->ConvertTo1DTensor(split_dims);
  auto in_tensor_shape = ctx->network()->addShape(*input(ctx, 0).trt_tensor_)->getOutput(0);
  auto split_tensor1 = ctx->network()
                         ->addElementWise(*in_tensor_shape, *split_dims_tensor, nvinfer1::ElementWiseOperation::kDIV)
                         ->getOutput(0);
  nvinfer1::Dims starts{rank};
  std::fill(starts.d, starts.d + rank, 0);
  nvinfer1::Dims strides{rank};
  std::fill(strides.d, strides.d + rank, 1);
  nvinfer1::ISliceLayer *slice_layer = ctx->network()->addSlice(*input(ctx, 0).trt_tensor_, starts, {}, strides);
  slice_layer->setInput(INPUT_INDEX, *split_tensor1);
  auto input1 = slice_layer->getOutput(0);
  std::vector<int> start_mask(rank, 0);
  start_mask[dim_] = 1;
  auto start_dims_tensor = ctx->ConvertTo1DTensor(start_mask);
  nvinfer1::ISliceLayer *slice_layer2 = ctx->network()->addSlice(*input(ctx, 0).trt_tensor_, {}, {}, strides);
  auto start_tensor = ctx->network()
                        ->addElementWise(*split_tensor1, *start_dims_tensor, nvinfer1::ElementWiseOperation::kPROD)
                        ->getOutput(0);
  slice_layer2->setInput(1, *start_tensor);
  slice_layer2->setInput(INPUT_INDEX, *split_tensor1);
  auto input2 = slice_layer2->getOutput(0);
  // sigmoid
  auto sigmoid_tensor = ctx->network()->addActivation(*input2, nvinfer1::ActivationType::kSIGMOID)->getOutput(0);
  // mul
  auto mul_layer = ctx->network()->addElementWise(*input1, *sigmoid_tensor, nvinfer1::ElementWiseOperation::kPROD);
  auto out_tensor = mul_layer->getOutput(0);
  this->layer_ = mul_layer;
  ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameGLU, GLUTensorRT)
}  // namespace mindspore::lite
