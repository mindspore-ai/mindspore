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

#include "src/extendrt/delegate/tensorrt/op/reverse_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/reverse_v2.h"

namespace mindspore::lite {
int ReverseTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                               const std::vector<TensorInfo> &out_tensors) {
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ReverseTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto input_helper = input(ctx, 0);
  auto dims = input_helper.trt_tensor_->getDimensions();
  for (int i = 0; i != dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      MS_LOG(ERROR) << "this version do not support dynamic reverse op : " << op_name_;
      return RET_ERROR;
    }
  }
  std::vector<nvinfer1::ITensor *> concat_inputs;
  auto reverse_op = AsOps<ops::ReverseV2>();
  auto axis = reverse_op->get_axis();
  if (axis.size() != 1) {
    MS_LOG(WARNING) << "reverse op has more than 1 axis for " << op_name_;
    return RET_ERROR;
  }
  for (int i = dims.d[axis[0]] - 1; i >= 0; --i) {
    nvinfer1::Dims start = nvinfer1::Dims{dims.nbDims, {}};
    std::fill(start.d, start.d + dims.nbDims, 0);
    start.d[axis[0]] = i;

    nvinfer1::Dims size = dims;
    size.d[axis[0]] = 1;

    nvinfer1::Dims stride = nvinfer1::Dims{dims.nbDims, {}};
    std::fill(stride.d, stride.d + dims.nbDims, 1);

    auto slice = ctx->network()->addSlice(*input_helper.trt_tensor_, start, size, stride)->getOutput(0);
    concat_inputs.push_back(slice);
  }

  auto concat_layer = ctx->network()->addConcatenation(concat_inputs.data(), concat_inputs.size());
  concat_layer->setAxis(axis[0]);
  this->layer_ = concat_layer;

  auto out_tensor = concat_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameReverseV2, ReverseTensorRT)
}  // namespace mindspore::lite
