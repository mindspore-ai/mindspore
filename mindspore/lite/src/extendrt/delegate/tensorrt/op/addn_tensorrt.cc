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

#include "src/extendrt/delegate/tensorrt/op/addn_tensorrt.h"
#include <numeric>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/addn.h"

namespace mindspore::lite {
int AddNTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                            const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() <= 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }

  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int AddNTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto *add_layer = ctx->network()->addElementWise(*input(ctx, 0).trt_tensor_, *input(ctx, 1).trt_tensor_,
                                                   nvinfer1::ElementWiseOperation::kSUM);
  if (add_layer == nullptr) {
    MS_LOG(ERROR) << "addElementWise failed for TensorRT : " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::ITensor *out_tensor = add_layer->getOutput(0);
  for (size_t i = 2; i < in_tensors_.size(); ++i) {
    add_layer =
      ctx->network()->addElementWise(*out_tensor, *input(ctx, i).trt_tensor_, nvinfer1::ElementWiseOperation::kSUM);
  }
  this->layer_ = add_layer;
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameAddN, AddNTensorRT)
}  // namespace mindspore::lite
