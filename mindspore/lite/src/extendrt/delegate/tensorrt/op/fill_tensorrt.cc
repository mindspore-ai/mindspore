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

#include <numeric>
#include "src/extendrt/delegate/tensorrt/op/fill_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int FillTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                            const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
#if TRT_VERSION_GE(8, 2)
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }

  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
#else
  MS_LOG(WARNING) << "low TensorRT version don't support fill op, please upgrade TensorRT version to 8.2 or higher";
  return RET_ERROR;
#endif
}

int FillTensorRT::AddInnerOp(TensorRTContext *ctx) {
#if TRT_VERSION_GE(8, 2)
  ITensorHelper fill_input;
  nvinfer1::FillOperation op = nvinfer1::FillOperation::kLINSPACE;
  auto *fill_layer = ctx->network()->addFill({}, op);
  if (fill_layer == nullptr) {
    MS_LOG(ERROR) << "addFill failed for TensorRT : " << op_name_;
    return RET_ERROR;
  }
  fill_layer->setInput(0, *input(ctx, 1).trt_tensor_);
  auto alpha_tensor =
    ConvertScalarToITensor(ctx, 0, in_tensors_[0].Data().get(), in_tensors_[0].DataType(), op_name_ + "_alpha");
  fill_layer->setInput(1, *alpha_tensor);
  int nbdims = input(ctx, 1).trt_tensor_->getDimensions().d[0];
  zeros_ = std::vector<float>(nbdims, 0.f);
  nvinfer1::Dims beta_dims{1, {nbdims}};
  nvinfer1::Weights weights{ConvertDataType(DataType::kNumberTypeFloat32), &zeros_[0], nbdims};
  auto beta_tensor = ctx->network()->addConstant(beta_dims, weights)->getOutput(0);
  fill_layer->setInput(INPUT_SIZE2, *beta_tensor);

  nvinfer1::ITensor *out_tensor = fill_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NHWC, true}, out_tensors_[0].Name());
  this->layer_ = fill_layer;
  return RET_OK;
#else
  MS_LOG(WARNING) << "low TensorRT version don't support fill op, please upgrade TensorRT version to 8.2 or higher";
  return RET_ERROR;
#endif
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Fill, FillTensorRT)
}  // namespace mindspore::lite
