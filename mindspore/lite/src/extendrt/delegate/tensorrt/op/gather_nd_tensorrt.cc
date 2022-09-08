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

#include "src/extendrt/delegate/tensorrt/op/gather_nd_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int GatherNDTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                                const std::vector<mindspore::MSTensor> &out_tensors) {
#if TRT_VERSION_GE(8, 2)
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (in_tensors[1].DataType() != DataType::kNumberTypeInt32) {
    MS_LOG(ERROR) << "Gather indices only support Int32";
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }

  return RET_OK;
#else
  MS_LOG(WARNING) << "low TensorRT version don't support gathernd op, please upgrade TensorRT version to 8.2 or higher";
  return RET_ERROR;
#endif
}

int GatherNDTensorRT::AddInnerOp(TensorRTContext *ctx) {
#if TRT_VERSION_GE(8, 2)
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  ITensorHelper gather_nd_input = input(ctx, 0);
  ITensorHelper indices_tensor = input(ctx, 1);
  if (indices_tensor.trt_tensor_->getDimensions().nbDims < 1) {
    MS_LOG(ERROR) << "addGather failed for TensorRT.";
    return RET_ERROR;
  }
  auto nbElementWiseDims = gather_nd_input.trt_tensor_->getDimensions().d[0];
  if (nbElementWiseDims == -1) {
    nbElementWiseDims = 0;
  }
  nvinfer1::IGatherLayer *gather_layer =
    ctx->network()->addGatherV2(*gather_nd_input.trt_tensor_, *indices_tensor.trt_tensor_, nvinfer1::GatherMode::kND);
  if (gather_layer == nullptr) {
    MS_LOG(ERROR) << "addGatherND failed for TensorRT.";
    return RET_ERROR;
  }
  gather_layer->setNbElementWiseDims(nbElementWiseDims);
  this->layer_ = gather_layer;
  gather_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *op_output = gather_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{op_output, gather_nd_input.format_, gather_nd_input.same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
#else
  MS_LOG(WARNING) << "low TensorRT version don't support gathernd op, please upgrade TensorRT version to 8.2 or higher";
  return RET_ERROR;
#endif
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_GatherNd, GatherNDTensorRT)
}  // namespace mindspore::lite
