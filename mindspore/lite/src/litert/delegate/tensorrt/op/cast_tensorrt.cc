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

#include "src/litert/delegate/tensorrt/op/cast_tensorrt.h"
#include "src/litert/delegate/tensorrt/op/cast_plugin.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>

namespace mindspore::lite {
int CastTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int CastTensorRT::AddInnerOp(TensorRTContext *ctx) {
  // cast to type tensor
  auto type_tensor = in_tensors_[1];
  if (type_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "unknown cast type of " << op_name_;
    return RET_ERROR;
  }
  auto type_data = static_cast<const int *>(type_tensor.Data().get());
  DataType data_type = static_cast<DataType>(type_data[0]);
  MS_LOG(DEBUG) << op_name_ << " cast to data type(43 float): " << type_data[0];
  nvinfer1::DataType dest_datatype = ConvertDataType(data_type);
  auto trt_tensor = input(ctx, 0).trt_tensor_;

#if TRT_VERSION_GE(7, 2)
  dest_datatype = (dest_datatype == nvinfer1::DataType::kBOOL ? nvinfer1::DataType::kINT32 : dest_datatype);
  auto cast_layer = ctx->network()->addIdentity(*trt_tensor);
#else
  auto plugin = std::make_shared<CastPlugin>(op_name_, trt_tensor->getType(), dest_datatype);
  nvinfer1::ITensor *inputTensors[] = {trt_tensor};
  nvinfer1::IPluginV2Layer *cast_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
#endif
  if (cast_layer == nullptr) {
    MS_LOG(ERROR) << "create cast layer failed for: " << op_name_;
    return RET_ERROR;
  }
#if TRT_VERSION_GE(7, 2)
  cast_layer->setOutputType(0, dest_datatype);
#endif
  cast_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *cast_out = cast_layer->getOutput(0);
  ctx->RegisterTensor(
    ITensorHelper{cast_out, input(ctx, 0).format_, input(ctx, 0).same_format_, input(ctx, 0).is_tensor_},
    out_tensors_[0].Name());
  this->layer_ = cast_layer;
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Cast, CastTensorRT)
}  // namespace mindspore::lite
