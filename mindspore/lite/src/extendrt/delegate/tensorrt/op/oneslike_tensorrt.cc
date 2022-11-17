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

#include <memory>
#include <functional>
#include "src/extendrt/delegate/tensorrt/op/oneslike_tensorrt.h"
#include "ops/ones_like.h"

namespace mindspore::lite {
int OneslikeTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int OneslikeTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  int input_nbdims = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  if (input_nbdims == -1) {
    MS_LOG(ERROR) << "oneslike op failed for " << op_name_;
    return RET_ERROR;
  }
  int ret = RunAsTrtOps(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "oneslike op failed for " << op_name_;
    return ret;
  }
  return ret;
}

template <typename T>
void *CreatConstTensorMem(int64_t element_num, const T value) {
  void *ptr = malloc(element_num * sizeof(T));
  if (ptr == nullptr) {
    MS_LOG(ERROR) << "malloc fail";
    return nullptr;
  }
  const T *begin = &(std::vector<T>(element_num, value))[0];
  memcpy(ptr, reinterpret_cast<const void *>(begin), element_num * sizeof(T));
  return ptr;
}

int OneslikeTensorRT::RunAsTrtOps(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto input_trt_tensor = input(ctx, 0).trt_tensor_;
  auto dims = input_trt_tensor->getDimensions();
  auto dtype = input_trt_tensor->getType();
  auto ms_shape = ConvertMSShape(dims);
  auto element_num = std::accumulate(ms_shape.begin(), ms_shape.end(), 1, std::multiplies<size_t>());
  void *ptr;
  if (dtype == nvinfer1::DataType::kFLOAT) {
    const float value = 1.0;
    ptr = CreatConstTensorMem(element_num, value);
  } else if (dtype == nvinfer1::DataType::kINT32) {
    const int value = 1;
    ptr = CreatConstTensorMem(element_num, value);
  } else {
    MS_LOG(ERROR) << "dtype not implement: " << dtype;
    return RET_ERROR;
  }
  nvinfer1::Weights weights{dtype, ptr, element_num};
  nvinfer1::IConstantLayer *oneslike_layer = ctx->network()->addConstant(dims, weights);

  CHECK_NULL_RETURN(oneslike_layer);
  auto out_tensor = oneslike_layer->getOutput(0);
  CHECK_NULL_RETURN(out_tensor);
  ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  oneslike_layer->setName(op_name_.c_str());
  this->layer_ = oneslike_layer;
  ctx->RegisterLayer(oneslike_layer, op_name_);
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameOnesLike, OneslikeTensorRT)
}  // namespace mindspore::lite
