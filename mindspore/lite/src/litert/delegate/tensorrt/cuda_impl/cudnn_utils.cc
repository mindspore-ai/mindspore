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

#include "src/litert/delegate/tensorrt/cuda_impl/cudnn_utils.h"
#include <unordered_map>

namespace mindspore::lite {
cudnnDataType_t ConvertCudnnDataType(nvinfer1::DataType trt_datatype) {
  std::unordered_map<nvinfer1::DataType, cudnnDataType_t> data_types = {{nvinfer1::DataType::kFLOAT, CUDNN_DATA_FLOAT},
                                                                        {nvinfer1::DataType::kHALF, CUDNN_DATA_HALF},
                                                                        {nvinfer1::DataType::kINT32, CUDNN_DATA_INT32},
                                                                        {nvinfer1::DataType::kINT8, CUDNN_DATA_INT8}};
  if (data_types.find(trt_datatype) != data_types.end()) {
    return data_types[trt_datatype];
  } else {
    MS_LOG(ERROR) << "invalid datatype for cudnn: " << static_cast<int>(trt_datatype);
  }
  return CUDNN_DATA_FLOAT;
}

int CudnnActivation(cudnnHandle_t handle, cudnnActivationDescriptor_t activation_desc,
                    const cudnnTensorDescriptor_t x_dsc, const void *x, const cudnnTensorDescriptor_t y_dsc, void *y) {
  float alpha = 1.0f;
  float beta = 0.0f;
  CUDNN_CHECK(cudnnActivationForward(handle, activation_desc, &alpha, x_dsc, x, &beta, y_dsc, y));
  return 0;
}
}  // namespace mindspore::lite
