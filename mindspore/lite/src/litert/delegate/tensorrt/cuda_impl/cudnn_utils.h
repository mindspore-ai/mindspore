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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_CUDA_IMPL_CUDNN_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_CUDA_IMPL_CUDNN_UTILS_H_

#include <cudnn.h>
#include <NvInfer.h>
#include "src/litert/delegate/tensorrt/cuda_impl/cuda_helper.h"
#include "src/common/log_util.h"

#define CUDNN_CHECK_VOID(err)                                            \
  do {                                                                   \
    cudnnStatus_t cudnn_err = (err);                                     \
    if (cudnn_err != CUDNN_STATUS_SUCCESS) {                             \
      MS_LOG(ERROR) << "cudnn error " << cudnnGetErrorString(cudnn_err); \
      return;                                                            \
    }                                                                    \
  } while (0)

#define CUDNN_CHECK(err)                                                 \
  do {                                                                   \
    cudnnStatus_t cudnn_err = (err);                                     \
    if (cudnn_err != CUDNN_STATUS_SUCCESS) {                             \
      MS_LOG(ERROR) << "cudnn error " << cudnnGetErrorString(cudnn_err); \
      return -1;                                                         \
    }                                                                    \
  } while (0)
namespace mindspore::lite {
cudnnDataType_t ConvertCudnnDataType(nvinfer1::DataType trt_datatype);

int CudnnActivation(cudnnHandle_t handle, cudnnActivationDescriptor_t activation_desc,
                    const cudnnTensorDescriptor_t x_esc, const void *x, const cudnnTensorDescriptor_t y_dsc, void *y);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_CUDA_IMPL_CUDNN_UTILS_H_
