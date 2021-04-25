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
#ifndef MINDSPORE_CCSRC_BACKEND_RUNTIME_DEVICE_GPU_TRT_LOADER_H_
#define MINDSPORE_CCSRC_BACKEND_RUNTIME_DEVICE_GPU_TRT_LOADER_H_

#include <memory>
#include <NvInfer.h>

namespace mindspore {
namespace device {
namespace gpu {
typedef nvinfer1::IBuilder *(*CreateInferBuilder_t)(nvinfer1::ILogger &, int);
typedef nvinfer1::IRuntime *(*CreateInferRuntime_t)(nvinfer1::ILogger &, int);
class TrtLoader {
 public:
  TrtLoader();
  ~TrtLoader();

  std::shared_ptr<nvinfer1::IBuilder> CreateInferBuilder(nvinfer1::ILogger *logger);
  std::shared_ptr<nvinfer1::IRuntime> CreateInferRuntime(nvinfer1::ILogger *logger);
  bool nvinfer_loaded() const { return nvinfer_loaded_; }

 private:
  bool nvinfer_loaded_;
  void *nvinfer_handle_;
  CreateInferBuilder_t create_infer_builder_;
  CreateInferRuntime_t create_infer_runtime_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_RUNTIME_DEVICE_GPU_TRT_LOADER_H_
