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

#include "runtime/device/gpu/trt_loader.h"

#include <dlfcn.h>
#include <memory>
#include <NvInferRuntimeCommon.h>
#include "backend/kernel_compiler/gpu/trt/trt_utils.h"

namespace mindspore {
namespace device {
namespace gpu {
TrtLoader::TrtLoader()
    : nvinfer_loaded_(false), nvinfer_handle_(nullptr), create_infer_builder_(nullptr), create_infer_runtime_(nullptr) {
  nvinfer_handle_ = dlopen("libnvinfer.so.7", RTLD_NOW | RTLD_LOCAL);
  if (nvinfer_handle_ == nullptr) {
    MS_LOG(WARNING) << "Can not open libnvinfer.so.7 " << dlerror()
                    << ". Install Tensor-RT and export LD_LIBRARY_PATH=${TENSORRT_HOME}/lib:$LD_LIBRARY_PATH.";
    return;
  }

  create_infer_builder_ = (CreateInferBuilder_t)dlsym(nvinfer_handle_, "createInferBuilder_INTERNAL");
  if (create_infer_builder_ == nullptr) {
    MS_LOG(WARNING) << "Failed to get createInferBuilder_INTERNAL symbol. " << dlerror();
    return;
  }

  create_infer_runtime_ = (CreateInferRuntime_t)dlsym(nvinfer_handle_, "createInferRuntime_INTERNAL");
  if (create_infer_runtime_ == nullptr) {
    MS_LOG(WARNING) << "Failed to get createInferRuntime_INTERNAL symbol. " << dlerror();
    return;
  }

  nvinfer_loaded_ = true;
}

TrtLoader::~TrtLoader() {
  if (nvinfer_handle_ != nullptr) {
    dlclose(nvinfer_handle_);
  }
}

std::shared_ptr<nvinfer1::IBuilder> TrtLoader::CreateInferBuilder(nvinfer1::ILogger *logger) {
  return TrtPtr<nvinfer1::IBuilder>(create_infer_builder_(*logger, NV_TENSORRT_VERSION));
}

std::shared_ptr<nvinfer1::IRuntime> TrtLoader::CreateInferRuntime(nvinfer1::ILogger *logger) {
  return TrtPtr<nvinfer1::IRuntime>(create_infer_runtime_(*logger, NV_TENSORRT_VERSION));
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
