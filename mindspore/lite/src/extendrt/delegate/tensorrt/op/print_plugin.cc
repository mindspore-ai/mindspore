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

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/op/print_plugin.h"

namespace mindspore::lite {
namespace {
template <typename T>
std::string Vector2String(T *arr, int num) {
  std::string result;
  for (int i = 0; i != num; ++i) {
    result += std::to_string(arr[i]) + " ";
  }
  return result;
}
}  // namespace

REGISTER_TENSORRT_PLUGIN(PrintPluginCreater);
template class TensorRTPluginCreater<PrintPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int PrintPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                         const void *const *inputs, void *const *outputs, void *workspace,
                         cudaStream_t stream) noexcept {
  return RunPrint(inputDesc, inputs, outputs, stream);
}

int PrintPlugin::RunPrint(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                          cudaStream_t stream) {
  int element_num = GetDimsVolume(inputDesc[0].dims);
  int sizeof_type = 4;
  cudaMemcpy(static_cast<void *>(outputs[0]), static_cast<const void *>(inputs[0]), element_num * sizeof_type,
             cudaMemcpyDeviceToDevice);
  if (inputDesc->type == nvinfer1::DataType::kHALF) {
    MS_LOG(ERROR) << "Only support int32 and float print";
    return RET_ERROR;
  }
  void *p = malloc(element_num * sizeof_type);
  cudaMemcpy(p, static_cast<const void *>(inputs[0]), element_num * sizeof_type, cudaMemcpyDeviceToHost);
  constexpr int max_num = 10;
  element_num = std::min(max_num, element_num);
  if (inputDesc->type == nvinfer1::DataType::kINT32) {
    int *ip = static_cast<int *>(p);
    MS_LOG(WARNING) << Vector2String(ip, element_num);
  }
  if (inputDesc->type == nvinfer1::DataType::kFLOAT) {
    float *fp = static_cast<float *>(p);
    MS_LOG(WARNING) << Vector2String(fp, element_num);
  }

  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *PrintPlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow) PrintPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "new plugin failed!";
    return nullptr;
  }
  return plugin;
}

size_t PrintPlugin::getSerializationSize() const noexcept { return 0; }

void PrintPlugin::serialize(void *buffer) const noexcept {}
}  // namespace mindspore::lite
