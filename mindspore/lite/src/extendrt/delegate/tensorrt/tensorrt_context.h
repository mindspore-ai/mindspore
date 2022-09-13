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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_CONTEXT_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_CONTEXT_H_

#include <experimental/optional>
#include <NvInfer.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "src/extendrt/delegate/tensorrt/tensorrt_runtime.h"

namespace mindspore::lite {
struct ITensorHelper {
  nvinfer1::ITensor *trt_tensor_{nullptr};
  mindspore::Format format_{Format::NCHW};
  bool same_format_{true};
  bool is_tensor{true};
};
class TensorRTContext {
 public:
  TensorRTContext() = default;
  ~TensorRTContext();
  bool Init();
  void SetRuntime(TensorRTRuntime *runtime);
  nvinfer1::INetworkDefinition *network();
  void RegisterLayer(nvinfer1::ILayer *layer, const std::string &basename);
  void RegisterTensor(ITensorHelper tensor, const std::string &basename);
  void RegisterTensorWithSameName(ITensorHelper tensor, const std::string &basename);
  bool HasTensor(const std::string &name) const;
  ITensorHelper MsName2Tensor(const std::string &ms_name);

  template <typename T>
  nvinfer1::ITensor *ConvertTo1DTensor(T value);
  template <typename T>
  nvinfer1::ITensor *ConvertTo1DTensor(const std::vector<T> &values);

 private:
  int counter_{0};
  nvinfer1::INetworkDefinition *network_{nullptr};
  std::unordered_map<std::string, ITensorHelper> ms_name2trt_tensor_;
  TensorRTRuntime *runtime_{nullptr};
  std::vector<void *> owner_memorys_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_CONTEXT_H_
