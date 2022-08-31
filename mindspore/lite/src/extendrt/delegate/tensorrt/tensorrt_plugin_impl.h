/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_TENSORRT_PLUGIN_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_TENSORRT_PLUGIN_IMPL_H_
#include "include/api/status.h"
#include "utils/log_adapter.h"
#include "extendrt/delegate/plugin/tensorrt_executor_plugin.h"

namespace mindspore::lite {
class TensorRTPluginImpl : public TensorRTExecutorPluginImplBase {
 public:
  TensorRTPluginImpl() = default;
  ~TensorRTPluginImpl() = default;
  int GetGPUGroupSize() const;
  int GetRankID() const;
};
}  // namespace mindspore::lite

extern "C" MS_API mindspore::lite::TensorRTExecutorPluginImplBase *CreateTensorRTPluginImpl();
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ASCEND_KERNEL_API_H_
