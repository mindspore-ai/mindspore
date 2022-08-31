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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_TENSORRT_EXECUTOR_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_TENSORRT_EXECUTOR_PLUGIN_H_
#include "include/api/status.h"
#include "utils/log_adapter.h"
#include "utils/macros.h"

namespace mindspore::lite {
class MS_API TensorRTExecutorPlugin {
 public:
  static TensorRTExecutorPlugin &GetInstance();
  bool Register();

  int GetGPUGroupSize();
  int GetRankID();

 private:
  TensorRTExecutorPlugin();
  ~TensorRTExecutorPlugin();

  void *handle_ = nullptr;
  bool is_registered_ = false;
  int group_size_ = 1;
  int rank_id_ = 0;
};

class TensorRTExecutorPluginImplBase {
 public:
  TensorRTExecutorPluginImplBase() = default;
  virtual ~TensorRTExecutorPluginImplBase() = default;
  virtual int GetGPUGroupSize() const = 0;
  virtual int GetRankID() const = 0;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_TENSORRT_EXECUTOR_PLUGIN_H_
