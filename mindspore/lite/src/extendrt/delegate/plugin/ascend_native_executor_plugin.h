/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_PLUGIN_ASCEND_NATIVE_EXECUTOR_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_PLUGIN_ASCEND_NATIVE_EXECUTOR_PLUGIN_H_

#include <string>
#include <memory>
#include "include/api/context.h"
#include "include/api/status.h"
#include "utils/log_adapter.h"
#include "mindapi/base/macros.h"
#include "base/base.h"
#include "common/config_infos.h"

namespace mindspore::lite {
class AscendNativeExecutorPluginImplBase {
 public:
  AscendNativeExecutorPluginImplBase() = default;
  virtual ~AscendNativeExecutorPluginImplBase() = default;
};

class MS_API AscendNativeExecutorPlugin {
 public:
  static AscendNativeExecutorPlugin &GetInstance();
  bool Register();

 private:
  AscendNativeExecutorPlugin();
  ~AscendNativeExecutorPlugin();

  std::string plugin_path_;
  void *handle_ = nullptr;
  bool is_registered_ = false;
  std::shared_ptr<AscendNativeExecutorPluginImplBase> ascend_native_plugin_impl_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_PLUGIN_ASCEND_NATIVE_EXECUTOR_PLUGIN_H_
