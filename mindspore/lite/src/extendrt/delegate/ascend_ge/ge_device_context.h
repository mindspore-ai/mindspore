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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_DEVICE_CONTEXT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_DEVICE_CONTEXT_H_

#include <string>
#include <memory>
#include <map>
#include <mutex>

#include "include/api/context.h"
#include "mindspore/core/utils/ms_context.h"

namespace mindspore {
class GeDeviceContext {
 public:
  GeDeviceContext(const GeDeviceContext &) = delete;
  GeDeviceContext &operator=(const GeDeviceContext &) = delete;

  static GeDeviceContext &GetInstance();
  void Initialize();
  void Destroy();

 private:
  GeDeviceContext() = default;
  ~GeDeviceContext() = default;
  void InitGe(const std::shared_ptr<MsContext> &inst_context);
  bool FinalizeGe(const std::shared_ptr<MsContext> &inst_context);
  void GetGeOptions(const std::shared_ptr<MsContext> &inst_context, std::map<std::string, std::string> *ge_options);
  void SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) const;

  int64_t call_num_ = 0;
  bool is_initialized_ = false;
  std::shared_ptr<MsContext> context_ = nullptr;
  std::mutex mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_DEVICE_CONTEXT_H_
