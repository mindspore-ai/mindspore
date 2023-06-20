/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "common/config_infos.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "mindspore/core/utils/ms_context.h"

namespace mindspore {
class GeDeviceContext {
 public:
  GeDeviceContext();
  ~GeDeviceContext();

  GeDeviceContext(const GeDeviceContext &) = delete;
  GeDeviceContext &operator=(const GeDeviceContext &) = delete;

  static std::shared_ptr<GeDeviceContext> InitGlobalContext(const std::shared_ptr<Context> &context,
                                                            const ConfigInfos &config_info = {});

 private:
  Status Initialize(const std::shared_ptr<Context> &context, const ConfigInfos &config_info = {});
  void Destroy();

  Status InitGe(const std::shared_ptr<MsContext> &inst_context, const std::shared_ptr<Context> &context,
                const ConfigInfos &config_info = {});
  bool FinalizeGe(const std::shared_ptr<MsContext> &inst_context);
  Status InitHccl(const std::shared_ptr<Context> &context, const ConfigInfos &config_info);

  void GetGeOptions(const std::shared_ptr<MsContext> &inst_context, const std::shared_ptr<Context> &context,
                    std::map<std::string, std::string> *ge_options, const ConfigInfos &config_info = {});
  void SetHcclOptions(const std::shared_ptr<Context> &context, std::map<std::string, std::string> *ge_options,
                      const ConfigInfos &config_info = {});
  void SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) const;
  std::shared_ptr<AscendDeviceInfo> GetGeAscendDeviceInfo(const std::shared_ptr<Context> &context);

  static std::weak_ptr<GeDeviceContext> global_ge_context_;
  static std::mutex global_ge_context_mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_DEVICE_CONTEXT_H_
