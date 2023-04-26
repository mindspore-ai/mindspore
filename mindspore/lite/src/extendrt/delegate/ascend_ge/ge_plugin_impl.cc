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

#include <memory>
#include "extendrt/delegate/ascend_ge/ge_plugin_impl.h"
#include "extendrt/delegate/ascend_ge/ge_device_context.h"
#include "extendrt/delegate/ascend_ge/ge_utils.h"

namespace mindspore {
Status AscendGeExecutorPluginImpl::AscendGeDeviceContextInitialize(const std::shared_ptr<Context> &context,
                                                                   const ConfigInfos &config_info) {
  ge_context_ = std::make_shared<GeDeviceContext>();
  if (ge_context_ == nullptr) {
    MS_LOG(ERROR) << "Create GeDeviceContext failed.";
    return kLiteUninitializedObj;
  }
  ge_context_->Initialize(context, config_info);
  return kSuccess;
}

void AscendGeExecutorPluginImpl::AscendGeDeviceContextDestroy() const {
  if (ge_context_ != nullptr) {
    ge_context_->Destroy();
  }
}

Status AscendGeExecutorPluginImpl::AdaptGraph(FuncGraphPtr graph) const { return GeUtils::AdaptGraph(graph); }

AscendGeExecutorPluginImpl *CreateAscendGeExecutorPluginImpl() { return new AscendGeExecutorPluginImpl(); }
}  // namespace mindspore
