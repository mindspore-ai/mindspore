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
Status AscendGeExecutorPluginImpl::AdaptGraph(FuncGraphPtr graph) const { return GeUtils::AdaptGraph(graph); }

bool AscendGeExecutorPluginImpl::AoeTuning(const FuncGraphPtr &graph,
                                           const std::shared_ptr<mindspore::Context> &context,
                                           const ConfigInfos &config_infos) {
  auto ge_graph_executor = InitGeGraphExecutor(context, config_infos);
  if (ge_graph_executor == nullptr) {
    return false;
  }
  return ge_graph_executor->AoeTuning(graph);
}

bool AscendGeExecutorPluginImpl::OfflineBuildGraph(const FuncGraphPtr &graph,
                                                   const std::shared_ptr<mindspore::Context> &context,
                                                   const ConfigInfos &config_infos) {
  auto ge_graph_executor = InitGeGraphExecutor(context, config_infos);
  if (ge_graph_executor == nullptr) {
    MS_LOG(ERROR) << "Failed to InitGeGraphExecutor";
    return false;
  }
  return ge_graph_executor->OfflineBuildGraph(graph);
}

std::shared_ptr<GeGraphExecutor> AscendGeExecutorPluginImpl::InitGeGraphExecutor(
  const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "Parameter context cannot be nullptr";
    return nullptr;
  }
  auto ge_graph_executor = std::make_shared<mindspore::GeGraphExecutor>(context, config_infos);
  if (ge_graph_executor == nullptr) {
    MS_LOG(ERROR) << "Failed to create GeGraphExecutor";
    return nullptr;
  }
  if (!ge_graph_executor->Init()) {
    MS_LOG(ERROR) << "Failed to init ge graph executor";
    return nullptr;
  }
  return ge_graph_executor;
}

AscendGeExecutorPluginImpl *CreateAscendGeExecutorPluginImpl() { return new AscendGeExecutorPluginImpl(); }
}  // namespace mindspore
