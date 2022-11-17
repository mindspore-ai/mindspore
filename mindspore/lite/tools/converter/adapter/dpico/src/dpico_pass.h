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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DPICO_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DPICO_PASS_H_

#include <vector>
#include <utility>
#include <memory>
#include <map>
#include <string>
#include "src/custom_creator.h"
#include "./op_enum_public.h"
#include "include/registry/pass_base.h"
#include "include/errorcode.h"
#include "src/graph_split_info.h"
#include "src/graph_split_api.h"

using mindspore::lite::STATUS;
namespace mindspore {
namespace dpico {
class DpicoPass : public registry::PassBase {
 public:
  DpicoPass() : PassBase("DpicoPass") {}
  ~DpicoPass() override = default;

  bool Execute(const api::FuncGraphPtr &func_graph) override;

 private:
  STATUS InitDpicoConfigInfo();
  void FetchFuncGraphs(const api::FuncGraphPtr &func_graph);
  STATUS CheckDynamicInputShape(const api::FuncGraphPtr &func_graph);
  STATUS MarkNodes(const api::FuncGraphPtr &func_graph);
  STATUS ParseMapperConfig(const api::FuncGraphPtr &func_graph);
  STATUS DataPrepare(const api::FuncGraphPtr &func_graph, bool *use_origin_config);
  STATUS ReplaceSubgraphWithCustom(const api::FuncGraphPtr &func_graph, bool use_origin_config);
  STATUS WriteOmBufferToFile(const std::shared_ptr<mapper::ModelCoreInfo> &om_model_info, size_t custom_id);
  STATUS RemoveTemporaryFiles();

  std::vector<api::FuncGraphPtr> func_graphs_;
  std::shared_ptr<CustomOpCreator> custom_op_creator_{nullptr};
  struct GraphSplitInfo graph_split_info_;
  bool save_tmp_files_{false};
  std::string dpico_config_path_{};
};
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_DPICO_PASS_H_
