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

#define USE_DEPRECATED_API
#include "tools/converter/anf_transform_for_ge.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <deque>
#include <map>
#include <tuple>
#include "nnacl/op_base.h"
#include "src/common/log_adapter.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "ir/primitive.h"
#include "include/registry/pass_registry.h"
#include "src/common/log_util.h"
#include "src/common/string_utils.h"
#include "src/common/config_infos.h"
#include "tools/converter/parser/parser_utils.h"
#include "include/backend/optimizer/graph_optimizer.h"
#include "mindspore/core/ops/op_name.h"
#include "tools/common/string_util.h"
#include "src/common/common.h"
#include "tools/optimizer/fusion/kv_cache_mgr_one_branch_fusion.h"
#include "tools/optimizer/fusion/kv_cache_mgr_concat_fusion.h"
#include "tools/optimizer/fusion/kv_cache_mgr_load_fusion.h"
#include "tools/optimizer/fusion/kv_cache_mgr_assign_fusion.h"
#include "tools/optimizer/fusion/flash_attention_fusion.h"
#include "tools/optimizer/graph/scalar_op_pass.h"
#include "tools/optimizer/graph/make_list_pass.h"

namespace mindspore::lite {
AnfTransformForGe::AnfTransformForGe() = default;

AnfTransformForGe::~AnfTransformForGe() = default;

int AnfTransformForGe::RunGeFusionPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto fusion_pm = std::make_shared<opt::LitePassManager>("anf fusion pass manager", false);
  CHECK_NULL_RETURN(fusion_pm);

  std::vector<opt::PassPtr> fusions{std::make_shared<opt::MakeListPass>(), std::make_shared<opt::ScalarOpPass>()};
  auto plugin_custom_ops = param->ascendGeOptionCfg.plugin_custom_ops;
  MS_LOG(INFO) << "plugin_custom_ops: " << plugin_custom_ops;
  if (find(plugin_custom_ops.begin(), plugin_custom_ops.end(), "All") != plugin_custom_ops.end() ||
      find(plugin_custom_ops.begin(), plugin_custom_ops.end(), "FlashAttention") != plugin_custom_ops.end()) {
    MS_LOG(INFO) << "using FlashAttention";
    fusions.push_back(std::make_shared<opt::FlashAttentionFusion>());
  }
  if (find(plugin_custom_ops.begin(), plugin_custom_ops.end(), "All") != plugin_custom_ops.end() ||
      find(plugin_custom_ops.begin(), plugin_custom_ops.end(), "KVCache") != plugin_custom_ops.end()) {
    MS_LOG(INFO) << "using KVCache";
    fusions.push_back(std::make_shared<opt::KVCacheMgrOneBranchFusion>());
    fusions.push_back(std::make_shared<opt::KVCacheMgrConcatFusion>());
    fusions.push_back(std::make_shared<opt::KVCacheMgrLoadFusion>());
    fusions.push_back(std::make_shared<opt::KVCacheMgrAssignFusion>());
  } else {
    MS_LOG(INFO) << "custom op fusion not used.";
  }

  for (size_t index = 0; index < fusions.size(); index++) {
    auto pass_ptr = fusions.at(index);
    MS_CHECK_TRUE_RET(pass_ptr != nullptr, RET_ERROR);
    auto pass_name = pass_ptr->name();
    if (param->fusion_blacklists.find(pass_name) != param->fusion_blacklists.end()) {
      MS_LOG(INFO) << "Disable fusion: " << pass_name;
      continue;
    }
    fusion_pm->AddPass(pass_ptr);
  }
  optimizer->AddPassManager(fusion_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run op fusion failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS AnfTransformForGe::Transform(const FuncGraphPtr &main_graph, const std::shared_ptr<ConverterPara> &param) {
  MS_CHECK_TRUE_MSG(main_graph != nullptr, RET_NULL_PTR, "Input func_graph is nullptr");
  MS_CHECK_TRUE_MSG(param != nullptr, RET_NULL_PTR, "Input converter param is nullptr");
  manager_ = Manage(main_graph, true);
  return RunGeFusionPass(main_graph, param);
}
}  // namespace mindspore::lite
