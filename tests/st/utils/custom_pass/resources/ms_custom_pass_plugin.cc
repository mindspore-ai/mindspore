/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "mindspore/core/include/utils/log_adapter.h"
#include "mindspore/ccsrc/include/utils/custom_pass/custom_pass_plugin.h"
#include "replace_addn_fusion/replace_addn_fusion_pass.h"
#include "add_neg_fusion/add_neg_fusion_pass.h"
#include "full_graph/full_graph_pass.h"

namespace mindspore {
namespace opt {

// Helper function to create pass with logging
template<typename PassType>
std::shared_ptr<Pass> CreatePassWithLogging(const std::string &pass_name) {
  auto pass = std::make_shared<PassType>();
  MS_LOG(INFO) << "Created pass '" << pass_name << "' successfully";
  return pass;
}

class MSCustomPassPlugin : public CustomPassPlugin {
 public:
  std::string GetPluginName() const override { return "ms_custom_pass_plugin"; }

  std::vector<std::string> GetAvailablePassNames() const override {
    return {"ReplaceAddNFusionPass", "AddNegFusionPass", "FullGraphPass", "FullGraphPass2"};
  }

  std::shared_ptr<Pass> CreatePass(const std::string &pass_name) const override {
    // Create a static map of pass creators to avoid repeated if-else chains
    static const std::unordered_map<std::string, std::function<std::shared_ptr<Pass>()>> pass_creators = {
      {"ReplaceAddNFusionPass", []() { return CreatePassWithLogging<ReplaceAddNFusionPass>("ReplaceAddNFusionPass"); }},
      {"AddNegFusionPass", []() { return CreatePassWithLogging<AddNegFusionPass>("AddNegFusionPass"); }},
      {"FullGraphPass", []() { return CreatePassWithLogging<FullGraphPass>("FullGraphPass"); }},
      {"FullGraphPass2", []() { return CreatePassWithLogging<FullGraphPass>("FullGraphPass"); }}
    };

    auto it = pass_creators.find(pass_name);
    if (it != pass_creators.end()) {
      return it->second();
    }

    // Log available passes
    std::string available_passes = "ReplaceAddNFusionPass, AddNegFusionPass, FullGraphPass, FullGraphPass2";
    MS_LOG(WARNING) << "Pass '" << pass_name << "' not found, available: " << available_passes;
    return nullptr;
  }
};

}  // namespace opt
}  // namespace mindspore

EXPORT_CUSTOM_PASS_PLUGIN(mindspore::opt::MSCustomPassPlugin)
