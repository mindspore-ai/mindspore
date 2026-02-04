/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "mindspore/core/include/utils/log_adapter.h"
#include "mindspore/ccsrc/include/backend/common/custom_pass/custom_pass_plugin.h"
#include "replace_addn_fusion/replace_addn_fusion_pass.h"
#include "add_neg_fusion/add_neg_fusion_pass.h"

namespace mindspore {
namespace opt {

class MSCustomPassPlugin : public CustomPassPlugin {
 public:
  std::string GetPluginName() const override { return "ms_custom_pass_plugin"; }

  std::vector<std::string> GetAvailablePassNames() const override {
    return {"ReplaceAddNFusionPass", "AddNegFusionPass"};
  }

  std::shared_ptr<Pass> CreatePass(const std::string &pass_name) const override {
    if (pass_name == "ReplaceAddNFusionPass") {
      auto pass = std::make_shared<ReplaceAddNFusionPass>();
      MS_LOG(INFO) << "Created pass '" << pass_name << "' successfully";
      return pass;
    } else if (pass_name == "AddNegFusionPass") {
      auto pass = std::make_shared<AddNegFusionPass>();
      MS_LOG(INFO) << "Created pass '" << pass_name << "' successfully";
      return pass;
    } else {
      MS_LOG(WARNING) << "Pass '" << pass_name << "' not found, available: ReplaceAddNFusionPass, AddNegFusionPass";
      return nullptr;
    }
  }
};

}  // namespace opt
}  // namespace mindspore

EXPORT_CUSTOM_PASS_PLUGIN(mindspore::opt::MSCustomPassPlugin)
