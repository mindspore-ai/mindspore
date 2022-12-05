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
#include <utility>
#include "src/litert/pass/online_fusion/online_fusion_pass_registry.h"

namespace mindspore::lite {
OnlineFusionRegistry *OnlineFusionRegistry::GetInstance() {
  static OnlineFusionRegistry instance;
  return &instance;
}

void OnlineFusionRegistry::RegOnlineFusionPass(const OnlineFusionPassName name,
                                               const OnlineFusionPassFunc online_fusion_pass) {
  online_fusion_pass_arrays_.emplace_back(std::make_pair(name, online_fusion_pass));
}

void OnlineFusionRegistry::DoOnlineFusionPass(mindspore::lite::SearchSubGraph *searchSubGraph) {
  if (searchSubGraph == nullptr) {
    return;
  }
  for (auto &online_fusion_pass_item : online_fusion_pass_arrays_) {
    MS_LOG(INFO) << "do online fusion pass : " << online_fusion_pass_item.first;
    online_fusion_pass_item.second(searchSubGraph);
  }
}
}  // namespace mindspore::lite
