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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_ONLINE_FUSION_ONLINE_FUSION_PASS_REGISTRY_H_
#define MINDSPORE_LITE_SRC_RUNTIME_ONLINE_FUSION_ONLINE_FUSION_PASS_REGISTRY_H_

#include <string>
#include <utility>
#include <vector>
#include "src/litert/sub_graph_split.h"

typedef std::string OnlineFusionPassName;
typedef int (*OnlineFusionPassFunc)(mindspore::lite::SearchSubGraph *);
typedef std::pair<std::string, OnlineFusionPassFunc> OnlineFusionPassItem;

namespace mindspore::lite {
class OnlineFusionRegistry {
 public:
  OnlineFusionRegistry() = default;
  virtual ~OnlineFusionRegistry() = default;

  static OnlineFusionRegistry *GetInstance();
  void RegOnlineFusionPass(const OnlineFusionPassName desc, const OnlineFusionPassFunc creator);
  void DoOnlineFusionPass(mindspore::lite::SearchSubGraph *searchSubGraph);

 private:
  std::vector<OnlineFusionPassItem> online_fusion_pass_arrays_;
};

class OnlineFusionPassRegistrar {
 public:
  OnlineFusionPassRegistrar(const OnlineFusionPassName &name, const OnlineFusionPassFunc &online_fusion_pass) {
    OnlineFusionRegistry::GetInstance()->RegOnlineFusionPass(name, online_fusion_pass);
  }
  ~OnlineFusionPassRegistrar() = default;
};

#define REG_ONLINE_FUSION_PASS(online_fusion_pass) \
  static OnlineFusionPassRegistrar g_##online_fusion_pass##Reg(#online_fusion_pass, online_fusion_pass);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_REGISTRY_H_
