/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ir/func_graph.h"
#include "gtest/gtest.h"

#ifndef MINDSPORE_UT_COMMON_RESOURCE_H
#define MINDSPORE_UT_COMMON_RESOURCE_H
namespace UT {
using UTKeyInfo = std::pair<std::string, std::string>;

class UTResourceManager {
 public:
  UTResourceManager() = default;
  ~UTResourceManager() {
    for (const auto &it : all_func_graphs_) {
      auto key_info = it.first;
      std::cout << "Unexpected unreleased func graph resource of case:" << key_info.first << "." << key_info.second
                << std::endl;
    }
    if (!all_func_graphs_.empty()) {
      std::cout << "Please check `TearDown` function of testcase, and make sure all func graphs can be dropped after "
                   "case executed, otherwise core dumped might occur."
                << std::endl;
    }
  }

  void HoldFuncGraph(const mindspore::FuncGraphPtr &fg) {
    const char *suite_name = testing::UnitTest::GetInstance()->current_test_suite()->name();
    const char *test_name = testing::UnitTest::GetInstance()->current_test_info()->name();
    auto new_fg = std::make_shared<mindspore::FuncGraph>();
    std::cout << "Hold func graph of case:" << suite_name << "." << test_name << std::endl;
    (void)all_func_graphs_[UTKeyInfo{suite_name, test_name}].insert(fg);
  }

  mindspore::FuncGraphPtr MakeAndHoldFuncGraph() {
    auto func_graph = std::make_shared<mindspore::FuncGraph>();
    HoldFuncGraph(func_graph);
    return func_graph;
  }

  void DropFuncGraph(const UTKeyInfo &ut_info) {
    if (all_func_graphs_.find(ut_info) == all_func_graphs_.cend()) {
      return;
    }
    std::cout << "Drop func graph of case:" << ut_info.first << "." << ut_info.second << std::endl;
    (void)all_func_graphs_.erase(ut_info);
  }

  void DropAllFuncGraphs() { all_func_graphs_.clear(); }

  static std::shared_ptr<UTResourceManager> GetInstance();

 private:
  static std::shared_ptr<UTResourceManager> inst_resource_manager_;
  std::map<UTKeyInfo, std::set<mindspore::FuncGraphPtr>> all_func_graphs_;
};

}  // namespace UT

#endif  // MINDSPORE_UT_COMMON_RESOURCE_H
