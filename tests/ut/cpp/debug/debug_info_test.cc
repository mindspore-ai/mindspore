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
#include <iostream>
#include <memory>
#include <set>
#include "common/common_test.h"
#include "utils/info.h"
#include "utils/trace_base.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"

namespace mindspore {
class TestDebugInfo : public UT::Common {
 public:
  TestDebugInfo() {}
};

// Feature: Debug info
// Description: Make a location
// Expectation: make a location with no error
TEST_F(TestDebugInfo, test_make_location) {
  LocationPtr loc1 = std::make_shared<Location>("/home/workspace/a.py", 1, 4, 2, 8, "", std::vector<std::string>());
  std::string result_str = loc1->ToString(kSourceLineTipDiscard);

  std::string expect_str("In file /home/workspace/a.py:1~2, 4~8\n");
  if (result_str != expect_str) {
    MS_LOG(ERROR) << "expect: " << expect_str;
    MS_LOG(ERROR) << "result: " << result_str;
  }
  ASSERT_TRUE(result_str == expect_str);
}

// Feature: Debug info
// Description: Deduplicate the debug infos which have the same print
// Expectation: a set of debug infos is deduplicated
TEST_F(TestDebugInfo, test_location_dedup) {
  LocationPtr loc1 = std::make_shared<Location>("file1.py", 0, 0, 0, 0, "", std::vector<std::string>());
  NodeDebugInfoPtr debug_info1 = std::make_shared<NodeDebugInfo>();
  debug_info1->set_location(loc1);

  LocationPtr loc2 = std::make_shared<Location>("file1.py", 0, 0, 0, 0, "", std::vector<std::string>());
  NodeDebugInfoPtr debug_info2 = std::make_shared<NodeDebugInfo>();
  debug_info2->set_location(loc2);

  LocationPtr loc3 = std::make_shared<Location>("file2.py", 0, 0, 0, 0, "", std::vector<std::string>());
  NodeDebugInfoPtr debug_info3 = std::make_shared<NodeDebugInfo>();
  debug_info3->set_location(loc3);

  std::set<NodeDebugInfoPtr, DebugInfoCompare> fused_debug_info_set;
  (void)fused_debug_info_set.emplace(debug_info1);
  ASSERT_TRUE(fused_debug_info_set.size() == 1);

  (void)fused_debug_info_set.emplace(debug_info2);
  ASSERT_TRUE(fused_debug_info_set.size() == 1);

  (void)fused_debug_info_set.emplace(debug_info3);
  ASSERT_TRUE(fused_debug_info_set.size() == 2);

  (void)fused_debug_info_set.emplace(debug_info3);
  ASSERT_TRUE(fused_debug_info_set.size() == 2);
}

// Feature: Debug info
// Description: Test adding a fused debug info
// Expectation: success
TEST_F(TestDebugInfo, test_fused_debug_info) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  std::vector<AnfNodeWeakPtr> inputs;
  CNodePtr cnode = std::make_shared<CNode>(inputs, fg);

  ASSERT_TRUE(cnode->fused_debug_infos().size() == 0);

  NodeDebugInfoPtr debug_info = std::make_shared<NodeDebugInfo>();
  cnode->AddFusedDebugInfo(debug_info);

  ASSERT_TRUE(cnode->fused_debug_infos().size() == 1);
}
}  // namespace mindspore