/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"

#ifdef OPEN_SOURCE
#include "ge/client/ge_api.h"
#else
#include "external/ge/ge_api.h"
#endif

#define private public
#include "transform/graph_ir/graph_builder.h"
#include "transform/graph_ir/df_graph_manager.h"

using UT::Common;

namespace mindspore {
namespace transform {

class TestDfGraphBuilder : public UT::Common {
 public:
  TestDfGraphBuilder() {}
  void SetUp();
  void TearDown();
};

void TestDfGraphBuilder::SetUp() {}

void TestDfGraphBuilder::TearDown() {}

TEST_F(TestDfGraphBuilder, TestBuildDatasetGraph) {
  DatasetGraphParam param4("queue_name", 1, 32, {0, 3}, {{32, 224, 224, 3}, {32}}, {});
  ASSERT_EQ(transform::SUCCESS, BuildDatasetGraph(param4));
  DfGraphManager::GetInstance().ClearGraph();
}

}  // namespace transform
}  // namespace mindspore
