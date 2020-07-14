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
#include "transform/graph_ir/df_graph_manager.h"

using UT::Common;

namespace mindspore {
namespace transform {

class TestDfGraphManager : public UT::Common {
 public:
  TestDfGraphManager() {}
};

TEST_F(TestDfGraphManager, TestAPI) {
  // test public interface:
  DfGraphManager& graph_manager = DfGraphManager::GetInstance();
  ASSERT_EQ(0, graph_manager.GetAllGraphs().size());

  // test public interface:
  std::shared_ptr<ge::Graph> ge_graph = std::make_shared<ge::Graph>();
  ASSERT_TRUE(graph_manager.AddGraph("test_graph", nullptr) != Status::SUCCESS);
  graph_manager.AddGraph("test_graph", ge_graph);
  ASSERT_EQ(1, graph_manager.GetAllGraphs().size());
  std::vector<DfGraphWrapperPtr> wrappers = graph_manager.GetAllGraphs();
  ASSERT_EQ("test_graph", wrappers.back()->name_);
  ASSERT_EQ(ge_graph, wrappers.back()->graph_ptr_);

  // test public interface:
  DfGraphWrapperPtr wrappers2 = graph_manager.GetGraphByName("test_graph");
  ASSERT_EQ(ge_graph, wrappers2->graph_ptr_);

  // test public interface:
  graph_manager.ClearGraph();
  ASSERT_EQ(0, graph_manager.GetAllGraphs().size());

  // test public interface:
  int id = graph_manager.GenerateId();
  assert(id > 0);
}

}  // namespace transform
}  // namespace mindspore
