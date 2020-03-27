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
#include <sstream>
#include <memory>
#include <algorithm>

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "utils/graph_utils.h"

#include "pipeline/parse/parse_base.h"
#include "pipeline/parse/parse.h"

namespace mindspore {

class TestGraphUtils : public UT::Common {
 public:
  TestGraphUtils() : getPyFun("gtest_input.utils.graph_utils_test", true), equiv_graph(), equiv_node() {}
  std::shared_ptr<FuncGraph> GetPythonFunction(std::string function);

 public:
  UT::PyFuncGraphFetcher getPyFun;

  FuncGraphPairMapEquiv equiv_graph;
  NodeMapEquiv equiv_node;
};

TEST_F(TestGraphUtils, Isomorphic) {
  std::shared_ptr<FuncGraph> g1 = getPyFun("test_graph_utils_isomorphic_1");
  std::shared_ptr<FuncGraph> g2 = getPyFun("test_graph_utils_isomorphic_2");
  std::shared_ptr<FuncGraph> g3 = getPyFun("test_graph_utils_isomorphic_3");
  ASSERT_TRUE(nullptr != g1);
  ASSERT_TRUE(nullptr != g2);
  ASSERT_TRUE(Isomorphic(g1, g2, &equiv_graph, &equiv_node));

  ASSERT_TRUE(nullptr != g3);
  ASSERT_FALSE(Isomorphic(g1, g3, &equiv_graph, &equiv_node));
}

}  // namespace mindspore
