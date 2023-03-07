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
#include "common/backend_common_test.h"
#include "frontend/operator/ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pass_manager.h"
#include "backend/common/pass/convert_const_input_to_attr.h"
#include "include/common/utils/utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt {
class TestAllToAllUnifyMindIr : public BackendCommon {
 public:
  TestAllToAllUnifyMindIr() : getPyFun_("gtest_input.pre_activate.all_to_all_unify_mindir_test", true) {}
  ~TestAllToAllUnifyMindIr() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestAllToAllUnifyMindIr, test_neighbor_exchange) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_neighbor_exchange", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{2, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTuple>(
    AbstractBasePtrList{std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x)});
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  bool has_all_to_all_v_node = false;
  for (const auto &n : TopoSort(func_graph->get_return())) {
    ASSERT_FALSE(IsPrimitiveCNode(n, prim::kPrimNeighborExchange));
    if (IsPrimitiveCNode(n, prim::kPrimAllToAllv)) {
      has_all_to_all_v_node = true;
    }
  }
  ASSERT_TRUE(has_all_to_all_v_node);
}

TEST_F(TestAllToAllUnifyMindIr, test_all_to_all) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_all_to_all", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{4, 2, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  bool has_all_to_all_v_node = false;
  bool has_concat_node = false;
  bool has_split_v_node = false;
  for (const auto &n : TopoSort(func_graph->get_return())) {
    ASSERT_FALSE(IsPrimitiveCNode(n, prim::kPrimAllToAll));
    if (IsPrimitiveCNode(n, prim::kPrimAllToAllv)) {
      has_all_to_all_v_node = true;
    }
    if (IsPrimitiveCNode(n, prim::kPrimConcatD)) {
      has_concat_node = true;
    }
    if (IsPrimitiveCNode(n, prim::kPrimSplitVD)) {
      has_split_v_node = true;
    }
  }
  ASSERT_TRUE(has_all_to_all_v_node);
  ASSERT_TRUE(has_concat_node);
  ASSERT_TRUE(has_split_v_node);
}
}  // namespace opt
}  // namespace mindspore
