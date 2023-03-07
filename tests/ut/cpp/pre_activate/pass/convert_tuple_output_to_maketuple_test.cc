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
#include "common/backend_common_test.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pass_manager.h"
#include "backend/common/pass/convert_tuple_output_to_maketuple.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
class TestHWTupleOutputToMakeTuple : public BackendCommon {
 public:
  TestHWTupleOutputToMakeTuple()
      : getPyFun_("gtest_input.pre_activate.convert_tuple_output_to_maketuple_test", true) {}
  ~TestHWTupleOutputToMakeTuple() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestHWTupleOutputToMakeTuple, test_convert_tuple_output_to_maketuple) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_convert_tuple_output_to_maketuple", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{5, 2, 10};
  std::vector<int64_t> shp_h{1, 2, 2};
  std::vector<int64_t> shp_c{1, 2, 2};
  std::vector<int64_t> shp_w{112, 1, 1};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto h_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_h);
  auto c_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_c);
  auto w_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_w);
  AbstractBasePtrList args_spec_list{x_abstract, h_abstract, c_abstract, w_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConvertTupleOutputToMaketuple>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_convert_tuple_output_to_maketuple", "after");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}
}  // namespace opt
}  // namespace mindspore
