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
#include <string>
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/parse/parse.h"
#include "include/common/debug/draw.h"

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/irpass.h"
#include "pipeline/jit/action.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace parse {
class TestParallelIf : public UT::Common {
 public:
  TestParallelIf() : getPyFun("gtest_input.pipeline.parse.parallel_if") {}
  virtual void SetUp();
  virtual void TearDown();
  py::function GetPythonFunction(std::string function);

  bool CheckIsomorphic(FuncGraphPtr basic, FuncGraphPtr manual, std::vector<opt::SubstitutionPtr> opts = {}) {
    opt::SubstitutionList transform(opts);
    FuncGraphPairMapEquiv equiv_graph;
    NodeMapEquiv equiv_node;

    opt::OptimizerPtr optimizer = std::make_shared<opt::Optimizer>("ut_test", std::make_shared<pipeline::Resource>());
    FuncGraphPtr basic_clone = BasicClone(basic);
    transform(basic_clone, optimizer);
    FuncGraphPtr manual_clone = BasicClone(manual);
    transform(manual_clone, optimizer);

    return Isomorphic(basic_clone, manual_clone, &equiv_graph, &equiv_node);
  }

  void CheckParallelIfTransform(const std::string &test_case) {
    FuncGraphPtr basic_graph = getPyFun.CallAndParseRet(test_case, "basic");
    ASSERT_TRUE(basic_graph != nullptr);
    FuncGraphPtr manual_graph = getPyFun.CallAndParseRet(test_case, "manual");
    ASSERT_TRUE(manual_graph != nullptr);

    pipeline::ResourcePtr res1 = std::make_shared<pipeline::Resource>();

    tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), std::vector<int64_t>{1});
    tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), std::vector<int64_t>{1});

    AbstractBasePtr abstract_x = abstract::FromValue(x_tensor, true);
    AbstractBasePtr abstract_y = abstract::FromValue(y_tensor, true);
    abstract::AbstractBasePtrList args_spec_list{abstract_x, abstract_y};

    abstract::AnalysisResult result = pipeline::AbstractAnalyze(res1, basic_graph, args_spec_list);
    auto new_basic_graph = pipeline::ProgramSpecialize(res1, basic_graph, result.context);

    pipeline::ResourcePtr res2 = std::make_shared<pipeline::Resource>();
    result = pipeline::AbstractAnalyze(res2, manual_graph, args_spec_list);
    auto new_manual_graph = pipeline::ProgramSpecialize(res2, manual_graph, result.context);

    auto patterns = std::vector<opt::SubstitutionPtr>({irpass_lib_.inline_, irpass_lib_.switch_simplify_});
    ASSERT_TRUE(CheckIsomorphic(new_basic_graph, new_manual_graph, patterns));

    abstract::AnalysisResultCacheMgr::GetInstance().Clear();
    abstract::AnalysisContext::ClearContext();
  }
 public:
  UT::PyFuncGraphFetcher getPyFun;
  opt::irpass::OptimizeIRPassLib irpass_lib_;
};

void TestParallelIf::SetUp() { UT::InitPythonPath(); }

void TestParallelIf::TearDown() {}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for test code with single if/else.
// Expectation: The funcgraph after transformation should be isomorphic with the funcgraph manually constructed.
TEST_F(TestParallelIf, SimpleIf) { CheckParallelIfTransform("test_simple_if"); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for test code with if-by-if.
// Expectation: The funcgraph after transformation should be isomorphic with the funcgraph manually constructed.
TEST_F(TestParallelIf, IfByIf) { CheckParallelIfTransform("test_if_by_if"); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for test code with if-in-if.
// Expectation: The funcgraph after transformation should be isomorphic with the funcgraph manually constructed.
TEST_F(TestParallelIf, IfInIf) { CheckParallelIfTransform("test_if_in_if"); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for test code with if-elif-else.
// Expectation: The funcgraph after transformation should be isomorphic with the funcgraph manually constructed.
TEST_F(TestParallelIf, IfElifElse) { CheckParallelIfTransform("test_if_elif_else"); }
}  // namespace parse
}  // namespace mindspore
