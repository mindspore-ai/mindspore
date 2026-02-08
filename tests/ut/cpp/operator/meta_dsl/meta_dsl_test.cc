/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "tests/ut/cpp/operator/meta_dsl/dense.h"
#include "ir/manager.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore::prim {
class TestMetaDsl : public UT::Common {
 public:
  TestMetaDsl() {}

  void SetUp() { UT::InitPythonPath(); }

  void TearDown() {}

  AbstractBasePtr EvalGraph(const FuncGraphPtr &func_graph, const AbstractBasePtrList &abs_list) {
    if (engine_ == nullptr) {
      std::shared_ptr<FuncGraphManager> graph_manager = MakeManager();
      engine_ = std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), graph_manager);
    }
    return engine_->Run(func_graph, abs_list).eval_result->abstract();
  }

 private:
  abstract::AnalysisEnginePtr engine_{nullptr};
};

FuncGraphPtr CreateFuncGraphWithDense() {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  std::vector<FuncGraphPtr> graphs{fg};
  auto func_graph_manager = std::make_shared<FuncGraphManager>(graphs);
  AnfNodePtr param_input = fg->add_parameter();
  AnfNodePtr param_weight = fg->add_parameter();
  AnfNodePtr param_bias = fg->add_parameter();
  auto dense = std::make_shared<DenseMetaImpl>();
  dense->set_prim(prim::kPrimDense);
  dense->set_manager(func_graph_manager);
  CNodePtr cnode = fg->NewCNode({NewValueNode(dense), param_input, param_weight, param_bias});
  fg->set_output(cnode);
  return fg;
}

/// Feature: Meta DSL
/// Description: Test MetaDSL with Dense.
/// Expectation: Run successfully.
TEST_F(TestMetaDsl, test_dense) {
  auto fg = CreateFuncGraphWithDense();
  auto abs_input = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>({2, 3}));
  auto abs_weight = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>({2, 3}));
  auto abs_bias = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>({2}));
  AbstractBasePtrList abs_list{abs_input, abs_weight, abs_bias};
  AbstractBasePtr res = EvalGraph(fg, abs_list);
  ASSERT_TRUE(res->isa<abstract::AbstractTensor>());
  auto res_shape = res->cast<abstract::AbstractTensorPtr>()->GetShape();
  ASSERT_TRUE(res_shape != nullptr);
  auto shape = res_shape->cast<abstract::TensorShapePtr>()->shape();
  std::vector<int64_t> expected_shape{2, 2};
  ASSERT_TRUE(shape == expected_shape);
}

/// Feature: Meta DSL
/// Description: Test MetaDSL with Dense.
/// Expectation: Throw an exception and catch it.
TEST_F(TestMetaDsl, test_dense_with_invalid_bias) {
  auto fg = CreateFuncGraphWithDense();
  auto abs_input = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>({4}));
  auto abs_weight = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>({9, 4}));
  auto abs_bias = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>({9, 1}));
  AbstractBasePtrList abs_list{abs_input, abs_weight, abs_bias};
  try {
    EvalGraph(fg, abs_list);
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("The dim of b should be equal to 0 or 1 if the dim of w is 2") !=
                std::string::npos);
  }
}
}  // namespace mindspore::prim
