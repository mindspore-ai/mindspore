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
#include <iostream>
#include "common/common_test.h"
#include "common/resource.h"
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "ir/value.h"
#include "ops/math_ops.h"
#include "ops/framework_ops.h"
#include "ops/nn_ops.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"

namespace mindspore {
class TestCompilerFailureMode : public UT::Common {
 public:
  TestCompilerFailureMode() {}

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

/// Feature: Failure mode for compiler.
/// Description: Primitive Add requires 2 inputs, but fg creates 3 inputs for cnode.
/// Expectation: Throw an exception and catch it.
TEST_F(TestCompilerFailureMode, test_create_abnormal_node) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  AnfNodePtr param0 = fg->add_parameter();
  AnfNodePtr param1 = fg->add_parameter();
  AnfNodePtr param2 = fg->add_parameter();
  CNodePtr cnode = fg->NewCNode({NewValueNode(prim::kPrimAdd), param0, param1, param2});
  fg->set_output(cnode);

  try {
    auto abs = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>({1}));
    AbstractBasePtrList abs_list{abs, abs, abs};
    AbstractBasePtr res = EvalGraph(fg, abs_list);
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("the inputs number should be 2") != std::string::npos);
  }
}

/// Feature: Failure mode for compiler.
/// Description: Incorrect use of depend causes the graph cycle exists.
/// Expectation: Throw an exception and catch it.
TEST_F(TestCompilerFailureMode, test_graph_cycle) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  AnfNodePtr param0 = fg->add_parameter();
  CNodePtr call_cnode = fg->NewCNode({{param0}});
  CNodePtr depend_cnode = fg->NewCNode({NewValueNode(prim::kPrimDepend), NewValueNode(MakeValue(1)), call_cnode});
  fg->set_output(depend_cnode);

  FuncGraphPtr fg1 = std::make_shared<FuncGraph>();
  CNodePtr cnode = fg1->NewCNode({{NewValueNode(fg), NewValueNode(fg1)}});
  fg1->set_output(cnode);

  try {
    AbstractBasePtr res = EvalGraph(fg1, {});
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("Exceed function call depth limit 1000") != std::string::npos);
  }
}

/// Feature: Failure mode for compiler.
/// Description: Auto monad generates abnormal graph.
/// Expectation: Throw an exception and catch it.
TEST_F(TestCompilerFailureMode, test_side_effect_abnormal_graph) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  AnfNodePtr param0 = fg->add_parameter();
  AnfNodePtr param1 = fg->add_parameter();
  CNodePtr cnode = fg->NewCNode({NewValueNode(prim::kPrimAssign), param0, NewValueNode(kUMonad), param1});
  fg->set_output(cnode);

  try {
    auto x_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>({1}));
    auto param_abs = std::make_shared<abstract::AbstractRefTensor>(x_abstract, std::make_shared<RefKey>("parameter"));
    AbstractBasePtr tensor_abs = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>({1}));
    AbstractBasePtr res = EvalGraph(fg, {param_abs, tensor_abs});
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("doesn't implement") != std::string::npos);
  }
}

/// Feature: Failure mode for compiler.
/// Description: Auto monad: Number of parameters of graphs does not match
/// Expectation: Throw an exception and catch it.
TEST_F(TestCompilerFailureMode, test_side_effect_incorrect_inputs_number) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  AnfNodePtr param0 = fg->add_parameter();
  AnfNodePtr param1 = fg->add_parameter();
  CNodePtr cnode = fg->NewCNode({NewValueNode(prim::kPrimAssign), param0, param1});
  fg->set_output(cnode);

  try {
    auto x_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>({1}));
    auto param_abs = std::make_shared<abstract::AbstractRefTensor>(x_abstract, std::make_shared<RefKey>("parameter"));
    AbstractBasePtr tensor_abs = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>({1}));
    AbstractBasePtr monad_abs = abstract::FromValue(kUMonad, false);
    AbstractBasePtr res = EvalGraph(fg, {param_abs, tensor_abs, monad_abs});
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what())
                  .find("The parameters number of the function is 2, but the number of provided arguments is 3") !=
                std::string::npos);
  }
}

/// Feature: Failure mode for compiler.
/// Description: IR pass error while replacing node.
/// Expectation: Throw an exception and catch it.
TEST_F(TestCompilerFailureMode, test_irpass_abnormal) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  AnfNodePtr param0 = fg->add_parameter();
  CNodePtr cnode = fg->NewCNode({NewValueNode(prim::kPrimMul), param0, NewValueNode(MakeValue(1.01))});
  fg->set_output(cnode);

  auto mgr = Manage(fg);
  mgr->Replace(cnode, param0);
  // Incorrect use "x * 1 = x"
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(10, kFloat32);
  AbstractBasePtr abs = abstract::FromValue(tensor, false);
  AbstractBasePtr res = EvalGraph(fg, {abs});
  auto res_data = reinterpret_cast<float *>(GetValue<tensor::TensorPtr>(res->BuildValue())->data_c());
  ASSERT_TRUE(*res_data == 10);
}

/// Feature: Failure mode for compiler.
/// Description: Abstract is nullptr.
/// Expectation: Throw an exception and catch it.
TEST_F(TestCompilerFailureMode, test_abstract_nullptr) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  AnfNodePtr param0 = fg->add_parameter();
  CNodePtr cnode = fg->NewCNode({NewValueNode(prim::kPrimAdd), param0, NewValueNode(MakeValue(1.01))});
  fg->set_output(cnode);

  try {
    AbstractBasePtr res = EvalGraph(fg, {nullptr});
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("The pointer[abs] is null") != std::string::npos);
  }
}

/// Feature: Failure mode for compiler.
/// Description: Create abnormal node in bprop.
/// Expectation: Throw an exception and catch it.
TEST_F(TestCompilerFailureMode, test_bprop_normal) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  AnfNodePtr param0 = fg->add_parameter();
  CNodePtr cnode = fg->NewCNode({NewValueNode(prim::kPrimBNInferGrad), param0});
  fg->set_output(cnode);

  try {
    auto abs = std::make_shared<abstract::AbstractTensor>(kInt32, std::vector<int64_t>({1}));
    AbstractBasePtr res = EvalGraph(fg, {abs});
  } catch (std::runtime_error const &err) {
    ASSERT_TRUE(std::string(err.what()).find("the input number must be greater than or equal to 3") !=
                std::string::npos);
  }
}
}  // namespace mindspore
