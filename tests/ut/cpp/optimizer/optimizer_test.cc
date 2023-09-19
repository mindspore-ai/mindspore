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
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/cse_pass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "frontend/optimizer/py_interpret_to_execute.h"
#include "include/common/debug/draw.h"

namespace mindspore {
namespace opt {
using Var = mindspore::Var;

class TestOptOptimizer : public UT::Common {
 public:
  TestOptOptimizer() : getPyFun("gtest_input.optimizer.opt_test", true), irpass() {}
  UT::PyFuncGraphFetcher getPyFun;
  irpass::OptimizeIRPassLib irpass;
};

class TestPyInterpretToPyExecute : public BackendCommon {
 public:
  TestPyInterpretToPyExecute() : getPyFun("gtest_input.optimizer.pyinterpret_dict_convert_test", true) {}
  ~TestPyInterpretToPyExecute() override = default;
  UT::PyFuncGraphFetcher getPyFun;

  void ChangeStringToScript(const pipeline::ResourcePtr &resource) {
    auto trans = resource->manager();
    MS_EXCEPTION_IF_NULL(trans);
    auto nodes = trans->all_nodes();
    for (const auto &node : nodes) {
      if (IsPrimitiveCNode(node, prim::kPrimPyInterpret)) {
        auto constexpr kScriptInputIdx = 1;
        auto cnode = node->cast<CNodePtr>();
        auto script_str_node = cnode->input(kScriptInputIdx);
        auto script_string = GetValueNode<StringImmPtr>(script_str_node);
        auto script = script_string->value();
        auto script_node = NewValueNode(std::make_shared<parse::Script>(script));
        cnode->set_input(kScriptInputIdx, script_node);
      }

      if (node->isa<ValueNode>()) {
        auto value_node = node->cast_ptr<ValueNode>();
        auto value = value_node->value();
        value_node->set_abstract(value->ToAbstract());
      }
    }
  }
};

TEST_F(TestOptOptimizer, test_step_opt) {
  FuncGraphPtr before = getPyFun("test_expandJ");

  ASSERT_TRUE(nullptr != before);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  std::shared_ptr<Optimizer> optimizer =
    Optimizer::MakeOptimizer("ut_test", res,
                             {{"main",
                               {
                                 // Branch culling
                                 irpass.switch_simplify_,

                                 // Safe inlining
                                 irpass.arithmetic_simplify_,
                                 irpass.inline_,
                               }},
                              {"grad", opt::OptPassConfig(opt::irpass::ExpandJPrim())},
                              {"cse", OptPassConfig(CSEPass(false))}},
                             true);
  EXPECT_TRUE(optimizer.get() != nullptr);

  auto after = optimizer->step(before);
}
}  // namespace opt
}  // namespace mindspore
