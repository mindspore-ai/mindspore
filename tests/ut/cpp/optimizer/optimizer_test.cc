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
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/cse_pass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "debug/draw.h"

namespace mindspore {
namespace opt {
using Var = mindspore::Var;

class TestOptOptimizer : public UT::Common {
 public:
  TestOptOptimizer() : getPyFun("gtest_input.optimizer.opt_test", true), irpass() {}
  UT::PyFuncGraphFetcher getPyFun;
  irpass::OptimizeIRPassLib irpass;
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

  draw::Draw("optimizer_test_expendJ_before.dot", before);
  draw::Draw("optimizer_test_expendJ_after.dot", after);
}

}  // namespace opt
}  // namespace mindspore
