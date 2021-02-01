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
#include <unordered_map>

#include "frontend/optimizer/ad/grad.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/manager.h"
#include "ir/value.h"
#include "ir/func_graph_cloner.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/parse/parse.h"
#include "debug/draw.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace ad {
class TestAD : public UT::Common {
 public:
  TestAD() : getPyFun("gtest_input.optimizer.ad", true) {}

 public:
  UT::PyFuncGraphFetcher getPyFun;
  pipeline::ResourceBasePtr resourcePtr = std::make_shared<pipeline::ResourceBase>();

 protected:
  void AssertExpect(const std::string& testCase) {
    FuncGraphPtr g = getPyFun(testCase);
    FuncGraphPtr dg = Grad(g, resourcePtr);
    AssertExpect(testCase, dg);
  }

  void AssertExpect(const std::string& testCase, const FuncGraphPtr& dg) { ASSERT_TRUE(dg != nullptr); }
};

TEST_F(TestAD, test_null) { AssertExpect("test_null"); }

TEST_F(TestAD, test_grad_add) { AssertExpect("test_grad_add"); }

TEST_F(TestAD, test_grad_expr) { AssertExpect("test_grad_expr"); }

TEST_F(TestAD, test_constant) { AssertExpect("test_constant"); }

TEST_F(TestAD, test_dup_args_in_call) { AssertExpect("test_dup_args_in_call"); }

TEST_F(TestAD, test_quadruple_args_in_call) { AssertExpect("test_quadruple_args_in_call"); }

TEST_F(TestAD, test_tuples) { AssertExpect("test_tuples"); }

TEST_F(TestAD, test_hof) { AssertExpect("test_hof"); }

TEST_F(TestAD, test_more_hof) { AssertExpect("test_more_hof"); }

TEST_F(TestAD, test_simple_closure) { AssertExpect("test_simple_closure"); }

TEST_F(TestAD, test_closure) { AssertExpect("test_closure"); }

TEST_F(TestAD, test_if) { AssertExpect("test_if"); }

TEST_F(TestAD, test_if2) { AssertExpect("test_if2"); }

TEST_F(TestAD, test_fact) { AssertExpect("test_fact"); }

TEST_F(TestAD, test_while) { AssertExpect("test_while"); }

TEST_F(TestAD, test_while_2) { AssertExpect("test_while_2"); }

TEST_F(TestAD, test_pow10) { AssertExpect("test_pow10"); }

TEST_F(TestAD, test_closures_in_tuples) { AssertExpect("test_closures_in_tuples"); }

TEST_F(TestAD, test_ops_fn) { AssertExpect("test_ops_fn"); }

TEST_F(TestAD, test_more_closure) { AssertExpect("test_more_closure"); }

TEST_F(TestAD, test_prim_scalar_add) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarAdd), resourcePtr);
  AssertExpect("test_prim_scalar_add", dg);
}

TEST_F(TestAD, test_prim_scalar_mul) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarMul), resourcePtr);
  AssertExpect("test_prim_scalar_mul", dg);
}

TEST_F(TestAD, test_prim_scalar_sub) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarSub), resourcePtr);
  AssertExpect("test_prim_scalar_sub", dg);
}

TEST_F(TestAD, test_prim_scalar_div) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarDiv), resourcePtr);
  AssertExpect("test_prim_scalar_div", dg);
}

TEST_F(TestAD, test_prim_scalar_pow) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarPow), resourcePtr);
  AssertExpect("test_prim_scalar_pow", dg);
}

TEST_F(TestAD, test_prim_scalar_exp) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarExp), resourcePtr);
  AssertExpect("test_prim_scalar_exp", dg);
}

TEST_F(TestAD, test_prim_scalar_uadd) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarUadd), resourcePtr);
  AssertExpect("test_prim_scalar_uadd", dg);
}

TEST_F(TestAD, test_prim_scalar_usub) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarUsub), resourcePtr);
  AssertExpect("test_prim_scalar_usub", dg);
}

TEST_F(TestAD, test_prim_scalar_gt) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarGt), resourcePtr);
  AssertExpect("test_prim_scalar_gt", dg);
}

TEST_F(TestAD, test_prim_scalar_lt) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarLt), resourcePtr);
  AssertExpect("test_prim_scalar_lt", dg);
}

TEST_F(TestAD, test_prim_scalar_ge) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarGe), resourcePtr);
  AssertExpect("test_prim_scalar_ge", dg);
}

TEST_F(TestAD, test_prim_scalar_le) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarLe), resourcePtr);
  AssertExpect("test_prim_scalar_le", dg);
}

TEST_F(TestAD, test_prim_tuple_getitem) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimTupleGetItem), resourcePtr);
  AssertExpect("test_prim_tuple_getitem", dg);
}

TEST_F(TestAD, test_prim_identity) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimIdentity), resourcePtr);
  AssertExpect("test_prim_identity", dg);
}

TEST_F(TestAD, test_prim_scalar_to_array) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarToArray), resourcePtr);
  AssertExpect("test_prim_scalar_to_array", dg);
}

TEST_F(TestAD, test_prim_array_to_scalar) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimArrayToScalar), resourcePtr);
  AssertExpect("test_prim_array_to_scalar", dg);
}

TEST_F(TestAD, test_prim_distribute) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimDistribute), resourcePtr);
  AssertExpect("test_prim_distribute", dg);
}

TEST_F(TestAD, test_prim_broadcast_shape) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimBroadcastShape), resourcePtr);
  AssertExpect("test_prim_broadcast_shape", dg);
}

TEST_F(TestAD, test_prim_J) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimJ), resourcePtr);
  AssertExpect("test_prim_J", dg);
}

TEST_F(TestAD, test_prim_switch) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimSwitch), resourcePtr);
  AssertExpect("test_prim_switch", dg);
}

TEST_F(TestAD, test_grad_cache) {
  FuncGraphPtr g = getPyFun("test_null");
  FuncGraphPtr dg1 = Grad(g, resourcePtr);
  FuncGraphPtr dg2 = Grad(g, resourcePtr);
  ASSERT_TRUE(dg1 == dg2);
}

TEST_F(TestAD, test_constant_output) { AssertExpect("test_constant_output"); }

}  // namespace ad
}  // namespace mindspore
