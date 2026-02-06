/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include "primitive/sequence_ops.h"
#include "primitive/comparison_ops.h"
#include "primitive/array_ops.h"
#include "primitive/arithmetic_ops.h"
#include "primitive/framework_ops.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/manager.h"
#include "ir/value.h"
#include "ir/func_graph_cloner.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "frontend/jit/ps/action.h"
#include "ir/tensor_new.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/parse/parse.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"
#include "utils/convert_utils.h"
#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "utils/ms_context.h"
#include "include/frontend/jit/ps/action_interface.h"
#include "include/frontend/optimizer/ad/grad_interface.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace ad {
class TestAD : public UT::Common {
 public:
  TestAD() : getPyFun("gtest_input.optimizer.ad", true) {}

 public:
  UT::PyFuncGraphFetcher getPyFun;
  pipeline::ResourcePtr resourcePtr = std::make_shared<pipeline::Resource>();

 protected:
  void AssertExpect(const std::string &testCase) {
    auto ms_context = MsContext::GetInstance();
    ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
    FuncGraphPtr g = getPyFun(testCase);
    resourcePtr->manager()->RemoveRoots();
    resourcePtr->manager()->AddFuncGraph(g, true);
    FuncGraphPtr dg = Grad(g, opt::Optimizer::MakeEmptyOptimizer(resourcePtr));
    AssertExpect(testCase, dg);
  }

  void AssertExpect(const std::string &testCase, const FuncGraphPtr &dg) { ASSERT_TRUE(dg != nullptr); }

  bool CheckEqual(const FuncGraphPtr &before, const FuncGraphPtr &after) {
    FuncGraphPairMapEquiv equiv_graph;
    NodeMapEquiv equiv_node;
    equiv_node.clear();
    equiv_graph.clear();
    return Isomorphic(before, after, &equiv_graph, &equiv_node);
  }

  void DoInline(const FuncGraphPtr &fg) {
    opt::irpass::OptimizeIRPassLib irpass;
    auto patterns = std::vector<opt::SubstitutionPtr>({irpass.inline_, irpass.tuple_list_get_item_eliminator_});
    opt::SubstitutionList transform(patterns);
    opt::OptimizerPtr optimizer = std::make_shared<opt::Optimizer>("ut_test", std::make_shared<pipeline::Resource>());
    transform(fg, optimizer);
  }
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

/// Feature: What feature you test
/// Description: What input in what scene
/// Expectation: success or throw xxx exception or result == xxx, etc.
TEST_F(TestAD, DISABLED_test_prim_scalar_add) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarAdd), resourcePtr);
  AssertExpect("test_prim_scalar_add", dg);
}

TEST_F(TestAD, DISABLED_test_prim_scalar_mul) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarMul), resourcePtr);
  AssertExpect("test_prim_scalar_mul", dg);
}

TEST_F(TestAD, DISABLED_test_prim_scalar_sub) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarSub), resourcePtr);
  AssertExpect("test_prim_scalar_sub", dg);
}

TEST_F(TestAD, DISABLED_test_prim_scalar_div) {
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

TEST_F(TestAD, DISABLED_test_prim_scalar_gt) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarGt), resourcePtr);
  AssertExpect("test_prim_scalar_gt", dg);
}

TEST_F(TestAD, DISABLED_test_prim_scalar_lt) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarLt), resourcePtr);
  AssertExpect("test_prim_scalar_lt", dg);
}

TEST_F(TestAD, DISABLED_test_prim_scalar_ge) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarGe), resourcePtr);
  AssertExpect("test_prim_scalar_ge", dg);
}

TEST_F(TestAD, DISABLED_test_prim_scalar_le) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimScalarLe), resourcePtr);
  AssertExpect("test_prim_scalar_le", dg);
}

TEST_F(TestAD, test_prim_tuple_getitem) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimTupleGetItem), resourcePtr);
  AssertExpect("test_prim_tuple_getitem", dg);
}

TEST_F(TestAD, DISABLED_test_prim_identity) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimIdentity), resourcePtr);
  AssertExpect("test_prim_identity", dg);
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

TEST_F(TestAD, test_prim_switch) {
  FuncGraphPtr dg = Kprim(NewValueNode(prim::kPrimSwitch), resourcePtr);
  AssertExpect("test_prim_switch", dg);
}

TEST_F(TestAD, test_grad_cache) {
  FuncGraphPtr g = getPyFun("test_null");
  FuncGraphPtr dg1 = Grad(g, opt::Optimizer::MakeEmptyOptimizer(resourcePtr));
  FuncGraphPtr dg2 = Grad(g, opt::Optimizer::MakeEmptyOptimizer(resourcePtr));
  ASSERT_TRUE(dg1 == dg2);
}

TEST_F(TestAD, test_constant_output) { AssertExpect("test_constant_output"); }

// Feature: Support automatic differentiation for complex number.
// Description: Test the imag bprop with complex inputs and complex outputs.
// Expectation: The final func_graph construct is correct.
TEST_F(TestAD, TestImagBpropComplexInputComplexOutput) {
  // Parse the forward fg and do renormalize.
  auto g = getPyFun.CallAndParseRet("get_test_ad_fn", "imag_forward");
  AbstractBasePtrList args_spec_list;
  tensor::TensorPtr x_tensor = tensor::from_spec(kNumberTypeComplex64, std::vector<int64_t>{2, 3, 4, 5}, device::DeviceType::kCPU);
  AbstractBasePtr abstract_v1 = abstract::FromValue(x_tensor, true);
  args_spec_list.push_back(abstract_v1);
  FuncGraphPtr new_g = pipeline::Renormalize(resourcePtr, g, args_spec_list);
  // Get the fprop.
  FuncGraphPtr dg = Grad(new_g, opt::Optimizer::MakeEmptyOptimizer(resourcePtr));
  // Make the top cell.
  auto top_fg = std::make_shared<FuncGraph>();
  auto input = top_fg->add_parameter();
  auto dout = top_fg->add_parameter();
  auto fprop_caller = top_fg->NewCNode({NewValueNode(dg), input});
  auto bprop = top_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fprop_caller, NewValueNode<int64_t>(1)});
  auto grads = top_fg->NewCNode({bprop, dout});
  top_fg->set_output(top_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), grads, NewValueNode<int64_t>(1)}));
  // Do renormalize for the top cell.
  tensor::TensorPtr y_tensor = tensor::from_spec(kNumberTypeFloat32, std::vector<int64_t>{2, 3, 4, 5}, device::DeviceType::kCPU);
  AbstractBasePtr abstract_v2 = abstract::FromValue(y_tensor, true);
  args_spec_list.push_back(abstract_v2);
  FuncGraphPtr new_top_fg = pipeline::Renormalize(resourcePtr, top_fg, args_spec_list);
  DoInline(new_top_fg);
  // Check the graph construct.
  auto after_g = getPyFun.CallAndParseRet("get_test_ad_fn", "imag_bprop_complex_input_complex_output");
  ASSERT_TRUE(CheckEqual(new_top_fg, after_g));
}

// Feature: Support automatic differentiation for complex number.
// Description: Test the imag bprop with real inputs and complex outputs.
// Expectation: The final func_graph construct is correct.
TEST_F(TestAD, TestImagBpropRealInputComplexOutput) {
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  // Parse the forward fg and do renormalize.
  auto g = getPyFun.CallAndParseRet("get_test_ad_fn", "imag_forward");
  AbstractBasePtrList args_spec_list;
  tensor::TensorPtr x_tensor = tensor::from_spec(kNumberTypeFloat32, std::vector<int64_t>{2, 3, 4, 5}, device::DeviceType::kCPU);
  AbstractBasePtr abstract_v1 = abstract::FromValue(x_tensor, true);
  args_spec_list.push_back(abstract_v1);
  FuncGraphPtr new_g = pipeline::Renormalize(resourcePtr, g, args_spec_list);
  // Get the fprop.
  FuncGraphPtr dg = Grad(new_g, opt::Optimizer::MakeEmptyOptimizer(resourcePtr));
  // Make the top cell.
  auto top_fg = std::make_shared<FuncGraph>();
  auto input = top_fg->add_parameter();
  auto dout = top_fg->add_parameter();
  auto fprop_caller = top_fg->NewCNode({NewValueNode(dg), input});
  auto bprop = top_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fprop_caller, NewValueNode<int64_t>(1)});
  auto grads = top_fg->NewCNode({bprop, dout});
  top_fg->set_output(top_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), grads, NewValueNode<int64_t>(1)}));
  // Do renormalize for the top cell.
  args_spec_list.push_back(abstract_v1);
  FuncGraphPtr new_top_fg = pipeline::Renormalize(resourcePtr, top_fg, args_spec_list);
  DoInline(new_top_fg);
  // Check the graph construct.
  auto after_g = getPyFun.CallAndParseRet("get_test_ad_fn", "imag_bprop_real_input_complex_output");
  ASSERT_TRUE(CheckEqual(new_top_fg, after_g));
}

// Feature: Support automatic differentiation for complex number.
// Description: Test the add bprop with real inputs and real outputs.
// Expectation: The final func_graph construct is correct.
TEST_F(TestAD, TestImagBpropRealInputRealOutput) {
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  // Parse the forward fg and do renormalize.
  auto g = getPyFun.CallAndParseRet("get_test_ad_fn", "add_forward");
  AbstractBasePtrList args_spec_list;
  tensor::TensorPtr x_tensor = tensor::from_spec(kNumberTypeFloat32, std::vector<int64_t>{2, 3, 4, 5}, device::DeviceType::kCPU);
  AbstractBasePtr abstract_v1 = abstract::FromValue(x_tensor, true);
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v1);
  FuncGraphPtr new_g = pipeline::Renormalize(resourcePtr, g, args_spec_list);
  // Get the fprop.
  FuncGraphPtr dg = Grad(new_g, opt::Optimizer::MakeEmptyOptimizer(resourcePtr));
  // Make the top cell.
  auto top_fg = std::make_shared<FuncGraph>();
  auto input_x = top_fg->add_parameter();
  auto input_y = top_fg->add_parameter();
  auto dout = top_fg->add_parameter();
  auto fprop_caller = top_fg->NewCNode({NewValueNode(dg), input_x, input_y});
  auto bprop = top_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fprop_caller, NewValueNode<int64_t>(1)});
  auto grads = top_fg->NewCNode({bprop, dout});
  top_fg->set_output(top_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), grads, NewValueNode<int64_t>(1)}));
  // Do renormalize for the top cell.
  args_spec_list.push_back(abstract_v1);
  FuncGraphPtr new_top_fg = pipeline::Renormalize(resourcePtr, top_fg, args_spec_list);
  DoInline(new_top_fg);
  // Check the graph construct.
  auto after_g = getPyFun.CallAndParseRet("get_test_ad_fn", "add_bprop");
  ASSERT_TRUE(CheckEqual(new_top_fg, after_g));
}

// Feature: Support automatic differentiation for complex number.
// Description: Test the add bprop with complex inputs and real outputs.
// Expectation: The final func_graph construct is correct.
TEST_F(TestAD, TestImagBpropComplexInputRealOutput) {
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  // Parse the forward fg and do renormalize.
  auto g = getPyFun.CallAndParseRet("get_test_ad_fn", "add_forward");
  AbstractBasePtrList args_spec_list;
  tensor::TensorPtr x_tensor = tensor::from_spec(kNumberTypeComplex64, std::vector<int64_t>{2, 3, 4, 5}, device::DeviceType::kCPU);
  AbstractBasePtr abstract_v1 = abstract::FromValue(x_tensor, true);
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v1);
  FuncGraphPtr new_g = pipeline::Renormalize(resourcePtr, g, args_spec_list);
  // Get the fprop.
  FuncGraphPtr dg = Grad(new_g, opt::Optimizer::MakeEmptyOptimizer(resourcePtr));
  // Make the top cell.
  auto top_fg = std::make_shared<FuncGraph>();
  auto input_x = top_fg->add_parameter();
  auto input_y = top_fg->add_parameter();
  auto dout = top_fg->add_parameter();
  auto fprop_caller = top_fg->NewCNode({NewValueNode(dg), input_x, input_y});
  auto bprop = top_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), fprop_caller, NewValueNode<int64_t>(1)});
  auto grads = top_fg->NewCNode({bprop, dout});
  top_fg->set_output(top_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), grads, NewValueNode<int64_t>(1)}));
  // Do renormalize for the top cell.
  tensor::TensorPtr y_tensor = tensor::from_spec(kNumberTypeFloat32, std::vector<int64_t>{2, 3, 4, 5}, device::DeviceType::kCPU);
  AbstractBasePtr abstract_v2 = abstract::FromValue(y_tensor, true);
  args_spec_list.push_back(abstract_v2);
  FuncGraphPtr new_top_fg = pipeline::Renormalize(resourcePtr, top_fg, args_spec_list);
  DoInline(new_top_fg);
  // Check the graph construct.
  auto after_g = getPyFun.CallAndParseRet("get_test_ad_fn", "add_bprop");
  ASSERT_TRUE(CheckEqual(new_top_fg, after_g));
}
}  // namespace ad
}  // namespace mindspore
