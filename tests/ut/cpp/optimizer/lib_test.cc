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
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "ir/value.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "pipeline/jit/resource.h"
#include "debug/draw.h"
#include "pipeline/jit/parse/data_converter.h"

namespace mindspore {
namespace opt {
using abstract::AnalysisResult;

class TestOptLib : public UT::Common {
 public:
  TestOptLib() : getPyFun("gtest_input.optimizer.opt_test", true), irpass() {}
  void SetUp() {
    UT::InitPythonPath();
    parse::data_converter::ClearObjectCache();
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  }
  FuncGraphPtr RunTransform(FuncGraphPtr gbefore, const SubstitutionList &transform) {
    equiv_node.clear();
    equiv_graph.clear();

    FuncGraphPtr gbefore_clone = BasicClone(gbefore);
    OptimizerPtr optimizer = std::make_shared<Optimizer>("ut_test", std::make_shared<pipeline::Resource>());
    transform(gbefore_clone, optimizer);
    return gbefore_clone;
  }
  FuncGraphPtr RunSubs(FuncGraphPtr before, std::vector<SubstitutionPtr> opts = {}) {
    SubstitutionList eq(opts);
    return RunTransform(before, eq);
  }
  bool CheckTransform(FuncGraphPtr gbefore, FuncGraphPtr gafter, const SubstitutionList &transform,
                      bool save_graphs = false) {
    equiv_node.clear();
    equiv_graph.clear();

    FuncGraphPtr gbefore_clone = BasicClone(gbefore);
    OptimizerPtr optimizer = std::make_shared<Optimizer>("ut_test", std::make_shared<pipeline::Resource>());
    transform(gbefore_clone, optimizer);
    if (save_graphs) {
      draw::Draw("before.dot", gbefore);
      draw::Draw("after.dot", gbefore_clone);
      draw::Draw("expected.dot", gafter);
    }
    return Isomorphic(gbefore_clone, gafter, &equiv_graph, &equiv_node);
  }
  bool CheckOpt(FuncGraphPtr before, FuncGraphPtr after, std::vector<SubstitutionPtr> opts = {},
                bool save_graphs = false) {
    if (nullptr == before || nullptr == after) {
      return false;
    }
    SubstitutionList eq(opts);
    return CheckTransform(before, after, eq, save_graphs);
  }

 public:
  UT::PyFuncGraphFetcher getPyFun;
  FuncGraphPairMapEquiv equiv_graph;
  NodeMapEquiv equiv_node;
  irpass::OptimizeIRPassLib irpass;
};

TEST_F(TestOptLib, test_simplify_always_true_false) {
  FuncGraphPtr before1 = getPyFun.CallAndParseRet("test_simplify_always_true_false", "before_1");
  FuncGraphPtr before2 = getPyFun.CallAndParseRet("test_simplify_always_true_false", "before_2");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_simplify_always_true_false", "after");
  auto patterns = std::vector<SubstitutionPtr>({irpass.switch_simplify_});
  ASSERT_TRUE(CheckOpt(before1, after, patterns));
  ASSERT_TRUE(CheckOpt(before2, after, patterns));
}

TEST_F(TestOptLib, test_inline) {
  FuncGraphPtr before1 = getPyFun.CallAndParseRet("test_inline", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_inline", "after");
  // add infer and renormalize
  std::shared_ptr<mindspore::pipeline::Resource> res = std::make_shared<mindspore::pipeline::Resource>();
  AbstractBasePtrList args_spec_list;
  tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), std::vector<int64_t>{2, 3});
  tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), std::vector<int64_t>{2, 3});

  AbstractBasePtr abstract_v1 = abstract::FromValue(x_tensor, true);
  AbstractBasePtr abstract_v2 = abstract::FromValue(y_tensor, true);
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);
  AnalysisResult result = pipeline::AbstractAnalyze(res, before1, args_spec_list);
  FuncGraphPtr new_graph = pipeline::ProgramSpecialize(res, before1, result.context);
  auto patterns = std::vector<SubstitutionPtr>({irpass.arithmetic_simplify_, irpass.switch_simplify_, irpass.inline_});
  ASSERT_TRUE(CheckOpt(new_graph, after, patterns));
}

TEST_F(TestOptLib, test_inline_successively) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_inline_successively", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_inline_successively", "after");
  auto patterns = std::vector<SubstitutionPtr>({irpass.inline_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_inline_closure) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_inline_closure", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_inline_closure", "after");
  auto patterns = std::vector<SubstitutionPtr>({irpass.inline_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_inline_deep_closure) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_inline_deep_closure", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_inline_deep_closure", "after");
  auto patterns = std::vector<SubstitutionPtr>({irpass.inline_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_inline_new_closure) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_inline_new_closure", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_inline_new_closure", "after");
  auto patterns = std::vector<SubstitutionPtr>({irpass.inline_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_inline_while) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_inline_while", "before");
  auto patterns = std::vector<SubstitutionPtr>({irpass.inline_});
  FuncGraphPtr after = RunSubs(before, patterns);
  ASSERT_TRUE(CheckOpt(before, after, patterns, true));
}

TEST_F(TestOptLib, test_arithmetic) {
  FuncGraphPtr b1_0 = getPyFun.CallAndParseRet("test_arithmetic", "multiply_by_zero_l");
  FuncGraphPtr b2_0 = getPyFun.CallAndParseRet("test_arithmetic", "multiply_by_zero_r");
  FuncGraphPtr b1 = getPyFun.CallAndParseRet("test_arithmetic", "multiply_by_one_l");
  FuncGraphPtr b2 = getPyFun.CallAndParseRet("test_arithmetic", "multiply_by_one_r");
  FuncGraphPtr b3 = getPyFun.CallAndParseRet("test_arithmetic", "add_zero_l");
  FuncGraphPtr b4 = getPyFun.CallAndParseRet("test_arithmetic", "add_zero_r");
  FuncGraphPtr b5 = getPyFun.CallAndParseRet("test_arithmetic", "elim_identity");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_arithmetic", "after");
  FuncGraphPtr after_0 = getPyFun.CallAndParseRet("test_arithmetic", "after_0");

  auto patterns = std::vector<SubstitutionPtr>({irpass.arithmetic_simplify_});

  ASSERT_TRUE(CheckOpt(b1_0, after_0, patterns));
  ASSERT_TRUE(CheckOpt(b2_0, after_0, patterns));
  ASSERT_TRUE(CheckOpt(b1, after, patterns));
  ASSERT_TRUE(CheckOpt(b2, after, patterns));
  ASSERT_TRUE(CheckOpt(b3, after, patterns));
  ASSERT_TRUE(CheckOpt(b4, after, patterns));
  ASSERT_TRUE(CheckOpt(b5, after, patterns));
}

TEST_F(TestOptLib, test_elim_cast_same_dtype) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elim_cast_same_dtype", "fp32_cast_fp32");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elim_cast_same_dtype", "after");
  // construct such case that cast srcT equal dstT
  auto &inputs = before->output()->cast<CNodePtr>()->inputs();
  if (inputs.size() > 2) {
    auto cast_node = inputs[0];
    auto cast_py = cast_node->cast<ValueNodePtr>()->value()->cast<PrimitivePyPtr>();
    cast_py->set_attr("SrcT", TypeIdToType(kNumberTypeFloat32));
    cast_py->set_attr("DstT", TypeIdToType(kNumberTypeFloat32));

    auto x_node = inputs[1];
    std::vector<int64_t> shp = {2, 3};
    tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    auto x_abstract = x_tensor->ToAbstract();
    x_node->set_abstract(x_abstract);

    TypePtr t = std::make_shared<TensorType>(std::make_shared<Float>(32));
    ValueNodePtr val = std::make_shared<ValueNode>(t);
    auto t_abstract = t->ToAbstract();
    val->set_abstract(t_abstract);
    before->output()->cast<CNodePtr>()->set_input(2, val);
  }
  FuncGraphPtr gbefore_clone = BasicClone(before);
  auto patterns = std::vector<SubstitutionPtr>({irpass.cast_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));

  TypePtr t = std::make_shared<Float>(32);
  ValueNodePtr val = std::make_shared<ValueNode>(t);
  auto t_abstract = t->ToAbstract();
  val->set_abstract(t_abstract);
  gbefore_clone->output()->cast<CNodePtr>()->set_input(2, val);
  ASSERT_TRUE(CheckOpt(gbefore_clone, after, patterns));
}

TEST_F(TestOptLib, test_elim_reshape_same_shape) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("elim_reshape_same_shape", "reshape_to_2_3");
  FuncGraphPtr after = getPyFun.CallAndParseRet("elim_reshape_same_shape", "after");
  // construct such case that shape is equal to reshape target
  auto &inputs = before->output()->cast<CNodePtr>()->inputs();
  if (inputs.size() > 1) {
    auto x_node = inputs[1];
    std::vector<int64_t> shp = {2, 3};
    tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    auto x_abstract = x_tensor->ToAbstract();
    x_node->set_abstract(x_abstract);
    before->output()->set_abstract(x_abstract);
  }
  auto patterns = std::vector<SubstitutionPtr>({irpass.reshape_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
  if (inputs.size() > 1) {
    auto x_node = inputs[1];
    std::vector<int64_t> shp = {3, 2};
    tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    auto x_abstract = x_tensor->ToAbstract();
    x_node->set_abstract(x_abstract);
  }
  ASSERT_FALSE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, elim_two_reshape) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("elim_two_reshape", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("elim_two_reshape", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.reshape_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, elim_two_cast) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("elim_two_cast", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("elim_two_cast", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.cast_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_elim_transpose) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elim_transpose", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elim_transpose", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.transpose_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_elim_depend_value) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elim_depend_value", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elim_depend_value", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.depend_value_elim_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_elim_tile_multiply_one) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elim_tile_multiply_one", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elim_tile_multiply_one", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.tile_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns, true));
}

TEST_F(TestOptLib, test_elim_reduce_mean_shape_one) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elim_reduce_mean_shape_one", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elim_reduce_mean_shape_one", "after");

  // construct such case that input x shape is (1), keepdims is true
  auto inputs = before->output()->cast<CNodePtr>()->inputs();
  if (inputs.size() > 2) {
    auto x_node = inputs[1];
    std::vector<int64_t> shp = {1};
    tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    auto x_abstract = x_tensor->ToAbstract();
    x_node->set_abstract(x_abstract);

    auto reduce_node = inputs[0];
    auto reduce = reduce_node->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
    reduce->set_attr("keep_dims", std::make_shared<BoolImm>(true));
  }

  auto patterns = std::vector<SubstitutionPtr>({irpass.reduce_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_elim_all_shape_one) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elim_all_shape_one", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elim_all_shape_one", "after");

  // construct such case that input x shape is (1) keep_dims is true
  auto inputs = before->output()->cast<CNodePtr>()->inputs();
  if (inputs.size() > 2) {
    auto x_node = inputs[1];
    std::vector<int64_t> shp = {1};
    tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    auto x_abstract = x_tensor->ToAbstract();
    x_node->set_abstract(x_abstract);

    auto reduce_node = inputs[0];
    auto reduce = reduce_node->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
    reduce->set_attr("keep_dims", std::make_shared<BoolImm>(true));
  }
  auto patterns = std::vector<SubstitutionPtr>({irpass.reduce_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_elim_sum_shape_one) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elim_sum_shape_one", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elim_sum_shape_one", "after");

  // construct such case that input x shape is (1) keepdims is true
  auto inputs = before->output()->cast<CNodePtr>()->inputs();
  if (inputs.size() > 2) {
    auto x_node = inputs[1];
    std::vector<int64_t> shp = {1};
    tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
    auto x_abstract = x_tensor->ToAbstract();
    x_node->set_abstract(x_abstract);

    auto reduce_node = inputs[0];
    auto reduce = reduce_node->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
    reduce->set_attr("keep_dims", std::make_shared<BoolImm>(true));
  }
  auto patterns = std::vector<SubstitutionPtr>({irpass.reduce_eliminate_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_tuple_getitem) {
  FuncGraphPtr make_get_0 = getPyFun.CallAndParseRet("test_tuple_getitem", "make_get_0");
  FuncGraphPtr make_get_1 = getPyFun.CallAndParseRet("test_tuple_getitem", "make_get_1");
  FuncGraphPtr after_0 = getPyFun.CallAndParseRet("test_tuple_getitem", "after_0");
  FuncGraphPtr after_1 = getPyFun.CallAndParseRet("test_tuple_getitem", "after_1");

  FuncGraphPtr make_get_const = std::make_shared<FuncGraph>();
  auto value_node_1 = NewValueNode(static_cast<int64_t>(1));
  auto value_node_2 = NewValueNode(static_cast<int64_t>(2));
  std::vector<int64_t> vec{1, 2};
  auto value_node_tuple = NewValueNode(MakeValue(vec));
  std::vector<AnfNodePtr> node_list{NewValueNode(prim::kPrimTupleGetItem), value_node_tuple, value_node_1};
  auto get_item = make_get_const->NewCNode(node_list);
  make_get_const->set_output(get_item);

  FuncGraphPtr after_2 = std::make_shared<FuncGraph>();
  after_2->set_output(value_node_2);

  auto patterns = std::vector<SubstitutionPtr>({irpass.item_tuple_or_list_eliminate_});
  ASSERT_TRUE(CheckOpt(make_get_0, after_0, patterns));
  ASSERT_TRUE(CheckOpt(make_get_1, after_1, patterns));
  ASSERT_TRUE(CheckOpt(make_get_const, after_2, patterns));
}

TEST_F(TestOptLib, test_tuple_setitem) {
  FuncGraphPtr before_0 = getPyFun.CallAndParseRet("test_tuple_setitem", "before_0");
  FuncGraphPtr before_1 = getPyFun.CallAndParseRet("test_tuple_setitem", "before_1");
  FuncGraphPtr after_0 = getPyFun.CallAndParseRet("test_tuple_setitem", "after_0");
  FuncGraphPtr after_1 = getPyFun.CallAndParseRet("test_tuple_setitem", "after_1");

  auto patterns = std::vector<SubstitutionPtr>({irpass.item_tuple_or_list_eliminate_});

  ASSERT_TRUE(CheckOpt(before_0, after_0, patterns));
  ASSERT_TRUE(CheckOpt(before_1, after_1, patterns));
}

TEST_F(TestOptLib, test_tuple_get_set_item) {
  FuncGraphPtr before_0 = getPyFun.CallAndParseRet("test_tuple_get_set_item", "before_0");
  FuncGraphPtr after_0 = getPyFun.CallAndParseRet("test_tuple_get_set_item", "after_0");
  FuncGraphPtr before_1 = getPyFun.CallAndParseRet("test_tuple_get_set_item", "before_0");
  FuncGraphPtr after_1 = getPyFun.CallAndParseRet("test_tuple_get_set_item", "after_0");

  auto patterns = std::vector<SubstitutionPtr>({irpass.item_tuple_or_list_eliminate_});

  ASSERT_TRUE(CheckOpt(before_0, after_0, patterns));
  ASSERT_TRUE(CheckOpt(before_1, after_1, patterns));
}

TEST_F(TestOptLib, test_partial) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_partial", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_partial", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.partial_eliminate_});

  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_replace_applicator) {
  FuncGraphPtr before1 = getPyFun.CallAndParseRet("test_replace_applicator", "before1");
  FuncGraphPtr before2 = getPyFun.CallAndParseRet("test_replace_applicator", "before2");
  FuncGraphPtr before3 = getPyFun.CallAndParseRet("test_replace_applicator", "before3");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_replace_applicator", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.replace_applicator_});

  ASSERT_TRUE(CheckOpt(before1, after, patterns));
  ASSERT_TRUE(CheckOpt(before2, after, patterns));
  ASSERT_TRUE(CheckOpt(before3, before3, patterns));
}

TEST_F(TestOptLib, test_specialize_on_graph_arguments) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_specialize_on_graph_arguments", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_specialize_on_graph_arguments", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.specialize_transform_});

  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_incorporate_getitem) {
  FuncGraphPtr before1 = getPyFun.CallAndParseRet("test_incorporate_getitem", "before1");
  FuncGraphPtr before2 = getPyFun.CallAndParseRet("test_incorporate_getitem", "before2");
  FuncGraphPtr after1 = getPyFun.CallAndParseRet("test_incorporate_getitem", "after1");
  FuncGraphPtr after2 = getPyFun.CallAndParseRet("test_incorporate_getitem", "after2");

  auto patterns = std::vector<SubstitutionPtr>({irpass.incorporate_getitem_set_});

  ASSERT_TRUE(CheckOpt(before1, after1, patterns));
  ASSERT_TRUE(CheckOpt(before2, after2, patterns));
}

TEST_F(TestOptLib, test_incorporate_getitem_through_switch) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_incorporate_getitem_through_switch", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_incorporate_getitem_through_switch", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.incorporate_getitem_set_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_incorporate_call) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_incorporate_call", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_incorporate_call", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.incorporate_call_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_incorporate_call_through_switch) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_incorporate_call_through_switch", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_incorporate_call_through_switch", "after");
  auto patterns = std::vector<SubstitutionPtr>({
    irpass.incorporate_call_switch_,
    irpass.incorporate_call_,
    irpass.arithmetic_simplify_,
  });
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_float_tuple_getitem_through_switch) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_float_tuple_getitem_through_switch", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_float_tuple_getitem_through_switch", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.float_tuple_getitem_switch_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_merge_addn) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_merge_addn", "before");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_merge_addn", "after");

  auto patterns = std::vector<SubstitutionPtr>({irpass.merge_addn_});
  ASSERT_TRUE(CheckOpt(before, after, patterns));
}

TEST_F(TestOptLib, test_filter_addn_zero) {
  FuncGraphPtr before1 = getPyFun.CallAndParseRet("test_addn_zero", "before_1");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_addn_zero", "after");
  FuncGraphPtr before2 = getPyFun.CallAndParseRet("test_addn_zero", "before_2");
  FuncGraphPtr before3 = getPyFun.CallAndParseRet("test_addn_zero", "before_3");
  FuncGraphPtr before4 = getPyFun.CallAndParseRet("test_addn_zero", "before_4");
  auto patterns = std::vector<SubstitutionPtr>({irpass.addn_zero_filter_});
  ASSERT_TRUE(CheckOpt(before1, after, patterns));
  ASSERT_TRUE(CheckOpt(before2, after, patterns));
  ASSERT_TRUE(CheckOpt(before3, after, patterns));
  ASSERT_TRUE(CheckOpt(before4, before4, patterns));
}

TEST_F(TestOptLib, test_minmax_grad) {
  FuncGraphPtr before11 = getPyFun.CallAndParseRet("test_minmax_grad", "before_11");
  FuncGraphPtr before12 = getPyFun.CallAndParseRet("test_minmax_grad", "before_12");
  FuncGraphPtr before2 = getPyFun.CallAndParseRet("test_minmax_grad", "before_2");
  FuncGraphPtr before31 = getPyFun.CallAndParseRet("test_minmax_grad", "before_31");
  FuncGraphPtr before32 = getPyFun.CallAndParseRet("test_minmax_grad", "before_32");
  FuncGraphPtr before4 = getPyFun.CallAndParseRet("test_minmax_grad", "before_4");
  auto patterns = std::vector<SubstitutionPtr>({irpass.minmaximum_grad_});
  ASSERT_TRUE(CheckOpt(before11, before11, patterns));
  ASSERT_TRUE(CheckOpt(before12, before12, patterns));
  ASSERT_TRUE(CheckOpt(before2, before2, patterns));
  ASSERT_TRUE(CheckOpt(before31, before31, patterns));
  ASSERT_TRUE(CheckOpt(before32, before32, patterns));
  ASSERT_TRUE(CheckOpt(before4, before4, patterns));
}

TEST_F(TestOptLib, test_reducesum_one) {
  FuncGraphPtr before1 = getPyFun.CallAndParseRet("test_reducesum_one", "before_1");
  FuncGraphPtr before2 = getPyFun.CallAndParseRet("test_reducesum_one", "before_2");
  FuncGraphPtr before3 = getPyFun.CallAndParseRet("test_reducesum_one", "before_3");
  FuncGraphPtr before4 = getPyFun.CallAndParseRet("test_reducesum_one", "before_4");
  FuncGraphPtr after1 = getPyFun.CallAndParseRet("test_reducesum_one", "after_1");
  FuncGraphPtr after2 = getPyFun.CallAndParseRet("test_reducesum_one", "after_2");
  FuncGraphPtr after3 = getPyFun.CallAndParseRet("test_reducesum_one", "after_3");
  auto patterns = std::vector<SubstitutionPtr>({irpass.reduce_eliminate_});

  std::vector<int64_t> shp = {3, 2, 2, 1};
  tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp);
  auto x_abstract = x_tensor->ToAbstract();

  std::vector<int64_t> shp2 = {3, 2, 1, 1};
  tensor::TensorPtr x_tensor2 = std::make_shared<tensor::Tensor>(kFloat32->type_id(), shp2);
  auto x_abstract2 = x_tensor2->ToAbstract();

  auto inputs = before1->output()->cast<CNodePtr>()->inputs();
  if (inputs.size() > 1) {
    auto x_node = inputs[1];
    x_node->set_abstract(x_abstract);
  }
  ASSERT_TRUE(CheckOpt(before1, after1, patterns));

  auto inputs2 = before2->output()->cast<CNodePtr>()->inputs();
  if (inputs2.size() > 1) {
    auto x_node2 = inputs2[1];
    x_node2->set_abstract(x_abstract2);
  }
  ASSERT_TRUE(CheckOpt(before2, after1, patterns));

  auto inputs3 = before2->output()->cast<CNodePtr>()->inputs();
  if (inputs3.size() > 1) {
    auto x_node3 = inputs3[1];
    x_node3->set_abstract(x_abstract);
  }
  ASSERT_TRUE(CheckOpt(before2, before2, patterns));

  auto inputs4 = before3->output()->cast<CNodePtr>()->inputs();
  if (inputs4.size() > 1) {
    auto x_node4 = inputs4[1];
    x_node4->set_abstract(x_abstract);
  }
  ASSERT_TRUE(CheckOpt(before3, after2, patterns));

  auto inputs5 = before4->output()->cast<CNodePtr>()->inputs();
  if (inputs5.size() > 1) {
    auto x_node5 = inputs5[1];
    x_node5->set_abstract(x_abstract2);
  }
  ASSERT_TRUE(CheckOpt(before4, after3, patterns));
}

TEST_F(TestOptLib, test_print_tuple_wrapper) {
  FuncGraphPtr before1 = getPyFun.CallAndParseRet("test_print_tuple_wrapper", "before1");
  FuncGraphPtr before2 = getPyFun.CallAndParseRet("test_print_tuple_wrapper", "before2");
  FuncGraphPtr before3 = getPyFun.CallAndParseRet("test_print_tuple_wrapper", "before3");
  FuncGraphPtr after1 = getPyFun.CallAndParseRet("test_print_tuple_wrapper", "after1");
  FuncGraphPtr after2 = getPyFun.CallAndParseRet("test_print_tuple_wrapper", "after2");
  auto patterns = std::vector<SubstitutionPtr>({irpass.print_tuple_wrapper_});
  ASSERT_TRUE(CheckOpt(before1, after1, patterns));
  ASSERT_TRUE(CheckOpt(before2, after2, patterns));
  ASSERT_TRUE(CheckOpt(before3, before3, patterns));
}

TEST_F(TestOptLib, test_constant_duplicate_mul) {
  FuncGraphPtr beforell = getPyFun.CallAndParseRet("test_constant_duplicate_mul", "beforell");
  FuncGraphPtr beforelr = getPyFun.CallAndParseRet("test_constant_duplicate_mul", "beforelr");
  FuncGraphPtr beforerl = getPyFun.CallAndParseRet("test_constant_duplicate_mul", "beforerl");
  FuncGraphPtr beforerr = getPyFun.CallAndParseRet("test_constant_duplicate_mul", "beforerr");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_constant_duplicate_mul", "after");
  auto patterns = std::vector<SubstitutionPtr>({irpass.arithmetic_simplify_});
  ASSERT_TRUE(CheckOpt(beforell, after, patterns));
  ASSERT_TRUE(CheckOpt(beforelr, after, patterns));
  ASSERT_TRUE(CheckOpt(beforerl, after, patterns));
  ASSERT_TRUE(CheckOpt(beforerr, after, patterns));
}

TEST_F(TestOptLib, test_adjust_allreduce_mul_add) {
  FuncGraphPtr beforell = getPyFun.CallAndParseRet("test_adjust_allreduce_mul_add", "beforell");
  FuncGraphPtr beforelr = getPyFun.CallAndParseRet("test_adjust_allreduce_mul_add", "beforelr");
  FuncGraphPtr beforerl = getPyFun.CallAndParseRet("test_adjust_allreduce_mul_add", "beforerl");
  FuncGraphPtr beforerr = getPyFun.CallAndParseRet("test_adjust_allreduce_mul_add", "beforerr");
  FuncGraphPtr after1 = getPyFun.CallAndParseRet("test_adjust_allreduce_mul_add", "after1");
  FuncGraphPtr before2r = getPyFun.CallAndParseRet("test_adjust_allreduce_mul_add", "before2r");
  FuncGraphPtr before2l = getPyFun.CallAndParseRet("test_adjust_allreduce_mul_add", "before2l");
  FuncGraphPtr after2 = getPyFun.CallAndParseRet("test_adjust_allreduce_mul_add", "after2");
  auto patterns = std::vector<SubstitutionPtr>({irpass.adjust_all_reduce_mul_add_});
  ASSERT_TRUE(CheckOpt(beforell, after1, patterns, true));
  ASSERT_TRUE(CheckOpt(beforelr, after1, patterns));
  ASSERT_TRUE(CheckOpt(beforerl, after1, patterns));
  ASSERT_TRUE(CheckOpt(beforerr, after1, patterns));
}

TEST_F(TestOptLib, test_row_tensor) {
  FuncGraphPtr before_get_indices = getPyFun.CallAndParseRet("test_row_tensor", "before_get_indices");
  FuncGraphPtr after_get_indices = getPyFun.CallAndParseRet("test_row_tensor", "after_get_indices");
  FuncGraphPtr before_get_values = getPyFun.CallAndParseRet("test_row_tensor", "before_get_values");
  FuncGraphPtr after_get_values = getPyFun.CallAndParseRet("test_row_tensor", "after_get_values");
  FuncGraphPtr before_get_dense_shape = getPyFun.CallAndParseRet("test_row_tensor", "before_get_dense_shape");
  FuncGraphPtr after_get_dense_shape = getPyFun.CallAndParseRet("test_row_tensor", "after_get_dense_shape");
  auto patterns = std::vector<SubstitutionPtr>({irpass.row_tensor_eliminate_});
  ASSERT_TRUE(CheckOpt(before_get_indices, after_get_indices, patterns));
  ASSERT_TRUE(CheckOpt(before_get_values, after_get_values, patterns));
  ASSERT_TRUE(CheckOpt(before_get_dense_shape, after_get_dense_shape, patterns));
}

TEST_F(TestOptLib, test_sparse_tensor) {
  FuncGraphPtr before_get_indices = getPyFun.CallAndParseRet("test_sparse_tensor", "before_get_indices");
  FuncGraphPtr after_get_indices = getPyFun.CallAndParseRet("test_sparse_tensor", "after_get_indices");
  FuncGraphPtr before_get_values = getPyFun.CallAndParseRet("test_sparse_tensor", "before_get_values");
  FuncGraphPtr after_get_values = getPyFun.CallAndParseRet("test_sparse_tensor", "after_get_values");
  FuncGraphPtr before_get_dense_shape = getPyFun.CallAndParseRet("test_sparse_tensor", "before_get_dense_shape");
  FuncGraphPtr after_get_dense_shape = getPyFun.CallAndParseRet("test_sparse_tensor", "after_get_dense_shape");
  auto patterns = std::vector<SubstitutionPtr>({irpass.sparse_tensor_eliminate_});
  ASSERT_TRUE(CheckOpt(before_get_indices, after_get_indices, patterns));
  ASSERT_TRUE(CheckOpt(before_get_values, after_get_values, patterns));
  ASSERT_TRUE(CheckOpt(before_get_dense_shape, after_get_dense_shape, patterns));
}
}  // namespace opt
}  // namespace mindspore
