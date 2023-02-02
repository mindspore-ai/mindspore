/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "pybind11/pybind11.h"

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/manager.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "pipeline/static_analysis/helper.h"
#include "frontend/operator/ops.h"
#include "include/common/debug/draw.h"
#include "ir/tensor.h"
#include "utils/symbolic.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace abstract {
namespace py = pybind11;
class UTPrimUtils {
 public:
  using AbstractTensorPtr = std::shared_ptr<AbstractTensor>;
  using AbstractTuplePtr = std::shared_ptr<AbstractTuple>;

  static const std::shared_ptr<Float> kF32;
  static const std::shared_ptr<Float> kF64;
  static const std::shared_ptr<Int> kI16;
  static const std::shared_ptr<Int> kI64;
  static const std::shared_ptr<UInt> kU64;

  static std::shared_ptr<AbstractType> TypeToAbstract(TypePtr t) { return std::make_shared<AbstractType>(t); }

  static AbstractTensorPtr ArrayFloat64Of(std::initializer_list<int64_t> shp) {
    auto ele = std::make_shared<AbstractScalar>(kAnyValue, kFloat64);
    return std::make_shared<AbstractTensor>(ele, std::make_shared<Shape>(shp));
  }

  static AbstractTensorPtr ArrayFloat32Of(std::initializer_list<int64_t> shp) {
    auto ele = std::make_shared<AbstractScalar>(kAnyValue, kFloat32);
    return std::make_shared<AbstractTensor>(ele, std::make_shared<Shape>(shp));
  }

  static AbstractTensorPtr ArrayInt32Of(std::initializer_list<int64_t> shp) {
    auto ele = std::make_shared<AbstractScalar>(kAnyValue, kInt64);
    return std::make_shared<AbstractTensor>(ele, std::make_shared<Shape>(shp));
  }

  static AbstractTuplePtr ShapeOf(std::initializer_list<int64_t> vals) {
    AbstractBasePtrList te;
    for (auto v : vals) {
      te.push_back(std::make_shared<AbstractScalar>(v));
    }
    return std::make_shared<AbstractTuple>(te);
  }

  static AbstractListPtr ListShapeOf(std::initializer_list<int64_t> vals) {
    AbstractBasePtrList te;
    for (auto v : vals) {
      te.push_back(std::make_shared<AbstractScalar>(v));
    }
    return std::make_shared<AbstractList>(te);
  }
};
const std::shared_ptr<Float> UTPrimUtils::kF64 = std::make_shared<Float>(64);
const std::shared_ptr<Float> UTPrimUtils::kF32 = std::make_shared<Float>(32);
const std::shared_ptr<Int> UTPrimUtils::kI16 = std::make_shared<Int>(16);
const std::shared_ptr<Int> UTPrimUtils::kI64 = std::make_shared<Int>(64);
const std::shared_ptr<UInt> UTPrimUtils::kU64 = std::make_shared<UInt>(64);
namespace {
/* skip ut test cases temporarily
AbstractBasePtr ArrayOfTensor(const TypePtr &t, std::initializer_list<int64_t> shp) {
  auto shape = std::vector<int64_t>(shp);
  auto tensor = std::make_shared<tensor::Tensor>(t->type_id(), shape);
  return ToAbstract(tensor);
}
*/
}  // namespace

class TestPrim : public UT::Common {
 public:
  TestPrim() : getPyFun("gtest_input.pipeline.infer", true) {}
  void SetUp();
  void TearDown();
  AnalysisEnginePtr engine_;
  UT::PyFuncGraphFetcher getPyFun;
};

void TestPrim::SetUp() { engine_ = SetupAnalysisEngine(); }

void TestPrim::TearDown() {
  // destroy resource
}

static FuncGraphPtr MakeFuncGraph(const PrimitivePtr prim, uint64_t nparam) {
  // build the func_graph manually, eg:
  // MakeFuncGraph(std::make_shared<Primitive>("scalar_add"), 2) means:
  /* python source code:
   * @mindspore
   * def f(x, y):
   *     return x + y
   */
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim));
  for (uint64_t i = 0; i < nparam; i++) {
    inputs.push_back(func_graph->add_parameter());
  }
  CNodePtr cnode_prim = func_graph->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(cnode_prim);
  CNodePtr cnode_return = func_graph->NewCNode(inputs);
  func_graph->set_return(cnode_return);
  return func_graph;
}

TEST_F(TestPrim, test_typeof) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 1;

  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  args_spec_list.push_back(abstract_v1);

  auto prim_typeof = std::make_shared<Primitive>("typeof");
  FuncGraphPtr func_graph = MakeFuncGraph(prim_typeof, 1);
  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  res->dump();
  TypePtr res_value = res->GetValueTrack()->cast<TypePtr>();
  res_value->dump();
  ASSERT_TRUE(*res_value == Int(64));
}

TEST_F(TestPrim, test_list_reduce) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 1;

  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  AbstractBasePtr abstract_v2 = FromValue(v1, false);
  auto abstract_list = std::make_shared<AbstractList>(AbstractBasePtrList({abstract_v1, abstract_v2}));
  auto prim_scalar_add = std::make_shared<Primitive>(prim::kScalarAdd);
  AbstractBasePtr abstract_func = ToAbstract(prim_scalar_add);

  args_spec_list.push_back(abstract_func);
  args_spec_list.push_back(abstract_list);
  args_spec_list.push_back(abstract_v1);

  auto prim_list_reduce = std::make_shared<Primitive>("list_reduce");
  FuncGraphPtr func_graph = MakeFuncGraph(prim_list_reduce, 3);
  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  res->dump();
  TypePtr res_type = res->GetTypeTrack();
  res_type->dump();
  ASSERT_TRUE(*res_type == Int(64));
}

TEST_F(TestPrim, test_array_to_scalar) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 1;

  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  auto abstract_a1 = std::make_shared<AbstractTensor>(abstract_v1, std::make_shared<Shape>());

  args_spec_list.push_back(abstract_a1);

  auto prim_array_to_scalar = std::make_shared<Primitive>("array_to_scalar");
  FuncGraphPtr func_graph = MakeFuncGraph(prim_array_to_scalar, 1);
  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  res->dump();
  TypePtr res_type = res->BuildType();
  res_type->dump();
  ASSERT_TRUE(*res_type == Int(64));
}

TEST_F(TestPrim, test_J_1) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 1;

  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  args_spec_list.push_back(abstract_v1);

  auto prim_J = std::make_shared<Primitive>("J");
  FuncGraphPtr func_graph = MakeFuncGraph(prim_J, 1);
  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  AbstractJTaggedPtr res_J = dyn_cast<AbstractJTagged>(res);
  ASSERT_TRUE(res_J != nullptr);
  ASSERT_TRUE(*(res_J->element()) == *abstract_v1);
}

TEST_F(TestPrim, test_J_2) {
  // def add(x):
  //   return x + x
  // def f(x):
  //   return J(add)(x)
  std::vector<AnfNodePtr> inputs;
  FuncGraphPtr func_graph1 = std::make_shared<FuncGraph>();
  inputs.push_back(NewValueNode(prim::kPrimScalarAdd));
  auto x = func_graph1->add_parameter();
  inputs.push_back(x);
  inputs.push_back(x);
  CNodePtr cnode1 = func_graph1->NewCNode(inputs);
  func_graph1->set_return(cnode1);

  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  inputs.clear();
  auto x1 = func_graph->add_parameter();
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimJ));
  inputs.push_back(NewValueNode(func_graph1));
  CNodePtr jf = func_graph->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(jf);
  inputs.push_back(x1);
  CNodePtr jf_jx = func_graph->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(jf_jx);
  CNodePtr cnode_return = func_graph->NewCNode(inputs);
  func_graph->set_return(cnode_return);

  int64_t v1 = 1;
  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  AbstractBasePtrList args_spec_list = {abstract_v1};
  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  res->dump();
  AbstractTuplePtr res_J = dyn_cast<AbstractTuple>(res);
  ASSERT_TRUE(res_J != nullptr);
  auto res_J_0 = res_J->elements()[0];
  ASSERT_TRUE(res_J_0 != nullptr);
  ASSERT_TRUE(*res_J_0 == *(FromValue(static_cast<int64_t>(2), false)));
  AbstractFunctionPtr res_J_1 = dyn_cast<AbstractFunction>(res_J->elements()[1]);
  ASSERT_TRUE(res_J_1 != nullptr);
}

// tail half
TEST_F(TestPrim, test_switch1) {
  PrimitivePtr switch_ = std::make_shared<Primitive>("Switch");
  FuncGraphPtr func_graph = MakeFuncGraph(switch_, 3);

  AbstractBasePtr arg0 = FromValue(true, false);
  AbstractBasePtr arg1 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtr arg2 = FromValue(static_cast<int64_t>(2), false);
  AbstractBasePtrList args_spec_list = {arg0, arg1, arg2};

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *arg1);
}

TEST_F(TestPrim, test_switch2) {
  PrimitivePtr switch_ = std::make_shared<Primitive>("Switch");
  FuncGraphPtr func_graph = MakeFuncGraph(switch_, 3);

  AbstractBasePtr arg0 = FromValue(false, false);
  AbstractBasePtr arg1 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtr arg2 = FromValue(static_cast<int64_t>(2), false);
  AbstractBasePtrList args_spec_list = {arg0, arg1, arg2};

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "make result res: " << res->ToString();
  MS_LOG(INFO) << "make result arg2: " << arg2->ToString();
  ASSERT_TRUE(*res == *arg2);
}

TEST_F(TestPrim, test_identity) {
  PrimitivePtr identity = std::make_shared<Primitive>("identity");
  FuncGraphPtr func_graph = MakeFuncGraph(identity, 1);

  AbstractBasePtr abstract_v1 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtrList args_spec_list = {abstract_v1};

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *abstract_v1);
}

TEST_F(TestPrim, test_broadcast_shape) {
  PrimitivePtr broadcast_shape = std::make_shared<Primitive>("broadcast_shape");
  FuncGraphPtr func_graph = MakeFuncGraph(broadcast_shape, 2);

  auto a = UTPrimUtils::ShapeOf({Shape::kShapeDimAny, Shape::kShapeDimAny});
  auto b = UTPrimUtils::ShapeOf({Shape::kShapeDimAny});
  std::vector<Any> expected{Shape::kShapeDimAny, Shape::kShapeDimAny};

  AbstractBasePtrList args_spec_list = {a, b};

  AbstractTuplePtr res = dyn_cast<AbstractTuple>(engine_->Run(func_graph, args_spec_list).eval_result->abstract());

  auto ret = res->BuildValue()->cast<ValueTuplePtr>()->value();
  std::vector<ValuePtr> element_list = {MakeValue(Shape::kShapeDimAny), MakeValue(Shape::kShapeDimAny)};
  ASSERT_TRUE(ret.size() == element_list.size());
  for (int64_t i = 0; i < element_list.size(); i++) {
    ASSERT_TRUE(*ret[i] == *element_list[i]);
  }
}

TEST_F(TestPrim, test_partial) {
  PrimitivePtr prim = prim::kPrimPartial;
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 3);

  PrimitivePtr add = prim::kPrimScalarAdd;
  AbstractBasePtr abstract_add = ToAbstract(add);
  AbstractBasePtr abstract_v1 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtr abstract_v2 = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtrList args_spec_list = {abstract_add, abstract_v1, abstract_v2};

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  AbstractBasePtrList fn_args_list = {abstract_v1, abstract_v2};
  auto expected = std::make_shared<PartialAbstractClosure>(
    std::make_shared<PrimitiveAbstractClosure>(prim::kPrimScalarAdd), fn_args_list);
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();
  ASSERT_TRUE(res->ToString() == expected->ToString());
}

/// Feature: Check inference result of primitive by build a FuncGraph with single primitive.
/// Description: Check inference result of primitive EnvironSet.
/// Expectation: Equal
TEST_F(TestPrim, test_environ_set) {
  FuncGraphPtr graph_embed = MakeFuncGraph(prim::kPrimEmbed, 1);
  AbstractBasePtr abstract_x = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtrList args_spec_list = {abstract_x};
  AbstractBasePtr embed_x = engine_->Run(graph_embed, args_spec_list).eval_result->abstract();

  FuncGraphPtr func_graph = MakeFuncGraph(prim::kPrimEnvironSet, 3);

  AbstractBasePtr abstract_environ = MakeEnvironAbstract();
  AbstractBasePtr abstract_y = FromValue(static_cast<int64_t>(2), false);
  args_spec_list = {abstract_environ, embed_x, abstract_y};

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  AbstractBasePtr exp = std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
  ASSERT_TRUE(*res == *exp);
}

/// Feature: Check inference result of primitive by build a FuncGraph with single primitive.
/// Description: Check inference result of primitive EnvironGet.
/// Expectation: Equal
TEST_F(TestPrim, test_environ_get) {
  FuncGraphPtr graph_embed = MakeFuncGraph(prim::kPrimEmbed, 1);
  AbstractBasePtr abstract_x = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtrList args_spec_list = {abstract_x};
  AbstractBasePtr embed_x = engine_->Run(graph_embed, args_spec_list).eval_result->abstract();

  FuncGraphPtr graph_environ_set = MakeFuncGraph(prim::kPrimEnvironSet, 3);

  AbstractBasePtr abstract_environ = MakeEnvironAbstract();
  AbstractBasePtr abstract_y = FromValue(static_cast<int64_t>(2), false);
  args_spec_list = {abstract_environ, embed_x, abstract_y};

  AbstractBasePtr res = engine_->Run(graph_environ_set, args_spec_list).eval_result->abstract();
  AbstractBasePtr exp = std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
  ASSERT_TRUE(*res == *exp);

  FuncGraphPtr graph_environ_get = MakeFuncGraph(prim::kPrimEnvironGet, 3);

  AbstractBasePtr abstract_z = FromValue(static_cast<int64_t>(3), false);
  args_spec_list = {res, embed_x, abstract_z};

  res = engine_->Run(graph_environ_get, args_spec_list).eval_result->abstract();

  ASSERT_TRUE(*res == *abstract_x);
}

/// Feature: Check inference result of primitive by build a FuncGraph with single primitive.
/// Description: Check inference result of primitive EnvironAdd.
/// Expectation: Equal
TEST_F(TestPrim, test_environ_add) {
  FuncGraphPtr graph_embed = MakeFuncGraph(prim::kPrimEmbed, 1);
  AbstractBasePtr abstract_x = FromValue(static_cast<int64_t>(1), false);
  AbstractBasePtrList args_spec_list = {abstract_x};
  AbstractBasePtr embed_x = engine_->Run(graph_embed, args_spec_list).eval_result->abstract();

  FuncGraphPtr graph_environ_set = MakeFuncGraph(prim::kPrimEnvironSet, 3);

  AbstractBasePtr abstract_environ = MakeEnvironAbstract();
  AbstractBasePtr abstract_y = FromValue(static_cast<int64_t>(2), false);
  args_spec_list = {abstract_environ, embed_x, abstract_y};

  AbstractBasePtr abstract_e1 = engine_->Run(graph_environ_set, args_spec_list).eval_result->abstract();
  AbstractBasePtr exp = std::make_shared<AbstractScalar>(kAnyValue, std::make_shared<EnvType>());
  ASSERT_TRUE(*abstract_e1 == *exp);

  AbstractBasePtr abstract_z = FromValue(static_cast<int64_t>(3), false);
  args_spec_list = {abstract_environ, embed_x, abstract_z};

  AbstractBasePtr abstract_e2 = engine_->Run(graph_environ_set, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*abstract_e2 == *exp);

  FuncGraphPtr graph_add = MakeFuncGraph(prim::kPrimEnvironAdd, 2);
  args_spec_list = {abstract_e1, abstract_e2};
  AbstractBasePtr res = engine_->Run(graph_add, args_spec_list).eval_result->abstract();

  ASSERT_TRUE(*res == *exp);
}

TEST_F(TestPrim, test_relu) {
  PrimitivePtr relu = prim::kPrimReLU;
  relu->AddAttr("T", MakeValue(static_cast<int64_t>(kNumberTypeFloat64)));
  FuncGraphPtr func_graph = MakeFuncGraph(relu, 1);

  AbstractBasePtr expected = UTPrimUtils::ArrayFloat64Of({2, 2, 2, 3});  // NCHW
  AbstractBasePtrList args_spec_list = {expected};

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *expected);
}

/*
TEST_F(TestPrim, test_relu2) {
  FuncGraphPtr func_graph = getPyFun("get_relu");
  ASSERT_TRUE(func_graph != nullptr);

  auto arr = ArrayOfTensor(UTPrimUtils::kF32, {3, 4, 5});
  auto expected = ArrayOfTensor(UTPrimUtils::kF32, {3, 4, 5});

  AbstractBasePtrList args_spec_list = {arr};
  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  auto res = dyn_cast<AbstractTensor>(ret);
  ASSERT_TRUE(*(res->GetShapeTrack()) == *(expected->GetShapeTrack()));
}

TEST_F(TestPrim, test_conv2d1) {
  std::shared_ptr<py::scoped_interpreter> env = python_adapter::set_python_scoped();
  py::tuple kernel_size(2);
  kernel_size[0] = 5;
  kernel_size[1] = 5;
  std::shared_ptr<FuncGraph> func_graph = getPyFun.CallAndParseRet("test_conv2d", 64, kernel_size, 0, 2, 1);

  // NCHW
  std::vector<int64_t> inputs_dims = {2, 20, 32, 32};
  std::vector<int64_t> weight_dims = {64, 20, 5, 5};

  tensor::TensorPtr inputs = std::make_shared<tensor::Tensor>();
  inputs->set_data_type(kNumberTypeInt32);
  inputs->set_shape(inputs_dims);
  // Cout, Cin, kernel_size
  tensor::TensorPtr weight = std::make_shared<tensor::Tensor>();
  weight->set_data_type(kNumberTypeInt32);
  weight->set_shape(weight_dims);

  AbstractBasePtr abstract_inputs = FromValue(inputs, true);
  AbstractBasePtr abstract_weight = FromValue(weight, true);
  AbstractBasePtrList args_spec_list = {abstract_inputs, abstract_weight};

  AbstractBasePtr expected = abstract_inputs->Clone();
  // NCHW
  std::vector<int64_t> shape = {2, 64, 14, 14};
  expected->set_shape(std::make_shared<Shape>(shape));

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();

  auto res_ptr = dyn_cast<AbstractTensor>(res);
  auto expected_ptr = dyn_cast<AbstractTensor>(expected);
  ASSERT_TRUE(*res_ptr->shape() == *expected_ptr->shape());
  ASSERT_TRUE(*res_ptr->element() == *expected_ptr->element());
}

TEST_F(TestPrim, test_conv2d) {
  FuncGraphPtr func_graph = getPyFun("get_conv2d");
  ASSERT_TRUE(func_graph != nullptr);

  auto input = ArrayOfTensor(UTPrimUtils::kF32, {10, 32, 32, 32});
  auto weight = ArrayOfTensor(UTPrimUtils::kF32, {64, 32, 3, 3});

  AbstractBasePtrList args_spec_list = {input, weight};
  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  auto res = dyn_cast<AbstractTensor>(ret);
  auto expected = ArrayOfTensor(UTPrimUtils::kF32, {10, 64, 16, 16});
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();
  ASSERT_TRUE(*(res->GetShapeTrack()) == *(expected->GetShapeTrack()));
}

TEST_F(TestPrim, test_conv2d_native) {
  FuncGraphPtr func_graph = getPyFun("get_conv2d_native");
  ASSERT_TRUE(func_graph != nullptr);

  auto input = ArrayOfTensor(UTPrimUtils::kF64, {10, 32, 32, 32});
  auto weight = ArrayOfTensor(UTPrimUtils::kF64, {3, 32, 3, 3});

  AbstractBasePtrList args_spec_list = {input, weight};
  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  auto res = dyn_cast<AbstractTensor>(ret);
  auto expected = ArrayOfTensor(UTPrimUtils::kF64, {10, 96, 16, 16});
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();
  ASSERT_TRUE(*(res->GetShapeTrack()) == *(expected->GetShapeTrack()));
}

TEST_F(TestPrim, test_biasAdd) {
  FuncGraphPtr func_graph = getPyFun("get_bias_add");
  ASSERT_TRUE(func_graph != nullptr);

  auto value = ArrayOfTensor(UTPrimUtils::kF32, {10, 32, 32, 32});
  auto bias = ArrayOfTensor(UTPrimUtils::kF32, {32});

  AbstractBasePtrList args_spec_list = {value, bias};
  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  auto res = dyn_cast<AbstractTensor>(ret);
  auto expected = ArrayOfTensor(UTPrimUtils::kF32, {10, 32, 32, 32});
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();
  ASSERT_TRUE(*(res->GetShapeTrack()) == *(expected->GetShapeTrack()));
}

TEST_F(TestPrim, test_softmax_cross_entropy_with_logits) {
  FuncGraphPtr func_graph = getPyFun("get_softmax_cross_entropy_with_logits");
  ASSERT_TRUE(func_graph != nullptr);

  auto logits = ArrayOfTensor(UTPrimUtils::kF32, {64, 10});
  auto labels = ArrayOfTensor(UTPrimUtils::kF32, {64, 10});

  AbstractBasePtrList args_spec_list = {logits, labels};
  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_NE(ret, nullptr);
  auto res = dyn_cast<AbstractTuple>(ret);
  auto loss = ArrayOfTensor(UTPrimUtils::kF32, {64});
  auto dLogits = ArrayOfTensor(UTPrimUtils::kF32, {64, 10});
  AbstractBasePtrList expected_list = {loss, dLogits};
  auto expected = std::make_shared<AbstractTuple>(expected_list);
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();

  auto res_ptr0 = dyn_cast<AbstractTuple>(res);
  auto expected_ptr0 = dyn_cast<AbstractTuple>(expected);

  ASSERT_GT((*res_ptr0).size(), 1);
  auto res_ptr = dyn_cast<AbstractTensor>((*res_ptr0)[1]);
  ASSERT_GT((*expected_ptr0).size(), 1);
  auto expected_ptr = dyn_cast<AbstractTensor>((*expected_ptr0)[1]);
  ASSERT_TRUE(*res_ptr->shape() == *expected_ptr->shape());
  ASSERT_TRUE(*res_ptr->element() == *expected_ptr->element());
}

TEST_F(TestPrim, test_tensor_to_scalar_prim) {
  FuncGraphPtr func_graph = getPyFun("get_tensor_to_scalar");
  ASSERT_TRUE(func_graph != nullptr);

  auto logits = ArrayOfTensor(UTPrimUtils::kF64, {64, 10});
  auto labels = ArrayOfTensor(UTPrimUtils::kF64, {64, 10});

  AbstractBasePtrList args_spec_list = {logits, labels};
  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  auto res = dyn_cast<AbstractScalar>(ret);
  AbstractScalarPtr expected = std::make_shared<AbstractScalar>(kAnyValue, kFloat64);
  expected->set_type(UTPrimUtils::kF64);
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_pooling) {
  PrimitivePtr pooling = prim::kPrimPooling;
  pooling->AddAttr("mode", MakeValue(std::string("avg")));
  pooling->AddAttr("pad_mode", MakeValue(std::string("valid")));
  pooling->AddAttr("nan_opt", MakeValue(0));
  pooling->AddAttr("window", MakeValue(2));
  pooling->AddAttr("pad", MakeValue(1));
  pooling->AddAttr("stride", MakeValue(1));
  pooling->AddAttr("data_mode", MakeValue(1));
  pooling->AddAttr("ceil_mode", MakeValue(0));
  FuncGraphPtr func_graph = MakeFuncGraph(pooling, 1);

  std::vector<int64_t> inputs_dims = {8, 64, 3, 3};
  auto inputs = std::make_shared<tensor::Tensor>();
  inputs->set_data_type(kNumberTypeFloat32);
  inputs->set_shape(inputs_dims);
  AbstractBasePtr abstract_input = FromValue(inputs, false);
  AbstractBasePtrList args_spec_list = {abstract_input};
  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();

  AbstractBasePtr expected = abstract_input->Clone()->Broaden();
  std::vector<int64_t> expected_dims = {8, 64, 2, 2};
  expected->set_shape(std::make_shared<Shape>(expected_dims));
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_hastype) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 1;
  TypePtr v2 = std::make_shared<Number>();

  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  AbstractTypePtr abstract_v2 = UTPrimUtils::TypeToAbstract(v2);
  AbstractBasePtr expected = FromValue(true, false);

  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);

  auto prim = std::make_shared<Primitive>("hastype");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 2);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_array_len) {
  AbstractBasePtrList args_spec_list;
  auto v1 = UTPrimUtils::ArrayFloat64Of({3, 4, 0, 2});
  auto expected = std::make_shared<AbstractScalar>(kAnyValue, kInt32);

  args_spec_list.push_back(v1);

  auto prim = std::make_shared<Primitive>("array_len");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 1);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_list_len) {
  AbstractBasePtrList args_spec_list;
  auto v1 = UTPrimUtils::ListShapeOf({3, 4, 0, 2});
  auto expected = std::make_shared<AbstractScalar>(4);

  args_spec_list.push_back(v1);

  auto prim = std::make_shared<Primitive>("list_len");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 1);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_tuple_len) {
  AbstractBasePtrList args_spec_list;
  auto v1 = UTPrimUtils::ShapeOf({3, 4, 0, 2});
  auto expected = std::make_shared<AbstractScalar>(4);

  args_spec_list.push_back(v1);

  auto prim = std::make_shared<Primitive>("tuple_len");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 1);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_tuple_reversed) {
  AbstractBasePtrList args_spec_list;
  auto v1 = UTPrimUtils::ShapeOf({0, 1, 2, 3});
  auto expected = UTPrimUtils::ShapeOf({3, 2, 1, 0});

  args_spec_list.push_back(v1);

  auto prim = std::make_shared<Primitive>("tuple_reversed");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 1);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "expect=" << expected->ToString();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_list_getitem) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 2;
  int64_t v2 = 1;

  AbstractBasePtr elem = FromValue(v1, false);
  AbstractBasePtr elem2 = FromValue(v2, false);
  AbstractBasePtrList elems = {elem, elem};
  auto abstract_v1 = std::make_shared<AbstractList>(elems);
  AbstractBasePtr abstract_v2 = FromValue(v2, false);

  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);

  auto prim = std::make_shared<Primitive>("ListGetItem");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 2);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *elem);
}

TEST_F(TestPrim, test_list_setitem) {
  int64_t v1 = 1;
  int64_t v2 = 2;

  AbstractBasePtr elem1 = FromValue(v1, false);
  AbstractBasePtr elem2 = FromValue(v2, false);
  AbstractBasePtrList elems = {elem1, elem1};
  auto abstract_tuple = std::make_shared<AbstractList>(elems);
  AbstractBasePtr abstract_v2 = FromValue(v1, false);
  AbstractBasePtr abstract_v3 = FromValue(v2, false);
  AbstractBasePtrList args_spec_list = {abstract_tuple, abstract_v2, abstract_v3};

  auto prim = std::make_shared<Primitive>("list_setitem");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 3);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "result: " << res->ToString();
  AbstractBasePtrList elems_exp = {elem1, elem2};
  auto expected = std::make_shared<AbstractList>(elems_exp);
  MS_LOG(INFO) << "expected: " << expected->ToString();

  auto res_list = dyn_cast<AbstractList>(res);
  ASSERT_TRUE(*expected == *res_list);
}

TEST_F(TestPrim, test_list_append) {
  int64_t v1 = 1;

  AbstractBasePtr elem1 = FromValue(v1, false);
  AbstractBasePtr elem2 = FromValue(v1, false);
  auto abstract_tuple = std::make_shared<AbstractList>(AbstractBasePtrList({elem1, elem2}));
  AbstractBasePtr abstract_v2 = FromValue(v1, false);
  AbstractBasePtrList args_spec_list = {abstract_tuple, abstract_v2};

  auto prim = std::make_shared<Primitive>("list_append");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 2);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "result: " << res->ToString();
  auto expected = std::make_shared<AbstractList>(AbstractBasePtrList({elem1, elem2}));
  MS_LOG(INFO) << "expected: " << expected->ToString();

  auto res_list = dyn_cast<AbstractList>(res);
  ASSERT_TRUE(*res_list == *expected);
}

TEST_F(TestPrim, test_tuple_setitem) {
  int64_t v1 = 1;
  int64_t v2 = 2;

  AbstractBasePtr elem1 = FromValue(v1, false);
  AbstractBasePtr elem2 = FromValue(v2, false);
  AbstractBasePtrList elems = {elem1, elem1};
  auto abstract_tuple = std::make_shared<AbstractTuple>(elems);
  AbstractBasePtr abstract_v2 = FromValue(v1, false);
  AbstractBasePtr abstract_v3 = FromValue(v2, false);
  AbstractBasePtrList args_spec_list = {abstract_tuple, abstract_v2, abstract_v3};

  auto prim = std::make_shared<Primitive>("tuple_setitem");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 3);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "result: " << res->ToString();
  AbstractBasePtrList elems_exp = {elem1, elem2};
  auto expected = std::make_shared<AbstractTuple>(elems_exp);
  MS_LOG(INFO) << "expected: " << expected->ToString();

  auto res_tuple = dyn_cast<AbstractTuple>(res);
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_make_list) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 2;
  int64_t v2 = 2;

  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  AbstractBasePtr abstract_v2 = FromValue(v2, false);

  auto expected = std::make_shared<AbstractList>(AbstractBasePtrList({abstract_v1, abstract_v2}));

  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);

  auto prim = std::make_shared<Primitive>("make_list");
  FuncGraphPtr func_graph = MakeFuncGraph(prim, 2);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_make_range) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 1;
  int64_t v2 = 4;

  AbstractBasePtr abstract_v1 = FromValue(v1);
  AbstractBasePtr abstract_v2 = FromValue(v2);
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);

  auto prim = std::make_shared<Primitive>("make_range");
  std::shared_ptr<FuncGraph> func_graph = MakeFuncGraph(prim, 2);

  AbstractBasePtr ele1 = FromValue(1);
  AbstractBasePtr ele2 = FromValue(2);
  AbstractBasePtr ele3 = FromValue(3);
  AbstractBasePtrList elem_list({ele1, ele2, ele3});
  AbstractBasePtr expected = std::make_shared<AbstractTuple>(elem_list);

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "res=" << res->ToString();
  MS_LOG(INFO) << "expected=" << expected->ToString();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_layernorm) {
  PrimitivePtr layerNorm = prim::kPrimLayerNorm;
  layerNorm->AddAttr("begin_norm_axis", MakeValue(1));
  layerNorm->AddAttr("begin_params_axis", MakeValue(1));

  std::shared_ptr<FuncGraph> func_graph = MakeFuncGraph(layerNorm, 3);

  std::vector<int64_t> inputs_dims = {128, 64, 32, 64};
  std::vector<int64_t> mean_var_dims = {128, 64, 32, 1};
  std::vector<int64_t> params_dims = {64, 32, 64};

  tensor::TensorPtr inputs = std::make_shared<tensor::Tensor>();
  inputs->set_data_type(kNumberTypeFloat32);
  inputs->set_shape(inputs_dims);

  tensor::TensorPtr mean_var = std::make_shared<tensor::Tensor>();
  mean_var->set_data_type(kNumberTypeFloat32);
  mean_var->set_shape(mean_var_dims);

  tensor::TensorPtr gamma = std::make_shared<tensor::Tensor>();
  gamma->set_data_type(kNumberTypeFloat32);
  gamma->set_shape(params_dims);

  tensor::TensorPtr beta = std::make_shared<tensor::Tensor>();
  beta->set_data_type(kNumberTypeFloat32);
  beta->set_shape(params_dims);

  AbstractBasePtr abstract_inputs = FromValue(inputs, true);
  AbstractBasePtr abstract_mean_var = FromValue(mean_var, true);
  AbstractBasePtr abstract_gamma = FromValue(gamma, true);
  AbstractBasePtr abstract_beta = FromValue(beta, true);
  AbstractBasePtrList args_spec_list = {abstract_inputs, abstract_gamma, abstract_beta};

  AbstractBasePtr expected0 = abstract_inputs->Clone();
  AbstractBasePtr expected1 = abstract_mean_var->Clone();
  AbstractBasePtr expected2 = abstract_mean_var->Clone();

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected0: " << expected0->ToString();
  MS_LOG(INFO) << "expected1: " << expected1->ToString();
  MS_LOG(INFO) << "expected2: " << expected2->ToString();

  std::shared_ptr<AbstractTuple> abs_tuple = dyn_cast<AbstractTuple>(res);
  ASSERT_TRUE(abs_tuple != nullptr);

  auto res_ptr0 = dyn_cast<AbstractTensor>(abs_tuple->elements()[0]);
  auto expected_ptr0 = dyn_cast<AbstractTensor>(expected0);
  ASSERT_TRUE(*res_ptr0->shape() == *expected_ptr0->shape());
  ASSERT_TRUE(*res_ptr0->element() == *expected_ptr0->element());

  auto res_ptr1 = dyn_cast<AbstractTensor>(abs_tuple->elements()[1]);
  auto expected_ptr1 = dyn_cast<AbstractTensor>(expected1);
  ASSERT_TRUE(*res_ptr1->shape() == *expected_ptr1->shape());
  ASSERT_TRUE(*res_ptr1->element() == *expected_ptr1->element());

  auto res_ptr2 = dyn_cast<AbstractTensor>(abs_tuple->elements()[2]);
  auto expected_ptr2 = dyn_cast<AbstractTensor>(expected2);
  ASSERT_TRUE(*res_ptr2->shape() == *expected_ptr2->shape());
  ASSERT_TRUE(*res_ptr2->element() == *expected_ptr2->element());
}

TEST_F(TestPrim, test_DropoutGenMask) {
  AbstractBasePtrList args_spec_list;

  auto arg0 = UTPrimUtils::ShapeOf({5, 5, 5, 5});

  std::vector<int64_t> keep_prob_shape = {};
  tensor::TensorPtr keep_prob = std::make_shared<tensor::Tensor>(0.5f);
  keep_prob->set_data_type(kNumberTypeFloat32);
  keep_prob->set_shape(keep_prob_shape);
  AbstractBasePtr abstract_keep_prob = FromValue(keep_prob);

  auto prim = std::make_shared<Primitive>("DropoutGenMask");
  std::shared_ptr<FuncGraph> func_graph = MakeFuncGraph(prim, 2);

  args_spec_list.push_back(arg0);
  args_spec_list.push_back(abstract_keep_prob);

  // should return a tensor with on dimension of 79 elements
  AbstractBasePtr expected = std::make_shared<AbstractTensor>(std::make_shared<AbstractScalar>(kAnyValue, kUInt8),
                                                              std::make_shared<Shape>(std::vector<int64_t>{79}));

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "res=" << res->ToString();
  MS_LOG(INFO) << "expected=" << expected->ToString();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_dropout) {
  std::shared_ptr<py::scoped_interpreter> env = python_adapter::set_python_scoped();
  std::shared_ptr<FuncGraph> func_graph = getPyFun.CallAndParseRet("test_dropout");

  std::vector<int64_t> inputs_dims = {2, 20, 32, 32};

  tensor::TensorPtr inputs = std::make_shared<tensor::Tensor>();
  inputs->set_data_type(kNumberTypeFloat32);
  inputs->set_shape(inputs_dims);

  AbstractBasePtr abstract_inputs = FromValue(inputs, true);
  std::vector<int64_t> keep_prob_shape = {};
  tensor::TensorPtr keep_prob = std::make_shared<tensor::Tensor>(0.5f);
  keep_prob->set_data_type(kNumberTypeFloat32);
  keep_prob->set_shape(keep_prob_shape);
  AbstractBasePtr abstract_keep_prob = FromValue(keep_prob);

  AbstractBasePtrList args_spec_list = {abstract_inputs, abstract_keep_prob};
  AbstractBasePtr expected = abstract_inputs->Clone();

  // NCHW
  std::vector<int64_t> shape = {2, 20, 32, 32};
  expected->set_shape(std::make_shared<Shape>(shape));

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();

  auto res_ptr = dyn_cast<AbstractTensor>(res);
  auto expected_ptr = dyn_cast<AbstractTensor>(expected);
  ASSERT_TRUE(*res_ptr->shape() == *expected_ptr->shape());
  ASSERT_TRUE(*res_ptr->element() == *expected_ptr->element());
}

TEST_F(TestPrim, test_BroadcastGradientArgs_01_dim) {
  PrimitivePtr broadcatGradientArgs = prim::kPrimBroadcastGradientArgs;
  std::shared_ptr<FuncGraph> func_graph = MakeFuncGraph(broadcatGradientArgs, 2);

  // broadcast shape: x: 8,5,3, y:3
  // output: ((),(0, 1))
  AbstractBasePtrList x_arg_list({abstract::FromValue(8), abstract::FromValue(5), abstract::FromValue(3)});
  AbstractBasePtrList y_arg_list({abstract::FromValue(3)});
  auto x_input = std::make_shared<AbstractTuple>(x_arg_list);
  auto y_input = std::make_shared<AbstractTuple>(y_arg_list);
  AbstractBasePtrList args_spec_list = {x_input, y_input};
  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  auto res = dyn_cast<AbstractTuple>(ret);
  AbstractBasePtrList x_idx_list;
  auto r_x = std::make_shared<AbstractTuple>(x_idx_list);
  AbstractBasePtrList y_idx_list({abstract::FromValue(0), abstract::FromValue(1)});
  auto r_y = std::make_shared<AbstractTuple>(y_idx_list);
  AbstractBasePtrList elem_list({r_x, r_y});
  auto expected = std::make_shared<AbstractTuple>(elem_list);
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_BroadcastGradientArgs_1_dim) {
  PrimitivePtr broadcatGradientArgs = prim::kPrimBroadcastGradientArgs;
  std::shared_ptr<FuncGraph> func_graph = MakeFuncGraph(broadcatGradientArgs, 2);

  // broadcast shape: x: 8,1,3, y:8 5 3
  // output: ((1),())
  AbstractBasePtrList x_arg_list({abstract::FromValue(8), abstract::FromValue(1), abstract::FromValue(3)});
  AbstractBasePtrList y_arg_list({abstract::FromValue(8), abstract::FromValue(5), abstract::FromValue(3)});
  auto x_input = std::make_shared<AbstractTuple>(x_arg_list);
  auto y_input = std::make_shared<AbstractTuple>(y_arg_list);
  AbstractBasePtrList args_spec_list = {x_input, y_input};
  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  auto res = dyn_cast<AbstractTuple>(ret);
  AbstractBasePtrList x_idx_list({abstract::FromValue(1)});
  auto r_x = std::make_shared<AbstractTuple>(x_idx_list);
  AbstractBasePtrList y_idx_list;
  auto r_y = std::make_shared<AbstractTuple>(y_idx_list);
  AbstractBasePtrList elem_list({r_x, r_y});
  auto expected = std::make_shared<AbstractTuple>(elem_list);
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();
  ASSERT_TRUE(*res == *expected);
}

TEST_F(TestPrim, test_DictGetItem) {
  PrimitivePtr dictGetItem = prim::kPrimDictGetItem;
  std::shared_ptr<FuncGraph> func_graph = MakeFuncGraph(dictGetItem, 2);

  std::vector<std::pair<std::string, ValuePtr>> tensor_map = {
    {"x", std::make_shared<tensor::Tensor>(kNumberTypeInt32, std::vector<int64_t>{2, 3, 4})},
    {"y", std::make_shared<tensor::Tensor>(kNumberTypeInt32, std::vector<int64_t>{2, 1, 4})}};
  ValueDictionary value_dict(tensor_map);
  AbstractBasePtr array_dict = value_dict.ToAbstract();
  AbstractBasePtr key = abstract::FromValue("x");
  AbstractBasePtrList args_spec_list = {array_dict, key};

  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  AbstractTensorPtr tensor_ret = dyn_cast<AbstractTensor>(ret);
  AbstractTensorPtr expect = dyn_cast<AbstractTensor>(FromValue(tensor_map[0].second));

  ASSERT_TRUE(*tensor_ret == *expect);
}

TEST_F(TestPrim, test_DictGetItem2) {
  PrimitivePtr dictGetItem = prim::kPrimDictGetItem;
  std::shared_ptr<FuncGraph> func_graph = MakeFuncGraph(dictGetItem, 2);

  AbstractBasePtr arr_x = ArrayOfTensor(UTPrimUtils::kF64, {3, 4, 5});
  AbstractBasePtr arr_y = ArrayOfTensor(UTPrimUtils::kF64, {1, 4, 5});
  AbstractBasePtr arr_z = ArrayOfTensor(UTPrimUtils::kF64, {3, 1, 5});
  std::vector<AbstractElementPair> array_map = {{"x", arr_x}, {"y", arr_y}, {"z", arr_z}};
  AbstractDictionaryPtr array_dict = std::make_shared<AbstractDictionary>(array_map);
  AbstractBasePtr key = abstract::FromValue("x");
  AbstractBasePtrList args_spec_list = {array_dict, key};

  AbstractBasePtr ret = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  AbstractTensorPtr tensor_ret = dyn_cast<AbstractTensor>(ret);
  AbstractTensorPtr expect = dyn_cast<AbstractTensor>(arr_x);

  ASSERT_TRUE(*tensor_ret == *expect);
}
*/

}  // namespace abstract
}  // namespace mindspore
