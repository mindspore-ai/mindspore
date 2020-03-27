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

#include "pipeline/static_analysis/evaluator.h"
#include "pipeline/static_analysis/prim.h"

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "pipeline/static_analysis/helper.h"

#include "debug/draw.h"

namespace mindspore {
namespace abstract {
namespace python_adapter = mindspore::parse::python_adapter;

class TestEvaluatorCacheMap : public UT::Common {
 public:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestEvaluatorCacheMap, test_evaluator_cache_map) {
  EvaluatorCacheMap cache;

  AbstractBasePtr abstract_v1 = FromValue(1, false);
  AbstractBasePtr abstract_v2 = FromValue(2, false);
  AbstractBasePtrList args_spec_list = {abstract_v1, abstract_v2};
  AbstractBasePtr abstract_val = FromValue(10, false);
  cache[args_spec_list] = abstract_val;

  auto iter = cache.find(args_spec_list);
  ASSERT_TRUE(iter != cache.end());
  ASSERT_TRUE(iter->second == abstract_val);

  AbstractBasePtr abstract_v1_variant1 = FromValue(1, false);
  AbstractBasePtr abstract_v2_variant1 = FromValue(2, false);
  AbstractBasePtrList args_spec_list_variant1 = {abstract_v1_variant1, abstract_v2_variant1};

  iter = cache.find(args_spec_list_variant1);
  ASSERT_TRUE(iter != cache.end());
  ASSERT_TRUE(iter->second == abstract_val);

  AbstractBasePtr abstract_v1_variant2 = FromValue(1, false);
  AbstractBasePtr abstract_v2_variant2 = FromValue(3, false);
  AbstractBasePtrList args_spec_list_variant2 = {abstract_v1_variant2, abstract_v2_variant2};

  iter = cache.find(args_spec_list_variant2);
  ASSERT_TRUE(iter == cache.end());
}

class TestStandardEvaluator : public UT::Common {
 public:
  TestStandardEvaluator() : getPyFun("gtest_input.pipeline.infer.infer_test", true), engine_(nullptr) {}
  void SetUp();
  void TearDown();

  UT::PyFuncGraphFetcher getPyFun;
  AnalysisEnginePtr engine_;
};

void TestStandardEvaluator::SetUp() { engine_ = SetupAnalysisEngine(); }

void TestStandardEvaluator::TearDown() {
  // destroy resource
}

TEST_F(TestStandardEvaluator, test_multiple_conv2d) {
  std::shared_ptr<py::scoped_interpreter> env = python_adapter::set_python_scoped();
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("test_multiple_conv2d");

  // NCHW
  std::vector<int> inputs_dims = {2, 20, 32, 32};
  std::vector<int> weight1_dims = {2, 20, 5, 5};
  std::vector<int> weight2_dims = {2, 2, 5, 5};

  tensor::TensorPtr inputs = std::make_shared<tensor::Tensor>();
  inputs->set_data_type(kNumberTypeInt32);
  inputs->set_shape(inputs_dims);
  // Cout, Cin, kernel_size
  tensor::TensorPtr weight1 = std::make_shared<tensor::Tensor>();
  weight1->set_data_type(kNumberTypeInt32);
  weight1->set_shape(weight1_dims);
  // Cout, Cin, kernel_size
  tensor::TensorPtr weight2 = std::make_shared<tensor::Tensor>();
  weight2->set_data_type(kNumberTypeInt32);
  weight2->set_shape(weight2_dims);

  AbstractBasePtr abstract_inputs = FromValue(inputs, true);
  AbstractBasePtr abstract_weight1 = FromValue(weight1, true);
  AbstractBasePtr abstract_weight2 = FromValue(weight2, true);
  AbstractBasePtrList args_spec_list = {abstract_inputs, abstract_weight1, abstract_weight2};

  AbstractBasePtr expected = abstract_inputs->Clone();
  // NCHW
  std::vector<int> shape = {2, 2, 6, 6};
  expected->set_shape(std::make_shared<Shape>(shape));

  AbstractBasePtr res = engine_->Run(func_graph, args_spec_list).inferred;
  MS_LOG(INFO) << "result: " << res->ToString();
  MS_LOG(INFO) << "expected: " << expected->ToString();

  auto res_ptr = dyn_cast<AbstractTensor>(res);
  auto expected_ptr = dyn_cast<AbstractTensor>(expected);
  ASSERT_TRUE(*res_ptr->shape() == *expected_ptr->shape());
  ASSERT_TRUE(*res_ptr->element() == *expected_ptr->element());
}

class TestPartialEvaluator : public UT::Common {
 public:
  TestPartialEvaluator() : getPyFun("gtest_input.pipeline.infer.infer_test", true), engine_(nullptr) {}
  void SetUp() { engine_ = SetupAnalysisEngine(); }
  void TearDown() {}

 public:
  UT::PyFuncGraphFetcher getPyFun;
  AnalysisEnginePtr engine_;
};

TEST_F(TestPartialEvaluator, test_infer_dataclass_resolved) {
  getPyFun.SetDoResolve(true);
  FuncGraphPtr func_graph = getPyFun("test_dataclass_fun_sub");
  ASSERT_TRUE(nullptr != func_graph);
  draw::Draw("test_dataclass_fun_sub.dot", func_graph);

  AbstractBasePtrList args_spec_list;
  float x = 5.1;

  AbstractBasePtr abstract_x = FromValue(x, false);
  args_spec_list.push_back(abstract_x);

  AbstractBasePtr abs_base_got = engine_->Run(func_graph, args_spec_list).inferred;
  ASSERT_TRUE(*(abs_base_got->GetTypeTrack()) == *(abstract_x->GetTypeTrack()));
  ASSERT_TRUE(abs_base_got->GetTypeTrack()->type_id() == kNumberTypeFloat32);
}

TEST_F(TestPartialEvaluator, test_infer_dataclass_unresolved) {
  getPyFun.SetDoResolve(false);
  FuncGraphPtr func_graph = getPyFun("test_dataclass_fun_add");
  ASSERT_TRUE(nullptr != func_graph);

  AbstractBasePtrList args_spec_list;
  float x = 5.2;

  AbstractBasePtr abstract_x = FromValue(x, false);
  args_spec_list.push_back(abstract_x);

  AbstractBasePtr abs_base_got = engine_->Run(func_graph, args_spec_list).inferred;
  ASSERT_TRUE(*(abs_base_got->GetTypeTrack()) == *(abstract_x->GetTypeTrack()));
  ASSERT_TRUE(abs_base_got->GetTypeTrack()->type_id() == kNumberTypeFloat32);
}

TEST_F(TestPartialEvaluator, test_infer_add_resolved) {
  getPyFun.SetDoResolve(true);
  FuncGraphPtr func_graph = getPyFun("test_fun_add");
  ASSERT_TRUE(nullptr != func_graph);

  AbstractBasePtrList args_spec_list;
  double x = 5.2;
  double y = 3.2;

  AbstractBasePtr abstract_x = FromValue(x, false);
  AbstractBasePtr abstract_y = FromValue(y, false);
  args_spec_list.push_back(abstract_x);
  args_spec_list.push_back(abstract_y);

  AbstractBasePtr abs_base_got = engine_->Run(func_graph, args_spec_list).inferred;
  ASSERT_TRUE(*(abs_base_got->GetTypeTrack()) == *(abstract_x->GetTypeTrack()));
  ASSERT_TRUE(abs_base_got->GetTypeTrack()->type_id() == kNumberTypeFloat64);
}

TEST_F(TestPartialEvaluator, test_infer_sub_unresolved) {
  getPyFun.SetDoResolve(false);
  FuncGraphPtr func_graph = getPyFun("test_fun_sub");
  ASSERT_TRUE(nullptr != func_graph);

  AbstractBasePtrList args_spec_list;
  double x = 5.1;
  double y = 3.1;

  AbstractBasePtr abstract_x = FromValue(x, false);
  AbstractBasePtr abstract_y = FromValue(y, false);
  args_spec_list.push_back(abstract_x);
  args_spec_list.push_back(abstract_y);

  AbstractBasePtr abs_base_got = engine_->Run(func_graph, args_spec_list).inferred;
  ASSERT_TRUE(*(abs_base_got->GetTypeTrack()) == *(abstract_x->GetTypeTrack()));
  ASSERT_TRUE(abs_base_got->GetTypeTrack()->type_id() == kNumberTypeFloat64);
}

TEST_F(TestPartialEvaluator, test_infer_net_construct_add_resolved) {
  getPyFun.SetDoResolve(true);
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("test_net_construct_add");
  ASSERT_TRUE(nullptr != func_graph);

  AbstractBasePtrList args_spec_list;
  double x = 1.2;
  double y = 2.2;

  AbstractBasePtr abstract_x = FromValue(x, false);
  AbstractBasePtr abstract_y = FromValue(y, false);
  args_spec_list.push_back(abstract_x);
  args_spec_list.push_back(abstract_y);

  AbstractBasePtr abs_base_got = engine_->Run(func_graph, args_spec_list).inferred;
  ASSERT_TRUE(*(abs_base_got->GetTypeTrack()) == *(abstract_x->GetTypeTrack()));
  ASSERT_TRUE(abs_base_got->GetTypeTrack()->type_id() == kNumberTypeFloat64);
}

TEST_F(TestPartialEvaluator, test_infer_construct_sub_unresolved) {
  getPyFun.SetDoResolve(false);
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("test_net_construct_sub");
  ASSERT_TRUE(nullptr != func_graph);
  draw::Draw("test_infer_simple_net.dot", func_graph);

  AbstractBasePtrList args_spec_list;
  double x = 1.2;
  double y = 2.2;

  AbstractBasePtr abstract_x = FromValue(x, false);
  AbstractBasePtr abstract_y = FromValue(y, false);
  args_spec_list.push_back(abstract_x);
  args_spec_list.push_back(abstract_y);

  AbstractBasePtr abs_base_got = engine_->Run(func_graph, args_spec_list).inferred;
  ASSERT_TRUE(*(abs_base_got->GetTypeTrack()) == *(abstract_x->GetTypeTrack()));
  ASSERT_TRUE(abs_base_got->GetTypeTrack()->type_id() == kNumberTypeFloat64);
}
}  // namespace abstract
}  // namespace mindspore
