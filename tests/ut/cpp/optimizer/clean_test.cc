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
#include "debug/draw.h"
#include "frontend/optimizer/clean.h"

namespace mindspore {
namespace opt {
using mindspore::abstract::AbstractAttribute;
using mindspore::abstract::AbstractClass;
using mindspore::abstract::AbstractError;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;

class TestClean : public UT::Common {
 public:
  TestClean() : getPyFun("gtest_input.optimizer.clean_test", true) {}
  virtual void SetUp();
  virtual void TearDown();

 public:
  UT::PyFuncGraphFetcher getPyFun;
  FuncGraphPtr me_graph;
};

void TestClean::SetUp() {
  // build the func_graph.
  me_graph = std::make_shared<FuncGraph>();
  me_graph->debug_info()->set_name("next");

  // build the nodes
  AnfNodePtr valuenode_next = NewValueNode(std::string("ms_next"));
  ParameterPtr parameter = std::make_shared<Parameter>(me_graph);
  AbstractBasePtr para_scalar = std::make_shared<AbstractScalar>(static_cast<int64_t>(0));
  AbstractBasePtr para_list = std::make_shared<AbstractList>(
    AbstractBasePtrList({std::make_shared<AbstractScalar>(kFloat64), std::make_shared<AbstractScalar>(kFloat64)}));
  AbstractBasePtrList para_elem{para_scalar, para_list};
  AbstractBasePtr para_tuple = std::make_shared<AbstractTuple>(para_elem);
  parameter->set_abstract(para_tuple);

  AbstractBasePtr app_float = std::make_shared<AbstractScalar>(kFloat64);
  AbstractBasePtr app_int = std::make_shared<AbstractScalar>(kFloat64);
  AbstractBasePtr app_list = std::make_shared<AbstractList>(
    AbstractBasePtrList({std::make_shared<AbstractScalar>(kFloat64), std::make_shared<AbstractScalar>(kFloat64)}));
  AbstractBasePtr app_tuple_inner = std::make_shared<AbstractTuple>(AbstractBasePtrList{app_int, app_list});
  AbstractBasePtr app_tuple = std::make_shared<AbstractTuple>(AbstractBasePtrList{app_float, app_tuple_inner});
  AnfNodePtr cnode_57 = me_graph->NewCNode({valuenode_next, parameter});
  cnode_57->set_abstract(app_tuple);

  AnfNodePtr cnode_67 = me_graph->NewCNode({NewValueNode(prim::kPrimPartial), valuenode_next, parameter});
  cnode_67->set_abstract(app_tuple);

  AnfNodePtr cnode_66 = me_graph->NewCNode({NewValueNode(prim::kPrimScalarAdd), cnode_57, cnode_67});
  cnode_66->set_abstract(app_float);

  AnfNodePtr valuenode_return = NewValueNode(prim::kPrimReturn);
  CNodePtr cnode_55 = me_graph->NewCNode({valuenode_return, cnode_66});
  cnode_55->set_abstract(app_tuple);

  me_graph->set_output(cnode_66);
  me_graph->set_return(cnode_55);
  me_graph->add_parameter(parameter);
}

void TestClean::TearDown() {}

TEST_F(TestClean, TestEraseClassGetAttr) {
  FuncGraphPtr func_graph;

  func_graph = getPyFun("test_erase_class_fn");
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);
  int dataclass_count = 0;

  for (auto node : manager->all_nodes()) {
    if (IsValueNode<parse::ClassObject>(node)) {
      dataclass_count++;
    }
    if (!node->isa<CNode>()) {
      continue;
    }
    auto input0 = node->cast<CNodePtr>()->input(0);
    if (IsValueNode<parse::ClassObject>(input0)) {
      std::vector<AbstractAttribute> attr = {{"x", std::make_shared<AbstractScalar>(kFloat64)},
                                             {"y", std::make_shared<AbstractScalar>(kFloat64)}};
      std::unordered_map<std::string, ValuePtr> methods;
      AbstractBasePtr abs_ptr = std::make_shared<AbstractClass>(Named("Point"), attr, methods);
      node->set_abstract(abs_ptr);
    }
  }

  ASSERT_EQ(dataclass_count, 1);

  SimplifyDataStructures(func_graph, manager);

  int tuple_getitem_count = 0;

  for (auto node : manager->all_nodes()) {
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      tuple_getitem_count++;
    }
  }

  ASSERT_EQ(dataclass_count, 1);
  ASSERT_EQ(tuple_getitem_count, 2);
}

TEST_F(TestClean, TestEraseClassMakeRecord) {
  // build the graph
  auto func_graph = std::make_shared<FuncGraph>();
  func_graph->debug_info()->set_name("test_make_record");

  auto cons_make_record = NewValueNode(prim::kPrimMakeRecord);
  auto para1 = std::make_shared<Parameter>(func_graph);
  auto para2 = std::make_shared<Parameter>(func_graph);

  para1->set_abstract(std::make_shared<AbstractScalar>(kAnyValue, kInt64));
  para2->set_abstract(std::make_shared<AbstractScalar>(kAnyValue, kInt64));
  std::vector<AbstractAttribute> attr = {{"x", std::make_shared<AbstractScalar>(kAnyValue, kInt64)},
                                         {"y", std::make_shared<AbstractScalar>(kAnyValue, kInt64)}};
  std::unordered_map<std::string, ValuePtr> methods;
  AbstractBasePtr abs_ptr = std::make_shared<AbstractClass>(Named("Point"), attr, methods);
  auto cons_class = NewValueNode(abs_ptr->BuildValue());
  cons_class->set_abstract(abs_ptr);

  std::vector<AnfNodePtr> inputs{cons_make_record, cons_class, para1, para2};
  auto apply22 = func_graph->NewCNode(inputs);

  auto cons_return = NewValueNode(prim::kPrimReturn);
  auto apply11 = func_graph->NewCNode({cons_return, apply22});
  apply11->set_abstract(abs_ptr);

  func_graph->set_output(apply22);
  func_graph->set_return(apply11);
  func_graph->add_parameter(para1);
  func_graph->add_parameter(para2);

  auto manager = Manage(func_graph);

  SimplifyDataStructures(func_graph, manager);
}

TEST_F(TestClean, TestEraseClassPartial) {
  // build the graph
  auto func_graph = std::make_shared<FuncGraph>();
  func_graph->debug_info()->set_name("test_partial");

  auto cons_partial = NewValueNode(prim::kPrimPartial);
  auto para1 = std::make_shared<Parameter>(func_graph);
  para1->set_abstract(std::make_shared<AbstractScalar>(kAnyValue, kInt64));

  auto cons_make_record = NewValueNode(prim::kPrimMakeRecord);

  std::vector<AbstractAttribute> attr = {{"x", std::make_shared<AbstractScalar>(kAnyValue, kInt64)},
                                         {"y", std::make_shared<AbstractScalar>(kAnyValue, kInt64)}};
  std::unordered_map<std::string, ValuePtr> methods;
  AbstractBasePtr abs_ptr = std::make_shared<AbstractClass>(Named("Point"), attr, methods);
  auto cons_class = NewValueNode(abs_ptr->BuildValue());
  cons_class->set_abstract(abs_ptr);

  std::vector<AnfNodePtr> inputs{cons_partial, cons_make_record, cons_class, para1};
  auto apply22 = func_graph->NewCNode(inputs);
  std::vector<AnfNodePtr> inputs_nopara{cons_partial, cons_make_record, cons_class};
  auto apply33 = func_graph->NewCNode(inputs_nopara);

  auto apply11 = func_graph->NewCNode({NewValueNode(prim::kPrimScalarAdd), apply22, apply33});

  auto cons_return = NewValueNode(prim::kPrimReturn);
  auto apply00 = func_graph->NewCNode({cons_return, apply11});
  apply00->set_abstract(abs_ptr);

  func_graph->set_output(apply22);
  func_graph->set_return(apply11);
  func_graph->add_parameter(para1);

  auto manager = Manage(func_graph);
  SimplifyDataStructures(func_graph, manager);
}

TEST_F(TestClean, TestEraseTuple) {
  ASSERT_TRUE(nullptr != me_graph);
  std::shared_ptr<FuncGraphManager> manager = Manage(me_graph);

  int abstract_tuple_count = 0;

  for (auto node : manager->all_nodes()) {
    auto dt = node->abstract();
    if (dyn_cast<AbstractTuple>(dt) != nullptr) {
      abstract_tuple_count++;
    }
  }
  ASSERT_EQ(abstract_tuple_count, 4);

  // erase tuple in CNode57 and Parameter
  EraseTuple(me_graph, manager);

  abstract_tuple_count = 0;
  for (auto node : manager->all_nodes()) {
    auto dt = node->abstract();
    if (dyn_cast<AbstractTuple>(dt) != nullptr) {
      abstract_tuple_count++;
    }
  }

  ASSERT_EQ(abstract_tuple_count, 3);
}

}  // namespace opt
}  // namespace mindspore
