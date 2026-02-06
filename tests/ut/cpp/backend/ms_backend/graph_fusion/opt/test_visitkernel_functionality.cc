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
#include <string>

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "include/utils/anfalgo.h"
#include "ir/anf.h"

namespace mindspore::graphkernel::test {
/// Feature: test functionality of method VisitKernel
/// Description: test functionality of VisitKernel in the situation of nested getitems
/// Expectation: the method should return correct parameter
TEST_F(GraphKernelCommonTestSuite, visit_nesting_getitems_0) {
  ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {128, 128});
  auto p1 = c.NewTensorInput("p1", kFloat32, {128, 128});
  auto p2 = c.NewTensorInput("p2", kFloat32, {128, 128});

  auto p3 = c.NewTensorInput("p3", kFloat32, {128, 128});
  auto p4 = c.NewTensorInput("p4", kFloat32, {128, 128});
  auto p5 = c.NewTensorInput("p5", kFloat32, {128, 128});

  auto maketuple_0 = c.NewCNode("MakeTuple", {p0, p1, p2});
  auto maketuple_1 = c.NewCNode("MakeTuple", {p3, p4, p5});
  auto maketuple_2 = c.NewCNode("MakeTuple", {maketuple_0, maketuple_1});

  auto tuple_get_item_0 = c.NewCNode("TupleGetItem", {maketuple_2, c.NewValueNode(MakeValue((int64_t)1))});
  auto tuple_get_item_1 = c.NewCNode("TupleGetItem", {tuple_get_item_0, c.NewValueNode(MakeValue((int64_t)2))});

  // visit the second output of tuple_get_item_1
  auto node_with_idx = common::AnfAlgo::VisitKernel(tuple_get_item_1, 2);
  EXPECT_TRUE(node_with_idx.first->isa<Parameter>());
  auto res_p = node_with_idx.first->cast<ParameterPtr>();
  const std::string &name = res_p->name();
  EXPECT_EQ(name, "p5");
  EXPECT_EQ(node_with_idx.second, 0);
}

/// Feature: test functionality of method VisitKernel
/// Description: test functionality of VisitKernel in the situation of nested getitems, a more complicated case.
/// Expectation: the method should return correct parameter
TEST_F(GraphKernelCommonTestSuite, visit_nesting_getitems_1) {
  ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {32, 32});
  auto p1 = c.NewTensorInput("p1", kFloat32, {32, 32});
  auto p2 = c.NewTensorInput("p2", kFloat32, {32, 32});
  auto p3 = c.NewTensorInput("p3", kFloat32, {32, 32});
  auto p4 = c.NewTensorInput("p4", kFloat32, {32, 32});
  auto p5 = c.NewTensorInput("p5", kFloat32, {32, 32});
  auto p6 = c.NewTensorInput("p6", kFloat32, {32, 32});

  auto maketuple_0 = c.NewCNode("MakeTuple", {p0, p1, p2});
  auto maketuple_1 = c.NewCNode("MakeTuple", {p3, maketuple_0, p4});
  auto maketuple_2 = c.NewCNode("MakeTuple", {maketuple_1, p5});
  auto maketuple_3 = c.NewCNode("MakeTuple", {p6, maketuple_2});
  auto tuple_get_item_0 = c.NewCNode("TupleGetItem", {maketuple_3, c.NewValueNode(MakeValue((int64_t)1))});
  auto tuple_get_item_1 = c.NewCNode("TupleGetItem", {tuple_get_item_0, c.NewValueNode(MakeValue((int64_t)0))});
  auto tuple_get_item_2 = c.NewCNode("TupleGetItem", {tuple_get_item_1, c.NewValueNode(MakeValue((int64_t)1))});
  auto tuple_get_item_3 = c.NewCNode("TupleGetItem", {tuple_get_item_2, c.NewValueNode(MakeValue((int64_t)2))});
  auto depend = c.NewCNode("Depend", {tuple_get_item_3, c.NewValueNode(kUMonad)});

  // visit the first output of depend, which is tuple_get_item_3
  auto node_with_idx = common::AnfAlgo::VisitKernel(depend, 0);
  EXPECT_TRUE(node_with_idx.first->isa<Parameter>());
  auto res_p = node_with_idx.first->cast<ParameterPtr>();
  const std::string &name = res_p->name();
  EXPECT_EQ(name, "p2");
  EXPECT_EQ(node_with_idx.second, 0);
}
}  // namespace mindspore::graphkernel::test
