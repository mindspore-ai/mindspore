/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ir/anf.h"
#include "ir/tensor.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/anf_utils.h"

namespace mindspore {

using Named = Named;
using tensor::Tensor;
using tensor::TensorPtr;
using tensor::TensorPtrList;

class TestAnf : public UT::Common {
 public:
  TestAnf() {}
};

TEST_F(TestAnf, test_ValueNode) {
  auto prim = std::make_shared<Primitive>(prim::kScalarAdd);
  ValueNodePtr c = NewValueNode(prim);
  ASSERT_EQ(c->isa<ValueNode>(), true);
  ASSERT_EQ(IsValueNode<Primitive>(c), true);
  ASSERT_EQ(IsValueNode<FuncGraph>(c), false);

  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  ValueNode c1(fg);
  ASSERT_EQ(c1.value()->isa<FuncGraph>(), true);
}

TEST_F(TestAnf, test_Parameter) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  Parameter a(fg);
  assert(a.isa<Parameter>());
}

TEST_F(TestAnf, test_CNode) {
  auto primitive = prim::kPrimScalarAdd;

  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  std::string s = fg->ToString();

  Parameter param(fg);
  std::vector<AnfNodePtr> params;
  CNode app_1(params, fg);
  params.push_back(NewValueNode(primitive));
  params.push_back(AnfNodePtr(new Parameter(param)));
  CNode app(params, fg);
  assert(app.isa<CNode>());
  assert(app.IsApply(primitive));
}

TEST_F(TestAnf, is_exception) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  Parameter a(fg);
  assert(!a.isa<CNode>());
  assert(!a.isa<ValueNode>());
}

/// Feature: anf_utils
/// Description: Test FlatParameterFinder
/// Expectation: FlatParameterFinder works as expected.
TEST_F(TestAnf, test_FlatParameterFinder) {
  auto t1 = std::make_shared<Tensor>(0.1f);
  auto t2 = std::make_shared<Tensor>(0.2f);
  auto t3 = std::make_shared<Tensor>(0.3f);
  auto t4 = std::make_shared<Tensor>(0.4f);
  auto flat_tensors = Tensor::FlattenTensors(TensorPtrList{t1, t2, t3});
  assert(flat_tensors.size() == 1);
  auto t5 = flat_tensors[0];

  auto fg = std::make_shared<FuncGraph>();
  auto p1 = std::make_shared<Parameter>(fg);
  auto p2 = std::make_shared<Parameter>(fg);
  auto p3 = std::make_shared<Parameter>(fg);
  auto p4 = std::make_shared<Parameter>(fg);
  auto p5 = std::make_shared<Parameter>(fg);
  auto p6 = std::make_shared<Parameter>(fg);
  p1->set_default_param(t1);
  p2->set_default_param(t2);
  p3->set_default_param(t3);
  p4->set_default_param(t4);
  p5->set_default_param(t5);

  FlatParameterFinder finder;
  finder.AddParameter(p1);
  std::vector<AnfNodePtr> nodes{p2, p3, p4, p5, p6};
  finder.AddNodes(nodes);
  auto flat_params = finder.GetFlatParameters();
  assert(flat_params.size() == 1);
  assert(*(flat_params.begin()) == p5);

  auto [flat_param1, offset1] = finder.FindFlatParameter(p1);
  assert(flat_param1 == p5);
  assert(offset1 == 0);

  auto [flat_param2, offset2] = finder.FindFlatParameter(p2);
  assert(flat_param2 == p5);
  assert(offset2 == sizeof(float));

  auto [flat_param3, offset3] = finder.FindFlatParameter(p3);
  assert(flat_param3 == p5);
  assert(offset3 == offset2 + sizeof(float));

  auto [flat_param4, offset4] = finder.FindFlatParameter(p4);
  assert(flat_param4 == nullptr);
  assert(offset4 == 0);

  auto [flat_param5, offset5] = finder.FindFlatParameter(p5);
  assert(flat_param5 == nullptr);
  assert(offset5 == 0);

  auto [flat_param6, offset6] = finder.FindFlatParameter(p6);
  assert(flat_param6 == nullptr);
  assert(offset6 == 0);
}

/// Feature: Flatten tensor
/// Description: Test is_sub_data() & has_sub_data() api
/// Expectation: API works as expected.
TEST_F(TestAnf, test_TensorWithSubData) {
  auto t1 = std::make_shared<Tensor>(0.1f);
  auto t2 = std::make_shared<Tensor>(0.2f);
  auto t3 = std::make_shared<Tensor>(0.3f);
  auto t4 = std::make_shared<Tensor>(0.4f);
  assert(!t1->data().is_sub_data());
  assert(!t1->data().has_sub_data());
  auto flat_tensors = Tensor::FlattenTensors(TensorPtrList{t1, t2, t3});
  assert(flat_tensors.size() == 1);
  assert(!flat_tensors[0]->data().is_sub_data());
  assert(flat_tensors[0]->data().has_sub_data());
  assert(t1->data().is_sub_data());
  assert(!t1->data().has_sub_data());
  assert(t2->data().is_sub_data());
  assert(!t2->data().has_sub_data());
}

/// Feature: Compression tensor
/// Description: test Tensor API.
/// Expectation: Tensor API work as expected.
TEST_F(TestAnf, test_CompressionTensor) {
  ShapeVector shape{1, 224, 224, 3};
  auto data_size = 50;
  auto tensor_int8 = std::make_shared<Tensor>(kNumberTypeInt8, shape, data_size, kFSE);
  ASSERT_EQ(tensor_int8->data_type(), kNumberTypeInt8);
  ASSERT_EQ(tensor_int8->shape(), shape);
  ASSERT_EQ(tensor_int8->DataSize(), data_size);
  ASSERT_EQ(tensor_int8->Size(), data_size);
  ASSERT_EQ(tensor_int8->compression_type(), kFSE);

  auto tensor_int16 = std::make_shared<Tensor>(kNumberTypeInt16, shape, data_size, kBitPacking);
  ASSERT_EQ(tensor_int16->data_type(), kNumberTypeInt16);
  ASSERT_EQ(tensor_int16->shape(), shape);
  ASSERT_EQ(tensor_int16->DataSize(), data_size);
  ASSERT_EQ(tensor_int16->Size(), data_size);
  ASSERT_EQ(tensor_int16->compression_type(), kBitPacking);
}
}  // namespace mindspore
