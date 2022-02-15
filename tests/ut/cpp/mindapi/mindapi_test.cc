/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <memory>
#include <sstream>
#include <unordered_map>
#include "common/common_test.h"
#include "mindapi/base/logging.h"
#include "mindapi/ir/func_graph.h"
#include "mindapi/ir/primitive.h"
#include "mindapi/ir/tensor.h"
#include "mindapi/ir/utils.h"

namespace mindspore::api {
class TestMindApi : public UT::Common {
 public:
  TestMindApi() = default;
};

/// Feature: MindAPI
/// Description: test basic 'is()' 'cast()'
/// Expectation: is/cast works correctly.
TEST_F(TestMindApi, test_base_isa_cast) {
  auto value_node = MakeShared<ValueNode>(MakeValue(0));
  auto base = MakeShared<Base>(value_node->impl());
  ASSERT_TRUE(base->isa<Base>());
  ASSERT_TRUE(base->isa<AnfNode>());
  ASSERT_TRUE(base->isa<ValueNode>());
  ASSERT_FALSE(base->isa<AbstractBase>());
  auto anf_node = base->cast<AnfNodePtr>();
  ASSERT_TRUE(anf_node != nullptr);
  ASSERT_TRUE(anf_node->impl() == value_node->impl());
  ASSERT_TRUE(base->cast<AbstractBasePtr>() == nullptr);
}

/// Feature: MindAPI
/// Description: test graph construction.
/// Expectation: graph is constructed as expected.
TEST_F(TestMindApi, test_graph_construction) {
  // fg(x) { return myprim(x, 1); }
  auto fg = FuncGraph::Create();
  auto x = fg->add_parameter();
  x->set_name("x");
  auto prim = MakeShared<Primitive>("myprim");
  auto prim_node = MakeShared<ValueNode>(prim);
  auto value_node = MakeShared<ValueNode>(MakeValue(1));
  auto cnode = fg->NewCNode({prim_node, x, value_node});
  fg->set_output(cnode);

  // Now we check the graph.
  ASSERT_EQ(fg->parameters().size(), 1);
  ASSERT_TRUE(fg->parameters()[0]->isa<Parameter>());
  ASSERT_EQ(fg->parameters()[0]->cast<ParameterPtr>()->name(), "x");

  auto ret_node = fg->get_return();
  ASSERT_TRUE(ret_node != nullptr);
  auto output_node = fg->output();
  ASSERT_TRUE(output_node != nullptr);
  ASSERT_TRUE(output_node->isa<CNode>());

  auto output_cnode = output_node->cast<CNodePtr>();
  ASSERT_EQ(output_cnode->inputs().size(), 3);
  ASSERT_TRUE(output_cnode->input(0)->isa<ValueNode>());
  ASSERT_TRUE(output_cnode->input(0)->cast<ValueNodePtr>()->value()->isa<Primitive>());
  ASSERT_EQ(output_cnode->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>()->name(), "myprim");
  ASSERT_TRUE(output_cnode->input(1)->isa<Parameter>());
  ASSERT_EQ(output_cnode->input(1)->cast<ParameterPtr>()->name(), "x");
  ASSERT_TRUE(output_cnode->input(2)->isa<ValueNode>());

  ASSERT_EQ(output_cnode->impl(), cnode->impl());
}

/// Feature: MindAPI
/// Description: test value related functions.
/// Expectation: value related functions work as expected.
TEST_F(TestMindApi, test_values) {
  int64_t one = 1;
  auto s = MakeValue("hello");
  auto i = MakeValue(one);
  auto i2 = MakeValue(2);
  auto b = MakeValue(true);
  auto f = MakeValue(3.14f);
  auto seq = MakeValue(std::vector<int64_t>{3, 4, 5});
  auto seq_str = MakeValue(std::vector<std::string>({"this", "is", "mindspore", "api"}));

  ASSERT_TRUE(s->isa<StringImm>());
  ASSERT_TRUE(i->isa<Int64Imm>());
  ASSERT_TRUE(i2->isa<Int64Imm>());
  ASSERT_TRUE(b->isa<BoolImm>());
  ASSERT_TRUE(f->isa<FP32Imm>());
  ASSERT_TRUE(seq->isa<ValueSequence>());
  ASSERT_TRUE(seq_str->isa<ValueSequence>());

  ASSERT_EQ(GetValue<std::string>(s), "hello");
  ASSERT_EQ(GetValue<int64_t>(i), one);
  ASSERT_EQ(GetValue<int64_t>(i2), 2);
  ASSERT_TRUE(GetValue<bool>(b));
  ASSERT_TRUE(std::abs(GetValue<float>(f) - 3.14f) < 0.00001f);

  ASSERT_EQ(GetValue<std::string>(i), "");
  ASSERT_EQ(GetValue<int64_t>(s), 0);
  ASSERT_FALSE(GetValue<bool>(s));
  ASSERT_EQ(GetValue<float>(s), 0.0f);

  auto seq_ptr = seq->cast<ValueSequencePtr>();
  ASSERT_TRUE(seq_ptr != nullptr);
  ASSERT_EQ(seq_ptr->size(), 3);
  ASSERT_EQ(seq_ptr->value().size(), 3);
  ASSERT_TRUE(seq_ptr->value()[0]->isa<Int64Imm>());
  ASSERT_EQ(GetValue<int64_t>(seq_ptr->value()[0]), 3);
  ASSERT_EQ(GetValue<int64_t>(seq_ptr->value()[1]), 4);
  ASSERT_EQ(GetValue<int64_t>(seq_ptr->value()[2]), 5);

  auto seq_values = GetValue<std::vector<int64_t>>(seq);
  ASSERT_EQ(seq_values.size(), 3);
  ASSERT_EQ(seq_values[0], 3);
  ASSERT_EQ(seq_values[1], 4);
  ASSERT_EQ(seq_values[2], 5);

  auto str_values = GetValue<std::vector<std::string>>(seq_str);
  ASSERT_EQ(str_values.size(), 4);
  ASSERT_EQ(str_values[0], "this");
  ASSERT_EQ(str_values[1], "is");
  ASSERT_EQ(str_values[2], "mindspore");
  ASSERT_EQ(str_values[3], "api");

  auto value_list = GetValue<ValuePtrList>(seq);
  ASSERT_EQ(value_list.size(), 3);
  ASSERT_EQ(utils::cast<int64_t>(value_list[0]), 3);
  ASSERT_EQ(utils::cast<int64_t>(value_list[1]), 4);
  ASSERT_EQ(utils::cast<int64_t>(value_list[2]), 5);

  std::vector<uint8_t> vec_uint8{5, 6, 7};
  auto uint8_seq = MakeValue<std::vector<uint8_t>>(vec_uint8);
  ASSERT_TRUE(uint8_seq->isa<ValueSequence>());
  auto uint8_values = GetValue<std::vector<uint8_t>>(uint8_seq);
  ASSERT_EQ(uint8_values.size(), 3);
  ASSERT_EQ(uint8_values[0], 5);
  ASSERT_EQ(uint8_values[1], 6);
  ASSERT_EQ(uint8_values[2], 7);

  auto seq_bool = MakeValue(std::vector<bool>{true, false, true});
  auto seq_bool_ptr = seq_bool->cast<ValueSequencePtr>();
  ASSERT_TRUE(seq_bool_ptr != nullptr);
  ASSERT_EQ(seq_bool_ptr->size(), 3);
  ASSERT_EQ(seq_bool_ptr->value().size(), 3);
  ASSERT_TRUE(seq_bool_ptr->value()[0]->isa<BoolImm>());
  ASSERT_TRUE(GetValue<bool>(seq_bool_ptr->value()[0]));
  ASSERT_FALSE(GetValue<bool>(seq_bool_ptr->value()[1]));
  ASSERT_TRUE(GetValue<bool>(seq_bool_ptr->value()[2]));

  auto prim = MakeShared<Primitive>("myprim");
  auto node = NewValueNode(prim);
  ASSERT_TRUE(node != nullptr);
  ASSERT_EQ(node->value(), prim);
}

/// Feature: MindAPI
/// Description: test graph manager functions.
/// Expectation: graph manager functions work as expected.
TEST_F(TestMindApi, test_func_graph_manager) {
  // fg(x, y) { return myprim(add(x, y), 1); }
  auto fg = FuncGraph::Create();
  auto x = fg->add_parameter();
  x->set_name("x");
  auto y = fg->add_parameter();
  y->set_name("y");
  auto add = MakeShared<Primitive>("add");
  auto add_node = MakeShared<ValueNode>(add);
  auto add_cnode = fg->NewCNode({add_node, x, y});
  auto prim = MakeShared<Primitive>("myprim");
  auto prim_node = MakeShared<ValueNode>(prim);
  auto value_node = MakeShared<ValueNode>(MakeValue(1));
  auto cnode = fg->NewCNode({prim_node, add_cnode, value_node});
  fg->set_output(cnode);

  auto mgr = FuncGraphManager::Manage(fg);
  ASSERT_TRUE(mgr != nullptr);
  ASSERT_TRUE(fg->manager() != nullptr);
  ASSERT_EQ(fg->manager()->impl(), mgr->impl());
  ASSERT_EQ(fg->manager(), mgr);

  ASSERT_EQ(cnode->input(1)->impl(), add_cnode->impl());
  mgr->Replace(add_cnode, x);
  ASSERT_EQ(cnode->input(1)->impl(), x->impl());

  mgr->SetEdge(cnode, 1, y);
  ASSERT_EQ(cnode->input(1)->impl(), y->impl());

  mgr->AddEdge(cnode, x);
  ASSERT_EQ(cnode->size(), 4);
  ASSERT_EQ(cnode->input(3)->impl(), x->impl());

  auto users = mgr->GetUsers(value_node);
  ASSERT_EQ(users.size(), 1);
  ASSERT_EQ(users[0].first, cnode);
  ASSERT_EQ(users[0].second, 2);
}

/// Feature: MindAPI
/// Description: test value node utils.
/// Expectation: value node utils work as expected.
TEST_F(TestMindApi, test_value_node_utils) {
  auto fg = FuncGraph::Create();
  auto fg_node = MakeShared<ValueNode>(fg);
  auto prim = MakeShared<Primitive>("myprim");
  auto prim_node = MakeShared<ValueNode>(prim);
  auto one = MakeShared<ValueNode>(MakeValue(1));
  auto cnode = fg->NewCNode({fg_node, prim_node, one});

  ASSERT_TRUE(GetValueNode<FuncGraphPtr>(cnode) == nullptr);

  auto fg1 = GetValueNode<FuncGraphPtr>(cnode->input(0));
  ASSERT_TRUE(fg1 != nullptr);
  ASSERT_TRUE(fg1->isa<FuncGraph>());

  auto prim1 = GetValueNode<PrimitivePtr>(cnode->input(1));
  ASSERT_TRUE(prim1 != nullptr);
  ASSERT_TRUE(prim1->isa<Primitive>());

  auto imm = GetValueNode<Int64ImmPtr>(cnode->input(2));
  ASSERT_TRUE(imm != nullptr);
  ASSERT_TRUE(imm->isa<Int64Imm>());
  ASSERT_EQ(imm->cast<Int64ImmPtr>()->value(), 1);

  auto value = GetValueNode(cnode->input(2));
  ASSERT_TRUE(value != nullptr);
  ASSERT_EQ(GetValue<int64_t>(value), 1);

  ASSERT_TRUE(GetValueNode<PrimitivePtr>(cnode->input(0)) == nullptr);
  ASSERT_TRUE(GetValueNode<FuncGraphPtr>(cnode->input(1)) == nullptr);
  ASSERT_TRUE(GetValueNode<StringImmPtr>(cnode->input(2)) == nullptr);

  // Test NewValueNode.
  auto int_node = NewValueNode(1);
  auto bool_node = NewValueNode(true);
  auto float_node = NewValueNode(1.23f);
  auto str_node = NewValueNode("hello");

  ASSERT_TRUE(int_node->value()->isa<Int64Imm>());
  ASSERT_EQ(int_node->value()->cast<Int64ImmPtr>()->value(), 1);
  ASSERT_TRUE(bool_node->value()->isa<BoolImm>());
  ASSERT_TRUE(bool_node->value()->cast<BoolImmPtr>()->value());
  ASSERT_TRUE(float_node->value()->isa<FP32Imm>());
  ASSERT_TRUE(std::abs(float_node->value()->cast<FP32ImmPtr>()->value() - 1.23f) < 0.0000001f);
  ASSERT_TRUE(str_node->value()->isa<StringImm>());
  ASSERT_EQ(str_node->value()->cast<StringImmPtr>()->value(), "hello");
}

/// Feature: MindAPI
/// Description: test SharedPtr.
/// Expectation: SharedPtr work as expected.
TEST_F(TestMindApi, test_object_ptr) {
  auto fg = FuncGraph::Create();
  auto fg_node = MakeShared<ValueNode>(fg);
  auto prim = MakeShared<Primitive>("myprim");
  auto prim_node = MakeShared<ValueNode>(prim);
  auto one = MakeShared<ValueNode>(MakeValue(1));
  auto cnode = fg->NewCNode({fg_node, prim_node, one});

  ASSERT_TRUE(fg != nullptr);
  ASSERT_FALSE(!fg);
  ASSERT_TRUE(fg ? true : false);
  ASSERT_TRUE((*cnode).input(0) == fg_node);
  ASSERT_TRUE(cnode->input(0) == fg_node);
  ASSERT_TRUE(cnode.get()->input(0) == fg_node);

  ASSERT_EQ(cnode->input(0), fg_node);
  ASSERT_EQ(cnode->input(1), prim_node);
  ASSERT_EQ(cnode->input(2), one);
  ASSERT_TRUE(cnode->input(0) != fg);

  AnfNodePtr p = fg_node;
  ASSERT_TRUE(p == fg_node);
  ASSERT_TRUE(p->isa<ValueNode>());
  ASSERT_TRUE(p->cast<ValueNodePtr>() != nullptr);
  ASSERT_TRUE(p->cast<ValueNodePtr>() == fg_node);

  p = cnode;
  ASSERT_TRUE(p == cnode);
  ASSERT_TRUE(p->isa<CNode>());
  ASSERT_TRUE(p->cast<CNodePtr>() != nullptr);
  ASSERT_TRUE(p->cast<CNodePtr>() == cnode);
  ASSERT_TRUE(p.get() == cnode.get());

  ASSERT_TRUE(p != nullptr);
  ASSERT_FALSE(p == nullptr);
  ASSERT_TRUE(p > nullptr);
  ASSERT_FALSE(p < nullptr);
  ASSERT_TRUE(p >= nullptr);
  ASSERT_FALSE(p <= nullptr);

  ASSERT_TRUE(nullptr != p);
  ASSERT_FALSE(nullptr == p);
  ASSERT_TRUE(nullptr < p);
  ASSERT_FALSE(nullptr > p);
  ASSERT_TRUE(nullptr <= p);
  ASSERT_FALSE(nullptr >= p);

  AnfNodePtr q = fg_node;
  ASSERT_TRUE(p != q);
  if (p.get()->impl() > q.get()->impl()) {
    ASSERT_TRUE(p > q);
    ASSERT_TRUE(p >= q);
    ASSERT_TRUE(q < p);
    ASSERT_TRUE(q <= p);
  } else {
    ASSERT_TRUE(p < q);
    ASSERT_TRUE(p <= q);
    ASSERT_TRUE(q > p);
    ASSERT_TRUE(q >= p);
  }

  std::stringstream ss1;
  std::stringstream ss2;
  ss1 << p;
  ss2 << cnode.get()->impl().get();
  ASSERT_EQ(ss1.str(), ss2.str());

  std::unordered_map<AnfNodePtr, AnfNodePtr> mymap;
  mymap.emplace(p, q);
  mymap.emplace(q, p);
  ASSERT_TRUE(mymap.find(p) != mymap.end());
  ASSERT_TRUE(mymap.find(q) != mymap.end());
  ASSERT_TRUE(mymap[p] == q);
  ASSERT_TRUE(mymap[q] == p);
}

/// Feature: MindAPI
/// Description: test Tensor API.
/// Expectation: Tensor API work as expected.
TEST_F(TestMindApi, test_tensor_api) {
  ShapeVector shape{1, 2, 3};
  auto tensor = MakeShared<Tensor>(kNumberTypeFloat32, shape);

  ASSERT_EQ(tensor->data_type(), kNumberTypeFloat32);
  ASSERT_EQ(tensor->shape(), shape);
  ASSERT_EQ(tensor->DataSize(), 6);
  ASSERT_EQ(tensor->Size(), 24);

  ShapeVector shape2{2, 3};
  tensor->set_data_type(kNumberTypeInt32);
  tensor->set_shape(shape2);
  ASSERT_EQ(tensor->data_type(), kNumberTypeInt32);
  ASSERT_EQ(tensor->shape(), shape2);

  // TensorType.
  TypePtr tensor_type = MakeShared<TensorType>(Type::GetType(TypeId::kNumberTypeFloat32));
  ASSERT_TRUE(tensor_type->isa<TensorType>());
  ASSERT_EQ(tensor_type->cast<TensorTypePtr>()->element()->type_id(), kNumberTypeFloat32);
}

/// Feature: MindAPI
/// Description: test Tensor with dynamic shape.
/// Expectation: Tensor API work as expected.
TEST_F(TestMindApi, test_tensor_with_dyn_shape) {
  ShapeVector shape{1, 2, -1, -2};
  auto tensor = MakeShared<Tensor>(kNumberTypeFloat32, shape);

  ASSERT_EQ(tensor->data_type(), kNumberTypeFloat32);
  ASSERT_EQ(tensor->shape(), shape);
  ASSERT_EQ(tensor->DataSize(), 0);
  ASSERT_EQ(tensor->Size(), 0);

  ShapeVector shape2{2, 3};
  tensor->set_data_type(kNumberTypeInt32);
  tensor->set_shape(shape2);
  ASSERT_EQ(tensor->data_type(), kNumberTypeInt32);
  ASSERT_EQ(tensor->shape(), shape2);

  ShapeVector shape3{1, -1, 3};
  auto tensor2 = MakeShared<Tensor>(kNumberTypeFloat32, shape);

  ASSERT_EQ(tensor2->data_type(), kNumberTypeFloat32);
  ASSERT_EQ(tensor2->shape(), shape);
  ASSERT_EQ(tensor2->DataSize(), 0);
  ASSERT_EQ(tensor2->Size(), 0);

  ShapeVector shape4{3, 4};
  tensor2->set_data_type(kNumberTypeInt32);
  tensor2->set_shape(shape4);
  ASSERT_EQ(tensor2->data_type(), kNumberTypeInt32);
  ASSERT_EQ(tensor2->shape(), shape4);
}

/// Feature: MindAPI
/// Description: test utils API.
/// Expectation: Tensor API work as expected.
TEST_F(TestMindApi, test_api_utils) {
  // Test utils::isa, utils::cast.
  auto anf_node = NewValueNode("hello");
  ASSERT_TRUE(utils::isa<AnfNode>(anf_node));
  ASSERT_TRUE(utils::isa<AnfNodePtr>(anf_node));
  ASSERT_FALSE(utils::isa<AbstractBase>(anf_node));
  ASSERT_TRUE(utils::cast<AnfNodePtr>(anf_node) != nullptr);
  ASSERT_TRUE(utils::cast<AbstractBasePtr>(anf_node) == nullptr);
  ASSERT_TRUE(utils::isa<std::string>(anf_node->value()));
  ASSERT_EQ(utils::cast<std::string>(anf_node->value()), "hello");

  auto int_value = MakeValue(123);
  ASSERT_TRUE(utils::isa<int64_t>(int_value));
  ASSERT_EQ(utils::cast<int64_t>(int_value), 123);

  anf_node = nullptr;
  ASSERT_FALSE(utils::isa<AnfNode>(anf_node));
  ASSERT_FALSE(utils::isa<AnfNodePtr>(anf_node));
  ASSERT_TRUE(utils::cast<AnfNodePtr>(anf_node) == nullptr);

  // Test clone graph.
  auto fg = FuncGraph::Create();
  auto x = fg->add_parameter();
  x->set_name("x");
  auto y = fg->add_parameter();
  y->set_name("y");
  auto add = MakeShared<Primitive>("add");
  auto add_node = MakeShared<ValueNode>(add);
  auto add_cnode = fg->NewCNode({add_node, x, y});
  auto prim = MakeShared<Primitive>("myprim");
  auto prim_node = MakeShared<ValueNode>(prim);
  auto value_node = MakeShared<ValueNode>(MakeValue(1));
  auto cnode = fg->NewCNode({prim_node, add_cnode, value_node});
  fg->set_output(cnode);

  auto cloned_fg = utils::CloneGraph(fg);
  ASSERT_TRUE(cloned_fg != nullptr);
  ASSERT_EQ(cloned_fg->parameters().size(), 2);
  auto new_output = cloned_fg->output();
  ASSERT_TRUE(new_output != nullptr);
  ASSERT_TRUE(new_output->isa<CNode>());
  ASSERT_EQ(new_output->cast<CNodePtr>()->size(), cnode->size());
  ASSERT_TRUE(new_output != cnode);
  ASSERT_TRUE(new_output->cast<CNodePtr>() != cnode);

  // Test get pad mode.
  auto pm_lower = MakeValue("pad");
  auto pm_upper = MakeValue("PAD");
  ASSERT_EQ(utils::GetPadMode(pm_lower), 0);
  ASSERT_EQ(utils::GetPadMode(pm_lower, false), 0);
  ASSERT_EQ(utils::GetPadMode(pm_upper, true), 0);
}

/// Feature: MindAPI
/// Description: test logging API.
/// Expectation: logging work as expected.
TEST_F(TestMindApi, test_api_logging) {
  std::string name = "mindspore";
  MS_LOG(DEBUG) << "hello debug";
  MS_LOG(INFO) << "hello info";
  MS_LOG(WARNING) << "hello warning";
  MS_LOG(ERROR) << "hello error";
  MS_LOG(ERROR) << name;
  MS_LOG(ERROR) << "hello " << name;
  MS_LOG(ERROR) << name << " hello";
  try {
    MS_LOG(EXCEPTION) << "hello exception";
    ASSERT_TRUE(false);
  } catch (...) {
  }
  ASSERT_TRUE(true);
}

/// Feature: MindAPI
/// Description: test AbstractSequence API.
/// Expectation: AbstractSequence work as expected.
TEST_F(TestMindApi, test_abstract_sequence) {
  AbstractBasePtrList abs_list;
  abs_list.emplace_back(MakeShared<AbstractScalar>(int64_t(1)));
  abs_list.emplace_back(MakeShared<AbstractScalar>(float(1.2f)));
  abs_list.emplace_back(MakeShared<AbstractScalar>(true));
  abs_list.emplace_back(MakeShared<AbstractScalar>(std::string("hello")));
  ShapeVector shape{1, 2, 3};
  abs_list.emplace_back(MakeShared<AbstractTensor>(TypeId::kNumberTypeFloat32, shape));
  auto abs_tuple = MakeShared<AbstractTuple>(abs_list);
  ASSERT_EQ(abs_tuple->elements().size(), abs_list.size());
  ASSERT_EQ(GetValue<int64_t>(abs_tuple->elements()[0]->value()), 1);
  ASSERT_TRUE(abs_tuple->elements()[1]->value()->isa<FP32Imm>());
  ASSERT_TRUE(GetValue<bool>(abs_tuple->elements()[2]->value()));
  ASSERT_EQ(GetValue<std::string>(abs_tuple->elements()[3]->value()), "hello");
  ASSERT_TRUE(abs_tuple->elements()[4]->isa<AbstractTensor>());
  ASSERT_EQ(abs_tuple->elements()[4]->type()->type_id(), TypeId::kObjectTypeTensorType);
  ASSERT_EQ(abs_tuple->elements()[4]->shape()->shape(), shape);
  ASSERT_EQ(abs_tuple->elements()[4]->cast<AbstractTensorPtr>()->element()->type()->type_id(),
            TypeId::kNumberTypeFloat32);
  ShapeVector shape2{2, 3, 4};
  abs_tuple->elements()[4]->set_shape(MakeShared<Shape>(shape2));
  ASSERT_EQ(abs_tuple->elements()[4]->shape()->shape(), shape2);
}
}  // namespace mindspore::api
