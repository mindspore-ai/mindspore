/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "ir/func_graph.h"
#include "ir/value.h"
#include "load_mindir/load_model.h"
#include "proto/mind_ir.pb.h"

namespace mindspore {
namespace {
void SetIdentityGraph(mind_ir::GraphProto *graph, const std::string &name) {
  graph->set_name(name);
  auto input = graph->add_input();
  input->set_name(name + "_input");
  input->add_tensor()->set_data_type(mind_ir::TensorProto::FLOAT);
  graph->add_output()->set_name(input->name());
}

mind_ir::ModelProto MakeGraphAttributeModel(const std::string &name, int64_t marker) {
  mind_ir::ModelProto model;
  model.set_producer_name("MindSpore");
  model.set_model_version("1.0");
  model.set_ir_version("1.1");
  model.set_mind_ir_version(mind_ir::IR_VERSION);
  SetIdentityGraph(model.mutable_graph(), "cache_test_top");
  auto attr = model.mutable_graph()->add_attribute();
  attr->set_name("nested");
  attr->set_type(mind_ir::AttributeProto::GRAPH);
  SetIdentityGraph(attr->mutable_g(), name);
  auto value = attr->mutable_g()->add_attribute();
  value->set_name("marker");
  value->set_type(mind_ir::AttributeProto::INT64);
  value->set_i(marker);
  return model;
}

FuncGraphPtr LoadGraphAttributeModel(const mind_ir::ModelProto &model) {
  const auto data = model.SerializeAsString();
  MindIRLoader loader(true, nullptr, 0, "AES-GCM", false);
  return loader.LoadMindIR(data.data(), data.size());
}
}  // namespace

/// Feature: MindIR graph attribute cache lifetime.
/// Description: Load two models with the same nested graph name and different attributes.
/// Expectation: Each load builds its own graph and preserves its own attribute value.
TEST(TestMindIRGraphAttributeCache, IsolatesLoads) {
  auto first = LoadGraphAttributeModel(MakeGraphAttributeModel("same_nested_graph", 1));
  auto second = LoadGraphAttributeModel(MakeGraphAttributeModel("same_nested_graph", 2));
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  auto first_value = first->get_attr("nested");
  auto second_value = second->get_attr("nested");
  ASSERT_NE(first_value, nullptr);
  ASSERT_NE(second_value, nullptr);
  auto first_graph = first_value->cast<FuncGraphPtr>();
  auto second_graph = second_value->cast<FuncGraphPtr>();
  ASSERT_NE(first_graph, nullptr);
  ASSERT_NE(second_graph, nullptr);
  EXPECT_NE(first_graph, second_graph);
  ASSERT_NE(first_graph->get_attr("marker"), nullptr);
  ASSERT_NE(second_graph->get_attr("marker"), nullptr);
  EXPECT_EQ(GetValue<int64_t>(first_graph->get_attr("marker")), 1);
  EXPECT_EQ(GetValue<int64_t>(second_graph->get_attr("marker")), 2);
}

/// Feature: MindIR graph attribute cache lifetime.
/// Description: Release models with distinct nested graph names after loading.
/// Expectation: No process-wide cache keeps the nested graphs alive.
TEST(TestMindIRGraphAttributeCache, ReleasesGraphsAfterLoad) {
  for (size_t i = 0; i < 3; ++i) {
    std::weak_ptr<FuncGraph> weak_graph;
    {
      auto graph = LoadGraphAttributeModel(MakeGraphAttributeModel("released_graph_" + std::to_string(i), 1));
      ASSERT_NE(graph, nullptr);
      auto value = graph->get_attr("nested");
      ASSERT_NE(value, nullptr);
      auto nested = value->cast<FuncGraphPtr>();
      ASSERT_NE(nested, nullptr);
      weak_graph = nested;
    }
    EXPECT_TRUE(weak_graph.expired());
  }
}

/// Feature: MindIR graph attribute cache deduplication.
/// Description: Load two attributes referring to the same named graph in one model.
/// Expectation: Both attributes reuse the same graph object.
TEST(TestMindIRGraphAttributeCache, ReusesGraphsWithinLoad) {
  auto model = MakeGraphAttributeModel("shared_nested_graph", 1);
  const auto attr = model.graph().attribute(0);
  auto duplicate = model.mutable_graph()->add_attribute();
  duplicate->CopyFrom(attr);
  duplicate->set_name("other");
  auto graph = LoadGraphAttributeModel(model);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(graph->get_attr("nested"), nullptr);
  EXPECT_EQ(graph->get_attr("nested"), graph->get_attr("other"));
}

/// Feature: Legacy hook dependencies and parser-local graph reuse.
/// Description: Load repeated attributes with an unresolved operator or input reference.
/// Expectation: They share one graph in this load, which is released with its owner.
TEST(TestMindIRGraphAttributeCache, ReusesAndReleasesIncompleteHookGraphs) {
  for (const bool missing_input : {false, true}) {
    auto model = MakeGraphAttributeModel("incomplete_hook", 1);
    auto nested = model.mutable_graph()->mutable_attribute(0)->mutable_g();
    auto node = nested->add_node();
    node->set_op_type(missing_input ? "MakeTuple" : "REF::missing_hook_operator");
    node->add_input(missing_input ? "missing_hook_input" : nested->input(0).name());
    node->add_output("hook_result");
    nested->mutable_output(0)->set_name("hook_result");
    const auto duplicate = model.graph().attribute(0);
    model.mutable_graph()->add_attribute()->CopyFrom(duplicate);
    model.mutable_graph()->mutable_attribute(1)->set_name("other");
    std::weak_ptr<FuncGraph> weak;
    {
      auto graph = LoadGraphAttributeModel(model);
      ASSERT_NE(graph, nullptr);
      auto value = graph->get_attr("nested");
      ASSERT_NE(value, nullptr);
      auto attribute_graph = value->cast<FuncGraphPtr>();
      ASSERT_NE(attribute_graph, nullptr);
      EXPECT_EQ(value, graph->get_attr("other"));
      weak = attribute_graph;
    }
    EXPECT_TRUE(weak.expired());
  }
}

/// Feature: MindIR graph attribute compatibility and cache isolation.
/// Description: Load a graph with no outputs, then retry in a new load with a valid graph of the same name.
/// Expectation: Legacy incomplete attributes remain loadable without contaminating the next load.
TEST(TestMindIRGraphAttributeCache, IncompleteGraphDoesNotContaminateNextLoad) {
  auto model = MakeGraphAttributeModel("retry_graph", 7);
  model.mutable_graph()->mutable_attribute(0)->mutable_g()->clear_output();
  std::weak_ptr<FuncGraph> weak;
  {
    auto incomplete = LoadGraphAttributeModel(model);
    ASSERT_NE(incomplete, nullptr);
    auto value = incomplete->get_attr("nested");
    ASSERT_NE(value, nullptr);
    auto nested = value->cast<FuncGraphPtr>();
    ASSERT_NE(nested, nullptr);
    EXPECT_EQ(nested->get_return(), nullptr);
    weak = nested;
  }
  EXPECT_TRUE(weak.expired());
  auto graph = LoadGraphAttributeModel(MakeGraphAttributeModel("retry_graph", 7));
  ASSERT_NE(graph, nullptr);
  auto value = graph->get_attr("nested");
  ASSERT_NE(value, nullptr);
  auto nested = value->cast<FuncGraphPtr>();
  ASSERT_NE(nested, nullptr);
  EXPECT_NE(nested->get_return(), nullptr);
  ASSERT_NE(nested->get_attr("marker"), nullptr);
  EXPECT_EQ(GetValue<int64_t>(nested->get_attr("marker")), 7);
}
/// Feature: MindIR graph attribute cycle validation.
/// Description: A nested graph refers to its own name directly or through a second graph.
/// Expectation: Both cycles fail before an in-progress graph becomes an attribute value.
TEST(TestMindIRGraphAttributeCache, RejectsCycles) {
  for (const bool indirect : {false, true}) {
    auto model = MakeGraphAttributeModel("cycle_a", 1);
    auto nested = model.mutable_graph()->mutable_attribute(0)->mutable_g();
    if (indirect) {
      auto next = nested->add_attribute();
      next->set_name("next");
      next->set_type(mind_ir::AttributeProto::GRAPH);
      nested = next->mutable_g();
      SetIdentityGraph(nested, "cycle_b");
    }
    auto back = nested->add_attribute();
    back->set_name("back");
    back->set_type(mind_ir::AttributeProto::GRAPH);
    back->mutable_g()->set_name("cycle_a");
    EXPECT_EQ(LoadGraphAttributeModel(model), nullptr);
  }
}

/// Feature: MindIR bool loading APIs propagate graph construction errors.
/// Description: Load valid models and models with no top-level outputs through the bool overloads.
/// Expectation: Valid input succeeds; malformed input returns false with a null graph, including absent user info.
TEST(TestMindIRGraphAttributeCache, BoolOverloadsPropagateFailure) {
  for (const bool valid : {true, false}) {
    auto model = MakeGraphAttributeModel("bool_result_graph", 1);
    if (!valid) {
      model.mutable_graph()->clear_output();
    }
    const auto data = model.SerializeAsString();
    MindIRLoader loader(true, nullptr, 0, "AES-GCM", false);
    FuncGraphPtr graph;
    std::string user_info;
    EXPECT_EQ(loader.LoadMindIR(data.data(), data.size(), "", &graph, &user_info), valid);
    EXPECT_EQ(graph != nullptr, valid);
    CryptoInfo crypto{};
    EXPECT_EQ(loader.LoadMindIR(data.data(), data.size(), "", crypto, &graph, &user_info), valid);
    EXPECT_EQ(graph != nullptr, valid);
    EXPECT_EQ(loader.LoadMindIR(data.data(), data.size(), nullptr, size_t{0}, &graph, &user_info), valid);
    EXPECT_EQ(graph != nullptr, valid);
    EXPECT_EQ(loader.LoadMindIR(data.data(), data.size(), nullptr, size_t{0}, &graph, nullptr), valid);
    EXPECT_EQ(graph != nullptr, valid);
  }
}
/// Feature: MindIR node graph attribute failure propagation.
/// Description: Primitive, node and primal attributes contain a self-referencing graph.
/// Expectation: Each loading attempt fails with a null graph.
TEST(TestMindIRGraphAttributeCache, NodeGraphAttributeFailuresPropagate) {
  for (int attribute_kind = 0; attribute_kind < 3; ++attribute_kind) {
    SCOPED_TRACE(attribute_kind);
    auto model = MakeGraphAttributeModel("valid_graph", 1);
    const auto definition = model.graph().attribute(0);
    auto node = model.mutable_graph()->add_node();
    node->set_op_type("MakeTuple");
    node->add_input("cache_test_top_input");
    node->add_output("node_attr_result");
    auto attr = attribute_kind == 0 ? node->add_attribute()
                                    : (attribute_kind == 1 ? node->add_node_attr() : node->add_primal_attr());
    attr->CopyFrom(definition);
    attr->mutable_g()->set_name("malformed_node_graph");
    auto back = attr->mutable_g()->add_attribute();
    back->set_name("back");
    back->set_type(mind_ir::AttributeProto::GRAPH);
    back->mutable_g()->set_name("malformed_node_graph");
    const auto data = model.SerializeAsString();
    MindIRLoader loader(true, nullptr, 0, "AES-GCM", false);
    FuncGraphPtr graph;
    std::string user_info;
    EXPECT_FALSE(loader.LoadMindIR(data.data(), data.size(), "", &graph, &user_info));
    EXPECT_EQ(graph, nullptr);
  }
}
}  // namespace mindspore
