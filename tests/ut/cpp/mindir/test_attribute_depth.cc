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
#include <exception>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ir/value.h"
#include "load_mindir/load_model.h"
#include "proto/mind_ir.pb.h"

namespace mindspore {
namespace {
using AttributeType = mind_ir::AttributeProto_AttributeType;
constexpr size_t kMaxAttributeDepth = 32;
const std::vector<std::vector<AttributeType>> kContainerPatterns = {
  {mind_ir::AttributeProto::DICT},
  {mind_ir::AttributeProto::LIST},
  {mind_ir::AttributeProto::TUPLE},
  {mind_ir::AttributeProto::DICT, mind_ir::AttributeProto::LIST, mind_ir::AttributeProto::TUPLE},
  {mind_ir::AttributeProto::LIST, mind_ir::AttributeProto::DICT},
  {mind_ir::AttributeProto::TUPLE, mind_ir::AttributeProto::DICT}};

mind_ir::ModelProto MakeModel() {
  mind_ir::ModelProto model;
  model.set_producer_name("MindSpore");
  model.set_model_version("1.0");
  model.set_ir_version("1.1");
  model.set_mind_ir_version(mind_ir::IR_VERSION);
  auto graph = model.mutable_graph();
  graph->set_name("attribute_depth_test");
  auto input = graph->add_input();
  input->set_name("input");
  input->add_tensor()->set_data_type(mind_ir::TensorProto::FLOAT);
  graph->add_output()->set_name("input");
  return model;
}

void SetNestedAttribute(mind_ir::AttributeProto *attr, size_t depth, const std::vector<AttributeType> &pattern) {
  for (size_t i = 0; i < depth; ++i) {
    const auto type = pattern[i % pattern.size()];
    attr->set_type(type);
    if (type == mind_ir::AttributeProto::DICT) {
      auto entry = attr->add_values();
      entry->set_name("key");
      attr = entry->add_values();
    } else {
      attr = attr->add_values();
    }
  }
  attr->set_type(mind_ir::AttributeProto::INT64);
  attr->add_ints(42);
}

FuncGraphPtr LoadModel(const mind_ir::ModelProto &model) {
  const auto data = model.SerializeAsString();
  MindIRLoader loader(true, nullptr, 0, "AES-GCM", false);
  return loader.LoadMindIR(data.data(), data.size());
}

void CheckNestedValue(ValuePtr value, size_t depth, const std::vector<AttributeType> &pattern) {
  for (size_t i = 0; i < depth; ++i) {
    ASSERT_NE(value, nullptr);
    const auto type = pattern[i % pattern.size()];
    if (type == mind_ir::AttributeProto::DICT) {
      auto dict = value->cast<ValueDictionaryPtr>();
      ASSERT_NE(dict, nullptr);
      ASSERT_EQ(dict->value().size(), 1);
      EXPECT_EQ(GetValue<std::string>(dict->value()[0].first), "key");
      value = dict->value()[0].second;
    } else {
      EXPECT_EQ(value->isa<ValueList>(), type == mind_ir::AttributeProto::LIST);
      EXPECT_EQ(value->isa<ValueTuple>(), type == mind_ir::AttributeProto::TUPLE);
      auto sequence = value->cast<ValueSequencePtr>();
      ASSERT_NE(sequence, nullptr);
      ASSERT_EQ(sequence->size(), 1);
      value = sequence->value()[0];
    }
  }
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(GetValue<int64_t>(value), 42);
}
}  // namespace

/// Feature: MindIR attribute nesting depth limit.
/// Description: Load dictionaries, sequences and mixed containers at and below the limit.
/// Expectation: All container types, dictionary keys and scalar values are preserved.
TEST(TestMindIRAttributeDepth, AcceptsDepthWithinLimit) {
  for (const auto &pattern : kContainerPatterns) {
    SCOPED_TRACE(testing::PrintToString(pattern));
    for (const size_t depth : {size_t{1}, kMaxAttributeDepth - 1, kMaxAttributeDepth}) {
      SCOPED_TRACE(depth);
      auto model = MakeModel();
      auto attr = model.mutable_graph()->add_attribute();
      attr->set_name("nested");
      SetNestedAttribute(attr, depth, pattern);
      auto graph = LoadModel(model);
      ASSERT_NE(graph, nullptr);
      CheckNestedValue(graph->get_attr("nested"), depth, pattern);
    }
  }
}

/// Feature: MindIR attribute nesting depth limit.
/// Description: Load 33 nested containers, including dictionary/sequence mutual recursion.
/// Expectation: Attribute parsing throws the depth-limit error before building a partial graph.
TEST(TestMindIRAttributeDepth, RejectsDepthAboveLimit) {
  for (const auto &pattern : kContainerPatterns) {
    SCOPED_TRACE(testing::PrintToString(pattern));
    auto model = MakeModel();
    auto attr = model.mutable_graph()->add_attribute();
    attr->set_name("nested");
    SetNestedAttribute(attr, kMaxAttributeDepth + 1, pattern);
    // Ensure protobuf itself accepts the input, so this exercises the attribute parser's limit.
    mind_ir::ModelProto decoded;
    ASSERT_TRUE(decoded.ParseFromString(model.SerializeAsString()));
    try {
      (void)LoadModel(model);
      FAIL() << "Expected an attribute nesting depth error.";
    } catch (const std::exception &error) {
      EXPECT_NE(std::string(error.what()).find("MindIR attribute nesting depth exceeds the limit 32"),
                std::string::npos);
    }
  }
}

/// Feature: MindIR attribute nesting depth limit.
/// Description: Load many sibling containers and separate attributes at the depth limit.
/// Expectation: Only active nesting contributes to the depth, not the number of parsed containers.
TEST(TestMindIRAttributeDepth, DoesNotAccumulateDepthAcrossSiblingsOrAttributes) {
  auto model = MakeModel();
  auto attr = model.mutable_graph()->add_attribute();
  attr->set_name("siblings");
  attr->set_type(mind_ir::AttributeProto::LIST);
  const size_t num_siblings = kMaxAttributeDepth + 1;
  const std::vector<AttributeType> pattern{mind_ir::AttributeProto::DICT, mind_ir::AttributeProto::TUPLE};
  for (size_t i = 0; i < num_siblings; ++i) {
    SetNestedAttribute(attr->add_values(), kMaxAttributeDepth - 1, pattern);
  }
  auto other_attr = model.mutable_graph()->add_attribute();
  other_attr->set_name("other");
  SetNestedAttribute(other_attr, kMaxAttributeDepth, pattern);
  auto graph = LoadModel(model);
  ASSERT_NE(graph, nullptr);
  auto siblings_value = graph->get_attr("siblings");
  ASSERT_NE(siblings_value, nullptr);
  auto siblings = siblings_value->cast<ValueListPtr>();
  ASSERT_NE(siblings, nullptr);
  ASSERT_EQ(siblings->size(), num_siblings);
  for (const auto &value : siblings->value()) {
    CheckNestedValue(value, kMaxAttributeDepth - 1, pattern);
  }
  CheckNestedValue(graph->get_attr("other"), kMaxAttributeDepth, pattern);
}
}  // namespace mindspore
