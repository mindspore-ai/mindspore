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
#include <functional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "load_mindir/load_model.h"
#include "proto/mind_ir.pb.h"

namespace mindspore {
namespace {
using AttributeType = mind_ir::AttributeProto_AttributeType;

mind_ir::ModelProto MakeModel() {
  mind_ir::ModelProto model;
  model.set_producer_name("MindSpore");
  model.set_model_version("1.0");
  model.set_ir_version("1.1");
  model.set_mind_ir_version(mind_ir::IR_VERSION);
  auto graph = model.mutable_graph();
  graph->set_name("scalar_attr_form_test");
  auto input = graph->add_input();
  input->set_name("input");
  input->add_tensor()->set_data_type(mind_ir::TensorProto::FLOAT);
  graph->add_output()->set_name("input");
  return model;
}

// A node whose op_type differs from "Constant" is imported as a CNode, whose
// attributes are parsed through GetAttrValueForCNode/SetCNodePrimAttrAndAbstract.
mind_ir::NodeProto *AddCNode(mind_ir::ModelProto *model, const std::string &name) {
  auto graph = model->mutable_graph();
  graph->mutable_output(0)->set_name(name);
  auto node = graph->add_node();
  node->set_op_type("Split");
  node->set_domain(name);
  node->add_input("input");
  node->add_output(name);
  return node;
}

// A node whose op_type is "Constant" is imported as a ValueNode, whose
// attributes are parsed through GetAttrValueForValueNode.
mind_ir::NodeProto *AddValueNode(mind_ir::ModelProto *model, const std::string &name) {
  auto graph = model->mutable_graph();
  graph->mutable_output(0)->set_name(name);
  auto node = graph->add_node();
  node->set_op_type("Constant");
  node->set_domain(name);
  node->add_output(name);
  return node;
}

mind_ir::AttributeProto *AddScalarFormAttr(mind_ir::NodeProto *node, const std::string &name,
                                           const std::string &ref_attr_name, AttributeType type) {
  auto attr = node->add_attribute();
  attr->set_name(name);
  attr->set_ref_attr_name(ref_attr_name);
  attr->set_type(type);
  return attr;
}

FuncGraphPtr LoadModel(const mind_ir::ModelProto &model) {
  const auto data = model.SerializeAsString();
  MindIRLoader loader;
  return loader.LoadMindIR(data.data(), data.size());
}

PrimitivePtr GetPrimitiveOfOutput(const FuncGraphPtr &graph) {
  auto output = graph->output();
  if (output == nullptr) {
    return nullptr;
  }
  auto cnode = output->cast<CNodePtr>();
  if (cnode == nullptr) {
    return nullptr;
  }
  return GetValueNode<PrimitivePtr>(cnode->input(0));
}

// The fields that historically drove the element loop of ObtainCNodeAttrInScalarForm.
struct MismatchedField {
  const char *name;
  std::function<void(mind_ir::AttributeProto *)> fill;
};

const std::vector<MismatchedField> &MismatchedFields() {
  static const std::vector<MismatchedField> fields = {
    {"ints", [](mind_ir::AttributeProto *attr) { attr->add_ints(1); }},
    {"doubles", [](mind_ir::AttributeProto *attr) { attr->add_doubles(1.5); }},
    {"floats", [](mind_ir::AttributeProto *attr) { attr->add_floats(1.5); }},
    {"strings", [](mind_ir::AttributeProto *attr) { attr->add_strings("x"); }},
  };
  return fields;
}
}  // namespace

/// Feature: MindIR scalar-form attribute parsing.
/// Description: A well-formed TENSORS scalar-form attribute on a CNode also carries stray ints.
///               The stray field used to drive the element loop past tensors_size() and read out
///               of bounds (abort / OOB read).
/// Expectation: Only the in-range tensors are parsed; the attribute is restored as declared.
TEST(TestMindIRScalarAttrForm, LoadsWellFormedTensorsAttrOnCNode) {
  auto model = MakeModel();
  auto node = AddCNode(&model, "n_tensor");
  auto attr = AddScalarFormAttr(node, "axis", "scalar:Tuple[value1,value2]", mind_ir::AttributeProto::TENSORS);
  attr->add_tensors()->set_data_type(mind_ir::TensorProto::FLOAT);
  attr->add_tensors()->set_data_type(mind_ir::TensorProto::INT32);
  // Stray ints must not extend the element count.
  attr->add_ints(9);
  attr->add_ints(9);
  attr->add_ints(9);
  auto graph = LoadModel(model);
  ASSERT_NE(graph, nullptr);
  auto prim = GetPrimitiveOfOutput(graph);
  ASSERT_NE(prim, nullptr);
  auto value = prim->GetAttr("axis");
  ASSERT_NE(value, nullptr);
  auto tuple = value->cast<ValueTuplePtr>();
  ASSERT_NE(tuple, nullptr);
  ASSERT_EQ(tuple->size(), 2);
  ASSERT_TRUE(tuple->value()[0]->isa<Type>());
  EXPECT_EQ(tuple->value()[0]->cast<TypePtr>()->type_id(), kFloat32->type_id());
  ASSERT_TRUE(tuple->value()[1]->isa<Type>());
  EXPECT_EQ(tuple->value()[1]->cast<TypePtr>()->type_id(), kInt32->type_id());
}

/// Feature: MindIR scalar-form attribute parsing.
/// Description: Load a well-formed TENSORS scalar-form attribute on a ValueNode.
/// Expectation: The ValueNode holds a tuple of the declared data types.
TEST(TestMindIRScalarAttrForm, LoadsWellFormedTensorsAttrOnValueNode) {
  auto model = MakeModel();
  auto node = AddValueNode(&model, "v_tensor");
  auto attr = AddScalarFormAttr(node, "dtype", "scalar:Tuple[value1]", mind_ir::AttributeProto::TENSORS);
  attr->add_tensors()->set_data_type(mind_ir::TensorProto::FLOAT);
  auto graph = LoadModel(model);
  ASSERT_NE(graph, nullptr);
  auto output = graph->output();
  ASSERT_NE(output, nullptr);
  auto value_node = output->cast<ValueNodePtr>();
  ASSERT_NE(value_node, nullptr);
  auto tuple = value_node->value()->cast<ValueTuplePtr>();
  ASSERT_NE(tuple, nullptr);
  ASSERT_EQ(tuple->size(), 1);
  ASSERT_TRUE(tuple->value()[0]->isa<Type>());
  EXPECT_EQ(tuple->value()[0]->cast<TypePtr>()->type_id(), kFloat32->type_id());
}

/// Feature: MindIR scalar-form attribute parsing.
/// Description: A TENSORS-type attribute with empty tensors but a non-empty scalar field was
///               historically indexed with the size of that scalar field, reading past the
///               tensors field (out-of-bounds abort / read) on the CNode path.
/// Expectation: No element is parsed, so the attribute is skipped and the model still loads.
TEST(TestMindIRScalarAttrForm, SkipsTensorsAttrWithoutTensorsOnCNode) {
  for (const auto &field : MismatchedFields()) {
    SCOPED_TRACE(field.name);
    auto model = MakeModel();
    auto node = AddCNode(&model, "n_bad");
    auto attr = AddScalarFormAttr(node, "axis", "scalar:Tuple[value1]", mind_ir::AttributeProto::TENSORS);
    field.fill(attr);
    auto graph = LoadModel(model);
    ASSERT_NE(graph, nullptr);
    auto prim = GetPrimitiveOfOutput(graph);
    ASSERT_NE(prim, nullptr);
    EXPECT_EQ(prim->GetAttr("axis"), nullptr);
  }
}

/// Feature: MindIR scalar-form attribute parsing.
/// Description: The same mismatched attribute on a ValueNode reads past the tensors field
///               (out-of-bounds abort / read) on the ValueNode path.
/// Expectation: No element is parsed, so the ValueNode is not built and the load fails cleanly.
TEST(TestMindIRScalarAttrForm, SkipsTensorsAttrWithoutTensorsOnValueNode) {
  for (const auto &field : MismatchedFields()) {
    SCOPED_TRACE(field.name);
    auto model = MakeModel();
    auto node = AddValueNode(&model, "v_bad");
    auto attr = AddScalarFormAttr(node, "dtype", "scalar:Tuple[value1]", mind_ir::AttributeProto::TENSORS);
    field.fill(attr);
    auto graph = LoadModel(model);
    EXPECT_EQ(graph, nullptr);
  }
}

/// Feature: MindIR scalar-form attribute parsing.
/// Description: A TENSORS scalar-form attribute whose tensor carries an unsupported data type
///               yields a null element; storing it would poison the value sequence (the
///               ValueNode path used to fail inside ValueTuple::ToAbstract()).
/// Expectation: The element is skipped: the ValueNode load fails cleanly and the CNode
///               attribute is absent, instead of building a poisoned sequence.
TEST(TestMindIRScalarAttrForm, SkipsTensorsAttrWithUnsupportedDataType) {
  {
    auto model = MakeModel();
    auto node = AddValueNode(&model, "v_undef");
    auto attr = AddScalarFormAttr(node, "dtype", "scalar:Tuple[value1]", mind_ir::AttributeProto::TENSORS);
    attr->add_tensors()->set_data_type(mind_ir::TensorProto::UNDEFINED);
    auto graph = LoadModel(model);
    EXPECT_EQ(graph, nullptr);
  }
  {
    auto model = MakeModel();
    auto node = AddCNode(&model, "n_undef");
    auto attr = AddScalarFormAttr(node, "axis", "scalar:Tuple[value1]", mind_ir::AttributeProto::TENSORS);
    attr->add_tensors()->set_data_type(mind_ir::TensorProto::UNDEFINED);
    auto graph = LoadModel(model);
    ASSERT_NE(graph, nullptr);
    auto prim = GetPrimitiveOfOutput(graph);
    ASSERT_NE(prim, nullptr);
    EXPECT_EQ(prim->GetAttr("axis"), nullptr);
  }
}

/// Feature: MindIR scalar-form attribute parsing.
/// Description: Mirror of the reported PoC: ref_attr_name "scalarx", type TENSORS, empty
///               tensors and 8192 ints, carried by a CNode attribute.
/// Expectation: The process survives and the model loads without the attribute.
TEST(TestMindIRScalarAttrForm, SurvivesReportedPocShapeOnCNode) {
  auto model = MakeModel();
  auto node = AddCNode(&model, "n_poc");
  auto attr = AddScalarFormAttr(node, "axis", "scalarx", mind_ir::AttributeProto::TENSORS);
  for (int i = 0; i < 8192; ++i) {
    attr->add_ints(1);
  }
  auto graph = LoadModel(model);
  ASSERT_NE(graph, nullptr);
  auto prim = GetPrimitiveOfOutput(graph);
  ASSERT_NE(prim, nullptr);
  EXPECT_EQ(prim->GetAttr("axis"), nullptr);
}
}  // namespace mindspore
