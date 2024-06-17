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

#define USE_DEPRECATED_API
#include <memory>
#include "common/common_test.h"
#include "ops/fusion/layer_norm_fusion.h"
#include "tools/converter/parser/onnx/onnx_layer_norm_parser.h"
#include "include/registry/converter_context.h"
#include "include/registry/node_parser_registry.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"
namespace mindspore {
class OnnxLayerNormParserTest : public mindspore::CommonTest {
 public:
  OnnxLayerNormParserTest() = default;
};

bool TestOnnxLayerNormNode1() {
  auto onnx_graph = std::make_shared<onnx::GraphProto>();
  onnx_graph->set_name("onnx_layer_norm_graph");
  onnx::NodeProto *onnx_layer_norm_node = onnx_graph->add_node();
  onnx_layer_norm_node->set_name("layer_layer_norm_node");
  onnx_layer_norm_node->set_op_type("LayerNormalization");
  onnx_layer_norm_node->add_input("data");

  onnx::AttributeProto *epsilon_attr = onnx_layer_norm_node->add_attribute();
  epsilon_attr->set_name("epsilon");
  epsilon_attr->set_f(0.1);
  epsilon_attr->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS);

  onnx::AttributeProto *axis_attr = onnx_layer_norm_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_i(1);
  axis_attr->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);

  ops::PrimitiveCPtr primitive_c;
  auto node_parser_builtin = lite::OnnxNodeParserRegistry::GetInstance().GetNodeParser(onnx_layer_norm_node->op_type());
  if (node_parser_builtin == nullptr) {
    MS_LOG(ERROR) << "not support onnx data type " << onnx_layer_norm_node->op_type();
    return false;
  }
  primitive_c = node_parser_builtin->Parse(*onnx_graph, *onnx_layer_norm_node);
  if (!primitive_c->HasAttr("axis")) {
    MS_LOG(ERROR) << "prim has no axis!";
    return false;
  }
  auto axis_value = GetValue<int>(primitive_c->GetAttr("axis"));
  if (axis_value != 1) {
    MS_LOG(ERROR) << "axis value is wrong!";
    return false;
  }
  if (!primitive_c->HasAttr("epsilon")) {
    MS_LOG(ERROR) << "prim has no axis!";
    return false;
  }
  auto epsilon_value = GetValue<float>(primitive_c->GetAttr("epsilon"));
  if (epsilon_value != 0.1) {
    MS_LOG(ERROR) << "axis value is wrong!";
    return false;
  }
  return true;
}

bool TestOnnxLayerNormNode2() {
  auto onnx_graph = std::make_shared<onnx::GraphProto>();
  onnx_graph->set_name("LN_graph");
  onnx::NodeProto *onnx_layer_norm_node = onnx_graph->add_node();
  onnx_layer_norm_node->set_name("LN");
  onnx_layer_norm_node->set_op_type("LayerNormalization");
  onnx_layer_norm_node->add_input("input_data");

  onnx::AttributeProto *epsilon_attr = onnx_layer_norm_node->add_attribute();
  epsilon_attr->set_name("epsilon");
  epsilon_attr->set_f(1);

  onnx::AttributeProto *axis_attr = onnx_layer_norm_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_i(1);

  ops::PrimitiveCPtr primitive_c;
  auto node_parser_builtin = lite::OnnxNodeParserRegistry::GetInstance().GetNodeParser(onnx_layer_norm_node->op_type());
  if (node_parser_builtin == nullptr) {
    MS_LOG(ERROR) << "not support onnx data type " << onnx_layer_norm_node->op_type();
    return false;
  }
  primitive_c = node_parser_builtin->Parse(*onnx_graph, *onnx_layer_norm_node);
  if (!primitive_c->HasAttr("axis")) {
    MS_LOG(ERROR) << "prim has no axis!";
    return false;
  }
  auto axis_value = GetValue<int>(primitive_c->GetAttr("axis"));
  if (axis_value != 1) {
    MS_LOG(ERROR) << "axis value is wrong!";
    return false;
  }
  if (!primitive_c->HasAttr("epsilon")) {
    MS_LOG(ERROR) << "prim has no axis!";
    return false;
  }
  auto epsilon_value = GetValue<float>(primitive_c->GetAttr("epsilon"));
  if (epsilon_value != 1) {
    MS_LOG(ERROR) << "axis value is wrong!";
    return false;
  }
  return true;
}

TEST_F(OnnxLayerNormParserTest, OnnxLayerNormParserTest1) {
  auto ret = TestOnnxLayerNormNode1();
  ASSERT_EQ(ret, true);
}

TEST_F(OnnxLayerNormParserTest, OnnxLayerNormParserTest2) {
  auto ret = TestOnnxLayerNormNode2();
  ASSERT_EQ(ret, true);
}
}  // namespace mindspore
