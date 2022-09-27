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

#define USE_DEPRECATED_API
#include <memory>
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/anf_transform.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "test/common/import_from_meta_graphT.h"

namespace mindspore {
class ActivationFusionTest : public mindspore::CommonTest {
 public:
  ActivationFusionTest() = default;
};
using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

namespace {
inline const int kActMinVal1 = -10;
inline const int kActMaxVal1 = 7;
inline const int kActMinVal2 = -5;
inline const int kActMaxVal2 = 10;
inline const int kHeight = 5;
inline const int kWidth = 5;
inline const int kChannel = 3;
inline const int kValueThreshold = 6;

MetaGraphTptr BuildGraph(schema::ActivationType first_act_type, schema::ActivationType second_act_type) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  // hard_tanh node [-10, 6]
  auto first_node = std::make_unique<schema::CNodeT>();
  first_node->inputIndex = {0};
  first_node->outputIndex = {1};
  first_node->primitive = std::make_unique<schema::PrimitiveT>();
  first_node->primitive->value.type = schema::PrimitiveType_Activation;
  auto prim1 = new schema::ActivationT;
  prim1->activation_type = first_act_type;
  if (first_act_type == schema::ActivationType_HARD_TANH) {
    prim1->min_val = kActMinVal1;
    prim1->max_val = kActMaxVal1;
  }
  first_node->primitive->value.value = prim1;
  first_node->name = "activation_1";
  meta_graph->nodes.emplace_back(std::move(first_node));

  auto second_node = std::make_unique<schema::CNodeT>();
  second_node->inputIndex = {1};
  second_node->outputIndex = {2};
  second_node->primitive = std::make_unique<schema::PrimitiveT>();
  second_node->primitive->value.type = schema::PrimitiveType_Activation;
  auto prim2 = new schema::ActivationT;
  prim2->activation_type = second_act_type;
  if (second_act_type == schema::ActivationType_HARD_TANH) {
    prim2->min_val = kActMinVal2;
    prim2->max_val = kActMaxVal2;
  }
  second_node->primitive->value.value = prim2;
  second_node->name = "activation_2";
  meta_graph->nodes.emplace_back(std::move(second_node));

  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {2};

  // input: data
  auto input = std::make_unique<schema::TensorT>();
  input->nodeType = lite::NodeType_Parameter;
  input->format = schema::Format_NHWC;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->dims = {1, kHeight, kWidth, kChannel};
  input->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input));

  // conv output
  auto mid_output = std::make_unique<schema::TensorT>();
  mid_output->nodeType = lite::NodeType_Parameter;
  mid_output->format = schema::Format_NHWC;
  mid_output->dataType = TypeId::kNumberTypeFloat32;
  mid_output->dims = {1, kHeight, kWidth, kChannel};
  meta_graph->allTensors.emplace_back(std::move(mid_output));

  // final output
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = {1, kHeight, kWidth, kChannel};
  meta_graph->allTensors.emplace_back(std::move(output));
  return meta_graph;
}
}  //  namespace
TEST_F(ActivationFusionTest, TestHardTanhReluNode) {
  auto meta_graph = BuildGraph(schema::ActivationType_HARD_TANH, schema::ActivationType_RELU);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph, nullptr);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
  auto &cnode = new_meta_graph->nodes.at(0);
  ASSERT_EQ(cnode->primitive->value.AsActivation()->activation_type, schema::ActivationType_HARD_TANH);
  ASSERT_EQ(cnode->primitive->value.AsActivation()->min_val, 0);
  ASSERT_EQ(cnode->primitive->value.AsActivation()->max_val, kActMaxVal1);
}

TEST_F(ActivationFusionTest, TestRelu6HardTanhNode) {
  auto meta_graph = BuildGraph(schema::ActivationType_RELU6, schema::ActivationType_HARD_TANH);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph, nullptr);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
  auto &cnode = new_meta_graph->nodes.at(0);
  ASSERT_EQ(cnode->primitive->value.AsActivation()->activation_type, schema::ActivationType_RELU6);
  ASSERT_EQ(cnode->primitive->value.AsActivation()->min_val, 0);
  ASSERT_EQ(cnode->primitive->value.AsActivation()->max_val, kValueThreshold);
}

TEST_F(ActivationFusionTest, TestBadCase_ReluSigmoid) {
  auto meta_graph = BuildGraph(schema::ActivationType_RELU, schema::ActivationType_SIGMOID);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph, nullptr);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 2);
  auto &first_node = new_meta_graph->nodes.at(0);
  auto &second_node = new_meta_graph->nodes.at(1);
  ASSERT_EQ(first_node->primitive->value.AsActivation()->activation_type, schema::ActivationType_RELU);
  ASSERT_EQ(second_node->primitive->value.AsActivation()->activation_type, schema::ActivationType_SIGMOID);
}
}  // namespace mindspore
