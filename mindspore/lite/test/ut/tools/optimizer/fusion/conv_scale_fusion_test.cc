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

#include <memory>
#include "schema/inner/model_generated.h"
#include "include/model.h"
#include "common/common_test.h"
#include "include/lite_session.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/anf_transform.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "test/common/import_from_meta_graphT.h"

namespace mindspore {
class ConvScaleFusionTest : public mindspore::CommonTest {
 public:
  ConvScaleFusionTest() = default;
};
using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

namespace {
// conv has 2 inputs
CNodeTptr BuildConv2D(int with_bias_flag) {
  auto convNode = std::make_unique<schema::CNodeT>();
  if (with_bias_flag) {
    convNode->inputIndex = {0, 1, 2};
    convNode->outputIndex = {3};
  } else {
    convNode->inputIndex = {0, 1};
    convNode->outputIndex = {2};
  }
  convNode->primitive = std::make_unique<schema::PrimitiveT>();
  convNode->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto prim1 = new schema::Conv2DFusionT;
  prim1->pad_mode = schema::PadMode_SAME;
  prim1->format = schema::Format_NHWC;
  prim1->stride = {1, 1};
  prim1->kernel_size = {3, 3};
  prim1->dilation = {1, 1};
  prim1->out_channel = 3;
  convNode->primitive->value.value = prim1;
  convNode->name = "Conv2D";
  return convNode;
}
// conv2d has 3 inputs
CNodeTptr BuildDepthwiseConv2D(int with_bias_flag) {
  auto convNode = std::make_unique<schema::CNodeT>();
  if (with_bias_flag) {
    convNode->inputIndex = {0, 1, 2};
    convNode->outputIndex = {3};
  } else {
    convNode->inputIndex = {0, 1};
    convNode->outputIndex = {2};
  }
  convNode->primitive = std::make_unique<schema::PrimitiveT>();
  convNode->primitive->value.type = schema::PrimitiveType_Conv2DFusion;
  auto prim1 = new schema::Conv2DFusionT;
  prim1->pad_mode = schema::PadMode_SAME;
  prim1->format = schema::Format_NHWC;
  prim1->stride = {1, 1};
  prim1->kernel_size = {3, 3};
  prim1->dilation = {1, 1};
  prim1->in_channel = 1;
  convNode->primitive->value.value = prim1;
  convNode->name = "Conv2D";
  return convNode;
}

MetaGraphTptr BuildGraph(schema::PrimitiveType conv_type, bool conv_with_bias) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  // conv node
  CNodeTptr convNode;
  if (conv_type == schema::PrimitiveType_Conv2DFusion) {
    convNode = BuildConv2D(conv_with_bias);
  } else {
    convNode = BuildDepthwiseConv2D(conv_with_bias);
  }

  meta_graph->nodes.emplace_back(std::move(convNode));

  // scale_node weight bias
  auto scale_node = std::make_unique<schema::CNodeT>();
  if (conv_with_bias) {
    scale_node->inputIndex = {3, 4, 5};
    scale_node->outputIndex = {6};
  } else {
    scale_node->inputIndex = {2, 3, 4};
    scale_node->outputIndex = {5};
  }

  scale_node->primitive = std::make_unique<schema::PrimitiveT>();
  scale_node->primitive->value.type = schema::PrimitiveType_ScaleFusion;
  auto prim2 = new schema::ScaleFusionT;
  scale_node->primitive->value.value = prim2;
  scale_node->name = "scale";
  meta_graph->nodes.emplace_back(std::move(scale_node));

  // input 0: data
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_ValueNode;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 5, 5, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // input 1: weight
  auto input1 = std::make_unique<schema::TensorT>();
  input1->nodeType = lite::NodeType_ValueNode;
  input1->format = schema::Format_KHWC;
  input1->dataType = TypeId::kNumberTypeFloat32;
  input1->dims = {8, 3, 3, 3};
  input1->data.resize(sizeof(float) * 8 * 3 * 3 * 3);
  meta_graph->allTensors.emplace_back(std::move(input1));

  if (conv_with_bias) {
    // input 00: bias
    auto input00 = std::make_unique<schema::TensorT>();
    input00->nodeType = lite::NodeType_ValueNode;
    input00->format = schema::Format_NHWC;
    input00->dataType = TypeId::kNumberTypeFloat32;
    input00->dims = {1, 5, 5, 3};
    input00->offset = -1;
    meta_graph->allTensors.emplace_back(std::move(input00));
  }

  // conv output
  auto conv_output = std::make_unique<schema::TensorT>();
  conv_output->nodeType = lite::NodeType_Parameter;
  conv_output->format = schema::Format_NHWC;
  conv_output->dataType = TypeId::kNumberTypeFloat32;
  conv_output->dims = {1, 5, 5, 8};
  meta_graph->allTensors.emplace_back(std::move(conv_output));

  // scale weight input
  auto input2 = std::make_unique<schema::TensorT>();
  input2->nodeType = lite::NodeType_ValueNode;
  input2->format = schema::Format_NHWC;
  input2->dataType = TypeId::kNumberTypeFloat32;
  input2->dims = {1, 5, 5, 8};
  input2->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input2));

  // scale bias input
  auto input3 = std::make_unique<schema::TensorT>();
  input3->nodeType = lite::NodeType_ValueNode;
  input3->format = schema::Format_NHWC;
  input3->dataType = TypeId::kNumberTypeFloat32;
  input3->dims = {1, 5, 5, 8};
  input3->data.resize(sizeof(float) * 8 * 5 * 5);
  meta_graph->allTensors.emplace_back(std::move(input3));

  // final scale output
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = {1, 5, 5, 8};
  meta_graph->allTensors.emplace_back(std::move(output));
  if (conv_with_bias) {
    meta_graph->inputIndex = {0};
    meta_graph->outputIndex = {6};
  } else {
    meta_graph->inputIndex = {0};
    meta_graph->outputIndex = {5};
  }
  return meta_graph;
}
}  //  namespace
TEST_F(ConvScaleFusionTest, TestConvScaleNode) {
  auto meta_graph = BuildGraph(schema::PrimitiveType_Conv2DFusion, true);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
  delete anf_transform;
}

TEST_F(ConvScaleFusionTest, TestDeptiwiseConvScaleNode) {
  auto meta_graph = BuildGraph(schema::PrimitiveType_Conv2DFusion, false);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
  for (auto &cnode : new_meta_graph->nodes) {
    ASSERT_EQ(cnode->inputIndex.size(), 3);
  }
  delete anf_transform;
}
}  // namespace mindspore
