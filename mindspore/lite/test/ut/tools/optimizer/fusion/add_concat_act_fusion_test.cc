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
#include "tools/optimizer/common/gllo_utils.h"
#include "test/common/import_from_meta_graphT.h"

namespace mindspore {
constexpr size_t kAddInputTensorWSize = 128;
constexpr size_t kConcatInputTensorWDims = 256;
constexpr size_t kGraphNodeSize = 3;
class AddConcatActivationFusionTest : public mindspore::CommonTest {
 public:
  AddConcatActivationFusionTest() = default;
};
using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

namespace {
CNodeTptr BuildAdd(const string &name, std::vector<uint32_t> input_index, std::vector<uint32_t> output_index) {
  auto add_node = std::make_unique<schema::CNodeT>();
  add_node->inputIndex = input_index;
  add_node->outputIndex = output_index;
  add_node->primitive = std::make_unique<schema::PrimitiveT>();
  add_node->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto prim1 = new schema::AddFusionT;
  prim1->activation_type = mindspore::schema::ActivationType_NO_ACTIVATION;
  add_node->primitive->value.value = prim1;
  add_node->name = name;
  return add_node;
}

CNodeTptr BuildConcat() {
  auto concat_node = std::make_unique<schema::CNodeT>();
  concat_node->inputIndex = {opt::kInputIndexTwo, opt::kInputIndexFive};
  concat_node->outputIndex = {opt::kInputIndexSix};
  concat_node->primitive = std::make_unique<schema::PrimitiveT>();
  concat_node->primitive->value.type = schema::PrimitiveType_Concat;
  auto prim1 = new schema::ConcatT;
  prim1->axis = 1;
  concat_node->primitive->value.value = prim1;
  concat_node->name = "Concat";
  return concat_node;
}

void BuildTensorT(const std::unique_ptr<schema::TensorT> &input, std::vector<int32_t> shape) {
  input->nodeType = lite::NodeType_Parameter;
  input->format = schema::Format_NHWC;
  input->dataType = TypeId::kNumberTypeFloat32;
  input->dims = shape;
  return;
}

MetaGraphTptr BuildGraph(schema::ActivationType activation_type) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  // add node1
  std::vector<uint32_t> input_index{0, 1};
  std::vector<uint32_t> output_index{opt::kInputIndexTwo};
  auto add_node1 = BuildAdd("add_node1", input_index, output_index);
  meta_graph->nodes.emplace_back(std::move(add_node1));

  // add node2
  input_index = std::vector<uint32_t>({opt::kInputIndexThree, opt::kInputIndexFour});
  output_index = std::vector<uint32_t>({opt::kInputIndexFive});
  auto add_node2 = BuildAdd("add_node2", input_index, output_index);
  meta_graph->nodes.emplace_back(std::move(add_node2));

  // concat node
  auto caoncat_node = BuildConcat();
  meta_graph->nodes.emplace_back(std::move(caoncat_node));

  // relu node
  auto next_node = std::make_unique<schema::CNodeT>();
  next_node->inputIndex = {opt::kInputIndexSix};
  next_node->outputIndex = {opt::kInputIndexSeven};
  next_node->primitive = std::make_unique<schema::PrimitiveT>();
  next_node->primitive->value.type = schema::PrimitiveType_Activation;
  auto prim4 = new schema::ActivationT;
  prim4->activation_type = activation_type;
  next_node->primitive->value.value = prim4;
  next_node->name = "activation";
  meta_graph->nodes.emplace_back(std::move(next_node));

  meta_graph->inputIndex = {0, 1, opt::kInputIndexThree, opt::kInputIndexFour};
  meta_graph->outputIndex = {opt::kInputIndexSeven};

  // input 0: data
  auto input0 = std::make_unique<schema::TensorT>();
  BuildTensorT(input0, {1, kAddInputTensorWSize});
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // input 1: data
  auto input2 = std::make_unique<schema::TensorT>();
  BuildTensorT(input2, {1, kAddInputTensorWSize});
  input2->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input2));

  // output 1: data
  auto add_node1_out = std::make_unique<schema::TensorT>();
  BuildTensorT(add_node1_out, {1, kAddInputTensorWSize});
  meta_graph->allTensors.emplace_back(std::move(add_node1_out));

  // input 2: data
  auto add_node2_input1 = std::make_unique<schema::TensorT>();
  BuildTensorT(add_node2_input1, {1, kAddInputTensorWSize});
  add_node2_input1->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(add_node2_input1));

  // input 3: data
  auto add_node2_input2 = std::make_unique<schema::TensorT>();
  BuildTensorT(add_node2_input2, {1, kAddInputTensorWSize});
  add_node2_input2->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(add_node2_input2));

  // output 2: data
  auto add_node2_out = std::make_unique<schema::TensorT>();
  BuildTensorT(add_node2_out, {1, kAddInputTensorWSize});
  meta_graph->allTensors.emplace_back(std::move(add_node2_out));

  // concat output
  auto concat_output = std::make_unique<schema::TensorT>();
  BuildTensorT(concat_output, {1, kConcatInputTensorWDims});
  meta_graph->allTensors.emplace_back(std::move(concat_output));

  // final output
  auto output = std::make_unique<schema::TensorT>();
  BuildTensorT(output, {1, kConcatInputTensorWDims});
  meta_graph->allTensors.emplace_back(std::move(output));
  return meta_graph;
}
}  //  namespace
TEST_F(AddConcatActivationFusionTest, TestAddConcatReluNode) {
  auto meta_graph = BuildGraph(schema::ActivationType_RELU6);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph, nullptr);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), kGraphNodeSize);
  for (auto &cnode : new_meta_graph->nodes) {
    if (cnode->primitive->value.type == schema::PrimitiveType_AddFusion) {
      ASSERT_EQ(cnode->primitive->value.AsAddFusion()->activation_type, schema::ActivationType_RELU6);
    }
  }
}
}  // namespace mindspore
