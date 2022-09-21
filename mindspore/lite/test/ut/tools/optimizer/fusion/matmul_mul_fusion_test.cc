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
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace {
constexpr int kMatMulInputDimsM = 128;
constexpr int kMatMulInputDimsK = 225;
constexpr int kMatMulInputDimsN = 98;
}  // namespace
class MatMulAddFusionTest : public mindspore::CommonTest {
 public:
  MatMulAddFusionTest() = default;
};
using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

namespace {
CNodeTptr BuildMatMul() {
  auto matmul_node = std::make_unique<schema::CNodeT>();
  matmul_node->inputIndex = {0, 1, opt::kInputIndexTwo};
  matmul_node->outputIndex = {opt::kInputIndexThree};
  matmul_node->primitive = std::make_unique<schema::PrimitiveT>();
  matmul_node->primitive->value.type = schema::PrimitiveType_MatMulFusion;
  auto prim1 = new schema::MatMulFusionT;
  prim1->transpose_a = false;
  prim1->transpose_b = false;
  matmul_node->primitive->value.value = prim1;
  matmul_node->name = "MatMul";
  return matmul_node;
}
CNodeTptr BuildMul() {
  auto mul_node = std::make_unique<schema::CNodeT>();
  mul_node->inputIndex = {opt::kInputIndexThree, opt::kInputIndexFour};
  mul_node->outputIndex = {opt::kInputIndexFive};
  mul_node->primitive = std::make_unique<schema::PrimitiveT>();
  mul_node->primitive->value.type = schema::PrimitiveType_MulFusion;
  auto prim1 = new schema::MulFusionT;
  prim1->activation_type = mindspore::schema::ActivationType_NO_ACTIVATION;
  mul_node->primitive->value.value = prim1;
  mul_node->name = "Mul";
  return mul_node;
}

MetaGraphTptr BuildGraph() {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";
  // create graph
  CNodeTptr matmul_node = BuildMatMul();
  meta_graph->nodes.emplace_back(std::move(matmul_node));
  auto mul_node = BuildMul();
  meta_graph->nodes.emplace_back(std::move(mul_node));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {opt::kInputIndexFour};

  // input 0:
  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {kMatMulInputDimsM, kMatMulInputDimsK};
  input0->data.resize(sizeof(float) * kMatMulInputDimsM * kMatMulInputDimsK);
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  // input 1:
  auto input1 = std::make_unique<schema::TensorT>();
  input1->nodeType = lite::NodeType_ValueNode;
  input1->format = schema::Format_KHWC;
  input1->dataType = TypeId::kNumberTypeFloat32;
  input1->dims = {kMatMulInputDimsK, kMatMulInputDimsN};
  input1->data.resize(sizeof(float) * kMatMulInputDimsK * kMatMulInputDimsN);
  meta_graph->allTensors.emplace_back(std::move(input1));

  // bias:
  auto input2 = std::make_unique<schema::TensorT>();
  input2->nodeType = lite::NodeType_ValueNode;
  input2->format = schema::Format_KHWC;
  input2->dataType = TypeId::kNumberTypeFloat32;
  input2->dims = {kMatMulInputDimsN};
  input2->data.resize(sizeof(float) * kMatMulInputDimsN);
  meta_graph->allTensors.emplace_back(std::move(input2));

  // matmul output
  auto matmul_output = std::make_unique<schema::TensorT>();
  matmul_output->nodeType = lite::NodeType_Parameter;
  matmul_output->format = schema::Format_NHWC;
  matmul_output->dataType = TypeId::kNumberTypeFloat32;
  matmul_output->dims = {kMatMulInputDimsM, kMatMulInputDimsN};
  meta_graph->allTensors.emplace_back(std::move(matmul_output));

  // mul weight
  auto input3 = std::make_unique<schema::TensorT>();
  input3->nodeType = lite::NodeType_ValueNode;
  input3->format = schema::Format_NHWC;
  input3->dataType = TypeId::kNumberTypeFloat32;
  input3->dims = {kMatMulInputDimsN};
  input3->data.resize(sizeof(float) * kMatMulInputDimsN);
  meta_graph->allTensors.emplace_back(std::move(input3));

  // final output
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = {kMatMulInputDimsM, kMatMulInputDimsN};
  meta_graph->allTensors.emplace_back(std::move(output));
  return meta_graph;
}
}  //  namespace
TEST_F(MatMulAddFusionTest, TestMatMulMulNode) {
  auto meta_graph = BuildGraph();
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph, nullptr);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
  MS_LOG(INFO) << "Passed";
}
}  // namespace mindspore
