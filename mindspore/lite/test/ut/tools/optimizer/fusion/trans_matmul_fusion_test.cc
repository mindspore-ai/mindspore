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
#include <vector>
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/anf_transform.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "test/common/import_from_meta_graphT.h"

namespace mindspore {
class TransMatMulFusionTest : public mindspore::CommonTest {
 public:
  TransMatMulFusionTest() = default;
};
using MetaGraphTptr = std::shared_ptr<schema::MetaGraphT>;
using CNodeTptr = std::unique_ptr<schema::CNodeT>;

namespace {
constexpr int kTransMatMulDim1 = 2;
constexpr int kTransMatMulDim2 = 3;
constexpr int kTransMatMulDim3 = 4;
inline const std::vector<int> kMatMulTransPerm1 = {0, 1, 3, 2};
inline const std::vector<int> kMatMulTransPerm2 = {0, 2, 1};
inline const std::vector<int> kInvalidPerm1 = {2, 1, 0};

struct TransInParam {
  std::vector<int32_t> dims;
  std::vector<int> perm;
  bool need_transpose;
};

void SetTransTensors(std::shared_ptr<schema::MetaGraphT> meta_graph, const std::vector<int> &perm,
                     const std::vector<int> &dims) {
  // input_0
  auto input_0 = std::make_unique<schema::TensorT>();
  input_0->nodeType = lite::NodeType_Parameter;
  input_0->format = schema::Format_NHWC;
  input_0->dataType = TypeId::kNumberTypeFloat32;
  input_0->dims = dims;
  input_0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input_0));

  // input_1
  auto input_1 = std::make_unique<schema::TensorT>();
  input_1->nodeType = lite::NodeType_ValueNode;
  input_1->format = schema::Format_NHWC;
  input_1->dataType = TypeId::kNumberTypeFloat32;
  input_1->dims = {static_cast<int>(perm.size())};
  input_1->data.resize(sizeof(int) * perm.size());
  memcpy(input_1->data.data(), perm.data(), sizeof(int) * perm.size());
  meta_graph->allTensors.emplace_back(std::move(input_1));

  // trans output
  auto conv_output = std::make_unique<schema::TensorT>();
  conv_output->nodeType = lite::NodeType_Parameter;
  conv_output->format = schema::Format_NHWC;
  conv_output->dataType = TypeId::kNumberTypeFloat32;
  conv_output->dims = dims;
  for (size_t i = 0; i < perm.size(); i++) {
    conv_output->dims.at(i) = dims.at(perm.at(i));
  }
  meta_graph->allTensors.emplace_back(std::move(conv_output));
  return;
}

MetaGraphTptr BuildGraph(const TransInParam &trans_param_a, const TransInParam &trans_param_b,
                         const std::vector<int> &output_dims) {
  auto perm_a = trans_param_a.perm;
  auto dims_a = trans_param_a.dims;
  auto transpose_a = trans_param_a.need_transpose;
  auto perm_b = trans_param_b.perm;
  auto dims_b = trans_param_b.dims;
  auto transpose_b = trans_param_b.need_transpose;

  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  // trans node a
  auto trans_node_a = std::make_unique<schema::CNodeT>();
  trans_node_a->inputIndex = {0, 1};
  trans_node_a->outputIndex = {2};
  trans_node_a->primitive = std::make_unique<schema::PrimitiveT>();
  trans_node_a->primitive->value.type = schema::PrimitiveType_Transpose;
  auto prim1 = new schema::TransposeT;
  trans_node_a->primitive->value.value = prim1;
  trans_node_a->name = "transpose_a";
  meta_graph->nodes.emplace_back(std::move(trans_node_a));

  // trans node b
  auto trans_node_b = std::make_unique<schema::CNodeT>();
  trans_node_b->inputIndex = {3, 4};
  trans_node_b->outputIndex = {5};
  trans_node_b->primitive = std::make_unique<schema::PrimitiveT>();
  trans_node_b->primitive->value.type = schema::PrimitiveType_Transpose;
  auto prim2 = new schema::TransposeT;
  trans_node_b->primitive->value.value = prim2;
  trans_node_b->name = "transpose_b";
  meta_graph->nodes.emplace_back(std::move(trans_node_b));

  // matmul node
  auto matmul_node = std::make_unique<schema::CNodeT>();
  matmul_node->inputIndex = {2, 5};
  matmul_node->outputIndex = {6};
  matmul_node->primitive = std::make_unique<schema::PrimitiveT>();
  matmul_node->primitive->value.type = schema::PrimitiveType_MatMulFusion;
  auto prim3 = new schema::MatMulFusionT;
  prim3->transpose_a = transpose_a;
  prim3->transpose_b = transpose_b;
  matmul_node->primitive->value.value = prim3;
  matmul_node->name = "matmul";
  meta_graph->nodes.emplace_back(std::move(matmul_node));

  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {6};

  SetTransTensors(meta_graph, perm_a, dims_a);
  SetTransTensors(meta_graph, perm_b, dims_b);

  // final output
  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->dims = output_dims;
  meta_graph->allTensors.emplace_back(std::move(output));
  return meta_graph;
}
}  //  namespace
TEST_F(TransMatMulFusionTest, TestTransMatMulNode1) {
  TransInParam trans_param_a;
  TransInParam trans_param_b;
  trans_param_a.dims = {1, kTransMatMulDim1, kTransMatMulDim2, kTransMatMulDim3};
  trans_param_a.perm = kMatMulTransPerm1;
  trans_param_a.need_transpose = false;
  trans_param_b.dims = {1, kTransMatMulDim1, kTransMatMulDim3, kTransMatMulDim2};
  trans_param_b.perm = kMatMulTransPerm1;
  trans_param_b.need_transpose = false;
  std::vector<int> output_dims = {1, kTransMatMulDim1, kTransMatMulDim3, kTransMatMulDim3};
  auto meta_graph = BuildGraph(trans_param_a, trans_param_b, output_dims);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph, nullptr);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
  auto &cnode = new_meta_graph->nodes.at(0);
  ASSERT_EQ(cnode->primitive->value.AsMatMulFusion()->transpose_a, true);
  ASSERT_EQ(cnode->primitive->value.AsMatMulFusion()->transpose_b, true);
}

TEST_F(TransMatMulFusionTest, TestTransMatMulNode2) {
  TransInParam trans_param_a;
  TransInParam trans_param_b;
  trans_param_a.dims = {1, kTransMatMulDim1, kTransMatMulDim2};
  trans_param_a.perm = kMatMulTransPerm2;
  trans_param_a.need_transpose = true;
  trans_param_b.dims = {1, kTransMatMulDim2, kTransMatMulDim1};
  trans_param_b.perm = kMatMulTransPerm2;
  trans_param_b.need_transpose = true;
  std::vector<int> output_dims = {1, kTransMatMulDim1, kTransMatMulDim1};
  auto meta_graph = BuildGraph(trans_param_a, trans_param_b, output_dims);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph, nullptr);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 1);
  auto &cnode = new_meta_graph->nodes.at(0);
  ASSERT_EQ(cnode->primitive->value.AsMatMulFusion()->transpose_a, false);
  ASSERT_EQ(cnode->primitive->value.AsMatMulFusion()->transpose_b, false);
}

TEST_F(TransMatMulFusionTest, TestBadCase_TransMatMul) {
  TransInParam trans_param_a;
  TransInParam trans_param_b;
  trans_param_a.dims = {kTransMatMulDim2, kTransMatMulDim1, kTransMatMulDim2};
  trans_param_a.perm = kInvalidPerm1;
  trans_param_a.need_transpose = false;
  trans_param_b.dims = {kTransMatMulDim1, kTransMatMulDim2, kTransMatMulDim2};
  trans_param_b.perm = kInvalidPerm1;
  trans_param_b.need_transpose = false;
  std::vector<int> output_dims = {kTransMatMulDim2, kTransMatMulDim1, kTransMatMulDim1};
  auto meta_graph = BuildGraph(trans_param_a, trans_param_b, output_dims);
  auto func_graph = lite::AnfImporterFromMetaGraphT::Fb2Anf(meta_graph.get());
  auto anf_transform = new lite::AnfTransform();
  auto new_graph = anf_transform->Transform(func_graph, nullptr);
  ASSERT_NE(nullptr, new_graph);
  auto new_meta_graph = lite::Export(new_graph);
  ASSERT_EQ(new_meta_graph->nodes.size(), 3);
  auto &cnode = new_meta_graph->nodes.at(2);
  ASSERT_EQ(cnode->primitive->value.AsMatMulFusion()->transpose_a, false);
  ASSERT_EQ(cnode->primitive->value.AsMatMulFusion()->transpose_b, false);
}
}  // namespace mindspore
