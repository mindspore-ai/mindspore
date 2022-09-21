/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either address or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/common_test.h"
#include "gmock/gmock.h"
#include "schema/inner/model_generated.h"
#include "src/litert/sub_graph_kernel.h"
#include "ir/dtype/type_id.h"
#include "include/model.h"
#include "src/litert/lite_session.h"

namespace mindspore {
namespace lite {
class SessionMock : public LiteSession {
 public:
  MOCK_METHOD0(RuntimeAllocatorValid, int());
};
}  // namespace lite

class OptimizeAllocator : public mindspore::CommonTest {
 public:
  OptimizeAllocator() = default;
};

void CreateModel1(mindspore::schema::MetaGraphT *meta_graph) {
  meta_graph->name = "graph";
  meta_graph->version = mindspore::Version();

  /*      cos
   *     /   \
   *   sin    |
   *     \   /
   *      add
   *       |
   * */

  auto cos = std::make_unique<mindspore::schema::CNodeT>();
  cos->inputIndex = {0};
  cos->outputIndex = {1};
  cos->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cos->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cos_primitive = new mindspore::schema::CosT;
  cos->primitive->value.value = cos_primitive;
  cos->name = "cos";

  auto sin = std::make_unique<mindspore::schema::CNodeT>();
  sin->inputIndex = {1};
  sin->outputIndex = {2};
  sin->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  sin->primitive->value.type = mindspore::schema::PrimitiveType_Sin;
  auto sin_primitive = new mindspore::schema::SinT;
  sin->primitive->value.value = sin_primitive;
  sin->name = "sin";

  auto add = std::make_unique<mindspore::schema::CNodeT>();
  add->inputIndex = {1, 2};
  add->outputIndex = {3};
  add->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  add->primitive->value.type = mindspore::schema::PrimitiveType_AddFusion;
  auto add_primitive = new mindspore::schema::AddFusionT;
  add->primitive->value.value = add_primitive;
  add->name = "add";

  /* tensors */
  auto tensor0 = std::make_unique<mindspore::schema::TensorT>();
  tensor0->nodeType = mindspore::lite::NodeType_Parameter;
  tensor0->format = mindspore::schema::Format_NHWC;
  tensor0->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor0->dims = {4};
  tensor0->offset = -1;
  tensor0->name = "input";

  auto tensor1 = std::make_unique<mindspore::schema::TensorT>();
  tensor1->nodeType = mindspore::lite::NodeType_Parameter;
  tensor1->format = mindspore::schema::Format_NHWC;
  tensor1->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor1->dims = {4};
  tensor1->offset = -1;
  tensor1->name = "cos";

  auto tensor2 = std::make_unique<mindspore::schema::TensorT>();
  tensor2->nodeType = mindspore::lite::NodeType_Parameter;
  tensor2->format = mindspore::schema::Format_NHWC;
  tensor2->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor2->dims = {4};
  tensor2->offset = -1;
  tensor2->name = "sin";

  auto tensor3 = std::make_unique<mindspore::schema::TensorT>();
  tensor3->nodeType = mindspore::lite::NodeType_Parameter;
  tensor3->format = mindspore::schema::Format_NHWC;
  tensor3->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor3->dims = {4};
  tensor3->offset = -1;
  tensor3->name = "add";

  meta_graph->nodes.emplace_back(std::move(cos));
  meta_graph->nodes.emplace_back(std::move(sin));
  meta_graph->nodes.emplace_back(std::move(add));

  meta_graph->allTensors.emplace_back(std::move(tensor0));
  meta_graph->allTensors.emplace_back(std::move(tensor1));
  meta_graph->allTensors.emplace_back(std::move(tensor2));
  meta_graph->allTensors.emplace_back(std::move(tensor3));

  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {3};
}

TEST_F(OptimizeAllocator, RuntimeAllocator1) {
  auto meta_graph = std::make_shared<mindspore::schema::MetaGraphT>();
  CreateModel1(meta_graph.get());

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  mindspore::schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());
  mindspore::lite::Model *model = mindspore::lite::Model::Import(content, size);

  auto context = std::make_shared<lite::InnerContext>();
  auto lite_session = new lite::SessionMock();
  ASSERT_NE(lite_session, nullptr);
  ON_CALL(*lite_session, RuntimeAllocatorValid).WillByDefault(testing::Return(0));

  auto ret = lite_session->Init(context);
  ASSERT_EQ(mindspore::lite::RET_OK, ret);

  ret = lite_session->CompileGraph(model);
  ASSERT_EQ(mindspore::lite::RET_OK, ret);
  ASSERT_NE(lite_session->get_kernels().front()->out_tensors().front()->allocator(), context->allocator);

  auto input = lite_session->GetInputs().front();
  std::vector<float> in_data = {1.0, 2.0, 3.0, 4.0};
  memcpy(input->MutableData(), in_data.data(), input->Size());

  ret = lite_session->RunGraph();
  ASSERT_EQ(mindspore::lite::RET_OK, ret);

  /* checkout output */
  void *out_data = lite_session->GetOutputs().begin()->second->MutableData();
  float *fp32_data = reinterpret_cast<float *>(out_data);

  ASSERT_LE(fabs(fp32_data[0] - (1.054698)), 0.01);
  ASSERT_LE(fabs(fp32_data[1] - (-0.820386)), 0.01);
  ASSERT_LE(fabs(fp32_data[2] - (-1.826014)), 0.01);
  ASSERT_LE(fabs(fp32_data[3] - (-1.261727)), 0.01);

  /* run loop */
  for (int i = 0; i < 2; i++) {
    ret = lite_session->RunGraph();
    ASSERT_EQ(mindspore::lite::RET_OK, ret);
  }

  delete lite_session;
}
}  // namespace mindspore
