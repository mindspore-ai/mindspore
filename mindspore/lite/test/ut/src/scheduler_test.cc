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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/common_test.h"
#include "schema/inner/model_generated.h"
#include "src/lite_session.h"
#include "ir/dtype/type_id.h"
#include "include/version.h"

using mindspore::kernel::KernelKey;
using mindspore::kernel::LiteKernel;
using mindspore::lite::InnerContext;
using mindspore::lite::LiteSession;
using mindspore::lite::Tensor;
using mindspore::schema::PrimitiveType_Abs;
using mindspore::TypeId::kNumberTypeFloat32;

class SchedulerTest : public mindspore::CommonTest {
 public:
  SchedulerTest() = default;
};

TEST_F(SchedulerTest, TestConstructSubGraphsTwoBranch) {
  auto meta_graph = std::make_shared<mindspore::schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->version = mindspore::lite::Version();

  auto split = std::make_unique<mindspore::schema::CNodeT>();
  split->inputIndex = {0};
  split->outputIndex = {1, 2};
  split->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  split->primitive->value.type = mindspore::schema::PrimitiveType_Split;
  auto primitive = new mindspore::schema::SplitT;
  primitive->output_num = 2;
  primitive->axis = 3;
  split->primitive->value.value = primitive;
  split->name = "split";

  auto abs1 = std::make_unique<mindspore::schema::CNodeT>();
  abs1->inputIndex = {1};
  abs1->outputIndex = {3};
  abs1->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  abs1->primitive->value.type = mindspore::schema::PrimitiveType_Abs;
  auto abs1_primitive = new mindspore::schema::AbsT;
  abs1->primitive->value.value = abs1_primitive;
  abs1->name = "gpu1";

  auto cons1 = std::make_unique<mindspore::schema::CNodeT>();
  cons1->inputIndex = {2};
  cons1->outputIndex = {4};
  cons1->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cons1->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cons1_primitive = new mindspore::schema::CosT;
  cons1->primitive->value.value = cons1_primitive;
  cons1->name = "cpu1";

  auto abs2 = std::make_unique<mindspore::schema::CNodeT>();
  abs2->inputIndex = {3};
  abs2->outputIndex = {5};
  abs2->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  abs2->primitive->value.type = mindspore::schema::PrimitiveType_Abs;
  auto abs2_primitive = new mindspore::schema::AbsT;
  abs2->primitive->value.value = abs2_primitive;
  abs2->name = "gpu2";

  auto cons2 = std::make_unique<mindspore::schema::CNodeT>();
  cons2->inputIndex = {4};
  cons2->outputIndex = {6};
  cons2->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cons2->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cons2_primitive = new mindspore::schema::CosT;
  cons2->primitive->value.value = cons2_primitive;
  cons2->name = "cpu2";

  auto concat = std::make_unique<mindspore::schema::CNodeT>();
  concat->inputIndex = {5, 6};
  concat->outputIndex = {7};
  concat->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  concat->primitive->value.type = mindspore::schema::PrimitiveType_Concat;
  auto concat_primitive = new mindspore::schema::ConcatT;
  concat_primitive->axis = 3;
  concat->primitive->value.value = concat_primitive;
  concat->name = "concat";

  auto tensor0 = std::make_unique<mindspore::schema::TensorT>();
  tensor0->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor0->format = mindspore::schema::Format_NHWC;
  tensor0->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor0->dims = {1, 16, 16, 4};
  tensor0->offset = -1;
  auto tensor1 = std::make_unique<mindspore::schema::TensorT>();
  tensor1->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor1->format = mindspore::schema::Format_NHWC;
  tensor1->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor1->dims = {1, 16, 16, 2};
  tensor1->offset = -1;
  auto tensor2 = std::make_unique<mindspore::schema::TensorT>();
  tensor2->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor2->format = mindspore::schema::Format_NHWC;
  tensor2->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor2->dims = {1, 16, 16, 2};
  tensor2->offset = -1;
  auto tensor3 = std::make_unique<mindspore::schema::TensorT>();
  tensor3->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor3->format = mindspore::schema::Format_NHWC;
  tensor3->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor3->dims = {1, 16, 16, 2};
  tensor3->offset = -1;
  auto tensor4 = std::make_unique<mindspore::schema::TensorT>();
  tensor4->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor4->format = mindspore::schema::Format_NHWC;
  tensor4->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor4->dims = {1, 16, 16, 2};
  tensor4->offset = -1;
  auto tensor5 = std::make_unique<mindspore::schema::TensorT>();
  tensor5->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor5->format = mindspore::schema::Format_NHWC;
  tensor5->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor5->dims = {1, 16, 16, 2};
  tensor5->offset = -1;
  auto tensor6 = std::make_unique<mindspore::schema::TensorT>();
  tensor6->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor6->format = mindspore::schema::Format_NHWC;
  tensor6->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor6->dims = {1, 16, 16, 2};
  tensor6->offset = -1;
  auto tensor7 = std::make_unique<mindspore::schema::TensorT>();
  tensor7->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor7->format = mindspore::schema::Format_NHWC;
  tensor7->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor7->dims = {1, 16, 16, 4};
  tensor7->offset = -1;

  meta_graph->nodes.emplace_back(std::move(split));
  meta_graph->nodes.emplace_back(std::move(abs1));
  meta_graph->nodes.emplace_back(std::move(cons1));
  meta_graph->nodes.emplace_back(std::move(abs2));
  meta_graph->nodes.emplace_back(std::move(cons2));
  meta_graph->nodes.emplace_back(std::move(concat));
  meta_graph->allTensors.emplace_back(std::move(tensor0));
  meta_graph->allTensors.emplace_back(std::move(tensor1));
  meta_graph->allTensors.emplace_back(std::move(tensor2));
  meta_graph->allTensors.emplace_back(std::move(tensor3));
  meta_graph->allTensors.emplace_back(std::move(tensor4));
  meta_graph->allTensors.emplace_back(std::move(tensor5));
  meta_graph->allTensors.emplace_back(std::move(tensor6));
  meta_graph->allTensors.emplace_back(std::move(tensor7));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {7};
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  mindspore::schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());
  auto model = mindspore::lite::Model::Import(content, size);
  auto context = new InnerContext();
  context->Init();
  mindspore::lite::DeviceContext gpu_device_ctx = {mindspore::lite::DT_GPU, {false}};
  context->device_list_.emplace_back(gpu_device_ctx);
  auto lite_session = new LiteSession();
  lite_session->Init(context);
  ASSERT_EQ(mindspore::lite::RET_OK, lite_session->CompileGraph(model));
}

TEST_F(SchedulerTest, TestConstructSubGraphsThreeBranch) {
  auto meta_graph = std::make_shared<mindspore::schema::MetaGraphT>();
  meta_graph->name = "graph";
  meta_graph->version = mindspore::lite::Version();

  auto split = std::make_unique<mindspore::schema::CNodeT>();
  split->inputIndex = {0};
  split->outputIndex = {1, 2, 3};
  split->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  split->primitive->value.type = mindspore::schema::PrimitiveType_Split;
  auto primitive = new mindspore::schema::SplitT;
  primitive->output_num = 3;
  primitive->axis = 3;
  split->primitive->value.value = primitive;
  split->name = "split";

  auto abs1 = std::make_unique<mindspore::schema::CNodeT>();
  abs1->inputIndex = {1};
  abs1->outputIndex = {4};
  abs1->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  abs1->primitive->value.type = mindspore::schema::PrimitiveType_Abs;
  auto abs1_primitive = new mindspore::schema::AbsT;
  abs1->primitive->value.value = abs1_primitive;
  abs1->name = "gpu1";

  auto abs2 = std::make_unique<mindspore::schema::CNodeT>();
  abs2->inputIndex = {2};
  abs2->outputIndex = {5};
  abs2->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  abs2->primitive->value.type = mindspore::schema::PrimitiveType_Abs;
  auto abs2_primitive = new mindspore::schema::AbsT;
  abs2->primitive->value.value = abs2_primitive;
  abs2->name = "gpu2";

  auto cons1 = std::make_unique<mindspore::schema::CNodeT>();
  cons1->inputIndex = {3};
  cons1->outputIndex = {6};
  cons1->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cons1->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cons1_primitive = new mindspore::schema::CosT;
  cons1->primitive->value.value = cons1_primitive;
  cons1->name = "cpu1";

  auto abs3 = std::make_unique<mindspore::schema::CNodeT>();
  abs3->inputIndex = {4};
  abs3->outputIndex = {7};
  abs3->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  abs3->primitive->value.type = mindspore::schema::PrimitiveType_Abs;
  auto abs3_primitive = new mindspore::schema::AbsT;
  abs3->primitive->value.value = abs3_primitive;
  abs3->name = "gpu3";

  auto abs4 = std::make_unique<mindspore::schema::CNodeT>();
  abs4->inputIndex = {5};
  abs4->outputIndex = {8};
  abs4->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  abs4->primitive->value.type = mindspore::schema::PrimitiveType_Abs;
  auto abs4_primitive = new mindspore::schema::AbsT;
  abs4->primitive->value.value = abs4_primitive;
  abs4->name = "gpu4";

  auto cons2 = std::make_unique<mindspore::schema::CNodeT>();
  cons2->inputIndex = {6};
  cons2->outputIndex = {9};
  cons2->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  cons2->primitive->value.type = mindspore::schema::PrimitiveType_Cos;
  auto cons2_primitive = new mindspore::schema::CosT;
  cons2->primitive->value.value = cons2_primitive;
  cons2->name = "cpu2";

  auto concat = std::make_unique<mindspore::schema::CNodeT>();
  concat->inputIndex = {7, 8, 8};
  concat->outputIndex = {10};
  concat->primitive = std::make_unique<mindspore::schema::PrimitiveT>();
  concat->primitive->value.type = mindspore::schema::PrimitiveType_Concat;
  auto concat_primitive = new mindspore::schema::ConcatT;
  concat_primitive->axis = 3;
  concat->primitive->value.value = concat_primitive;
  concat->name = "concat";

  auto tensor0 = std::make_unique<mindspore::schema::TensorT>();
  tensor0->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor0->format = mindspore::schema::Format_NHWC;
  tensor0->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor0->dims = {1, 16, 16, 3};
  tensor0->offset = -1;
  auto tensor1 = std::make_unique<mindspore::schema::TensorT>();
  tensor1->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor1->format = mindspore::schema::Format_NHWC;
  tensor1->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor1->dims = {1, 16, 16, 1};
  tensor1->offset = -1;
  auto tensor2 = std::make_unique<mindspore::schema::TensorT>();
  tensor2->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor2->format = mindspore::schema::Format_NHWC;
  tensor2->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor2->dims = {1, 16, 16, 1};
  tensor2->offset = -1;
  auto tensor3 = std::make_unique<mindspore::schema::TensorT>();
  tensor3->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor3->format = mindspore::schema::Format_NHWC;
  tensor3->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor3->dims = {1, 16, 16, 1};
  tensor3->offset = -1;
  auto tensor4 = std::make_unique<mindspore::schema::TensorT>();
  tensor4->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor4->format = mindspore::schema::Format_NHWC;
  tensor4->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor4->dims = {1, 16, 16, 1};
  tensor4->offset = -1;
  auto tensor5 = std::make_unique<mindspore::schema::TensorT>();
  tensor5->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor5->format = mindspore::schema::Format_NHWC;
  tensor5->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor5->dims = {1, 16, 16, 1};
  tensor5->offset = -1;
  auto tensor6 = std::make_unique<mindspore::schema::TensorT>();
  tensor6->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor6->format = mindspore::schema::Format_NHWC;
  tensor6->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor6->dims = {1, 16, 16, 1};
  tensor6->offset = -1;
  auto tensor7 = std::make_unique<mindspore::schema::TensorT>();
  tensor7->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor7->format = mindspore::schema::Format_NHWC;
  tensor7->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor7->dims = {1, 16, 16, 1};
  tensor7->offset = -1;
  auto tensor8 = std::make_unique<mindspore::schema::TensorT>();
  tensor8->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor8->format = mindspore::schema::Format_NHWC;
  tensor8->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor8->dims = {1, 16, 16, 1};
  tensor8->offset = -1;
  auto tensor9 = std::make_unique<mindspore::schema::TensorT>();
  tensor9->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor9->format = mindspore::schema::Format_NHWC;
  tensor9->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor9->dims = {1, 16, 16, 1};
  tensor9->offset = -1;
  auto tensor10 = std::make_unique<mindspore::schema::TensorT>();
  tensor10->nodeType = mindspore::lite::NodeType_ValueNode;
  tensor10->format = mindspore::schema::Format_NHWC;
  tensor10->dataType = mindspore::TypeId::kNumberTypeFloat32;
  tensor10->dims = {1, 16, 16, 3};
  tensor10->offset = -1;

  meta_graph->nodes.emplace_back(std::move(split));
  meta_graph->nodes.emplace_back(std::move(abs1));
  meta_graph->nodes.emplace_back(std::move(abs2));
  meta_graph->nodes.emplace_back(std::move(cons1));
  meta_graph->nodes.emplace_back(std::move(abs3));
  meta_graph->nodes.emplace_back(std::move(abs4));
  meta_graph->nodes.emplace_back(std::move(cons2));
  meta_graph->nodes.emplace_back(std::move(concat));
  meta_graph->allTensors.emplace_back(std::move(tensor0));
  meta_graph->allTensors.emplace_back(std::move(tensor1));
  meta_graph->allTensors.emplace_back(std::move(tensor2));
  meta_graph->allTensors.emplace_back(std::move(tensor3));
  meta_graph->allTensors.emplace_back(std::move(tensor4));
  meta_graph->allTensors.emplace_back(std::move(tensor5));
  meta_graph->allTensors.emplace_back(std::move(tensor6));
  meta_graph->allTensors.emplace_back(std::move(tensor7));
  meta_graph->allTensors.emplace_back(std::move(tensor8));
  meta_graph->allTensors.emplace_back(std::move(tensor9));
  meta_graph->allTensors.emplace_back(std::move(tensor10));
  meta_graph->inputIndex = {0};
  meta_graph->outputIndex = {10};
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  mindspore::schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());
  auto model = mindspore::lite::Model::Import(content, size);
  auto context = new InnerContext();
  context->Init();
  mindspore::lite::DeviceContext gpu_device_ctx = {mindspore::lite::DT_GPU, {false}};
  context->device_list_.emplace_back(gpu_device_ctx);
  auto lite_session = new LiteSession();
  lite_session->Init(context);
  ASSERT_EQ(mindspore::lite::RET_OK, lite_session->CompileGraph(model));
}
