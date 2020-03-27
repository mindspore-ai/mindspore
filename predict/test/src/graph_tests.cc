/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <gtest/gtest.h>
#include <cstdio>
#include <string>
#include "schema/inner/ms_generated.h"
#include "src/graph.h"
#include "common/file_utils.h"
#include "test/test_context.h"
#include "include/session.h"

namespace mindspore {
namespace predict {
class GraphTest : public ::testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  std::string root;
};

void InitMsGraphAllTensor(SubGraphDefT *msSubgraph) {
  ASSERT_NE(msSubgraph, nullptr);
  std::unique_ptr<TensorDefT> tensor (new (std::nothrow) TensorDefT);
  ASSERT_NE(tensor, nullptr);
  tensor->refCount = MSConst_WEIGHT_REFCOUNT;
  tensor->format = Format_NCHW;
  tensor->dataType = DataType_DT_FLOAT;
  tensor->dims = {1, 1, 1, 2};
  tensor->offset = -1;
  tensor->data.resize(0);
  msSubgraph->allTensors.emplace_back(std::move(tensor));

  std::unique_ptr<TensorDefT> tensor2(new (std::nothrow) TensorDefT);
  ASSERT_NE(tensor2, nullptr);
  tensor2->refCount = MSConst_WEIGHT_REFCOUNT;
  tensor2->format = Format_NCHW;
  tensor2->dataType = DataType_DT_FLOAT;
  tensor2->dims = {1, 1, 1, 2};
  tensor2->offset = -1;
  tensor2->data.resize(0);
  msSubgraph->allTensors.emplace_back(std::move(tensor2));

  std::unique_ptr<TensorDefT> tensor3(new (std::nothrow) TensorDefT);
  ASSERT_NE(tensor3, nullptr);
  tensor3->refCount = 0;
  tensor3->format = Format_NCHW;
  tensor3->dataType = DataType_DT_FLOAT;
  tensor3->dims = {1, 1, 1, 2};
  tensor3->offset = -1;
  tensor3->data.resize(0);
  msSubgraph->allTensors.emplace_back(std::move(tensor3));
}

void FreeOutputs(std::map<std::string, std::vector<Tensor *>> *outputs) {
  for (auto &output : (*outputs)) {
    for (auto &outputTensor : output.second) {
      delete outputTensor;
    }
  }
  outputs->clear();
}

void FreeInputs(std::vector<Tensor *> *inputs) {
  for (auto &input : *inputs) {
    input->SetData(nullptr);
    delete input;
  }
  inputs->clear();
  return;
}

TEST_F(GraphTest, CreateFromFileAdd) {
  auto msGraph = std::unique_ptr<GraphDefT>(new (std::nothrow) GraphDefT());
  ASSERT_NE(msGraph, nullptr);
  msGraph->name = "test1";
  auto msSubgraph = std::unique_ptr<SubGraphDefT>(new (std::nothrow) SubGraphDefT());
  ASSERT_NE(msSubgraph, nullptr);
  msSubgraph->name = msGraph->name + "_1";
  msSubgraph->inputIndex = {0, 1};
  msSubgraph->outputIndex = {2};

  std::unique_ptr<NodeDefT> node(new (std::nothrow) NodeDefT);
  ASSERT_NE(node, nullptr);
  std::unique_ptr<OpDefT> opDef(new (std::nothrow) OpDefT);
  ASSERT_NE(opDef, nullptr);
  node->opDef = std::move(opDef);
  node->opDef->isLastConv = false;
  node->opDef->inputIndex = {static_cast<unsigned int>(0), 1};
  node->opDef->outputIndex = {static_cast<unsigned int>(2)};
  node->opDef->name = msSubgraph->name + std::to_string(0);
  node->fmkType = FmkType_CAFFE;

  auto attr = std::unique_ptr<AddT>(new (std::nothrow) AddT());
  ASSERT_NE(attr, nullptr);
  attr->format = DataFormatType_NCHW;
  node->opDef->attr.type = OpT_Add;
  node->opDef->attr.value = attr.release();

  msSubgraph->nodes.emplace_back(std::move(node));

  InitMsGraphAllTensor(msSubgraph.get());
  msGraph->subgraphs.emplace_back(std::move(msSubgraph));

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = mindspore::predict::GraphDef::Pack(builder, msGraph.get());
  builder.Finish(offset);
  int size = builder.GetSize();
  void *content = builder.GetBufferPointer();

  Context ctx;
  auto session = CreateSession(static_cast<char *>(content), size, ctx);

  std::vector<float> tmpT = {1, 2};
  void *in1Data = tmpT.data();
  std::vector<float> tmpT2 = {3, 5};
  void *in2Data = tmpT2.data();

  auto inputs = session->GetInput();
  inputs[0]->SetData(in1Data);
  inputs[1]->SetData(in2Data);

  auto ret = session->Run(inputs);
  EXPECT_EQ(0, ret);
  auto outputs = session->GetAllOutput();
  EXPECT_EQ(4, reinterpret_cast<float *>(outputs.begin()->second.front()->GetData())[0]);
  EXPECT_EQ(7, reinterpret_cast<float *>(outputs.begin()->second.front()->GetData())[1]);

  FreeOutputs(&outputs);
  FreeInputs(&inputs);
}
}  // namespace predict
}  // namespace mindspore
