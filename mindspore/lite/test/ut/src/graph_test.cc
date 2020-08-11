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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <climits>
#include <string>
#include <iostream>
#include <memory>
#include <fstream>
#include "common/common_test.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/lite/include/lite_session.h"
#include "mindspore/lite/src/executor.h"
#include "mindspore/lite/schema/inner/anf_ir_generated.h"

namespace mindspore {
class TestLiteInference : public mindspore::CommonTest {
 public:
  TestLiteInference() {}
};

std::string RealPath(const char *path) {
  if (path == nullptr) {
    return "";
  }
  if ((strlen(path)) >= PATH_MAX) {
    return "";
  }

  std::shared_ptr<char> resolvedPath(new (std::nothrow) char[PATH_MAX]{0});
  if (resolvedPath == nullptr) {
    return "";
  }

  auto ret = realpath(path, resolvedPath.get());
  if (ret == nullptr) {
    return "";
  }
  return resolvedPath.get();
}

char *ReadModelFile(const char *file, size_t *size) {
  if (file == nullptr) {
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::ifstream ifs(RealPath(file));
  if (!ifs.good()) {
    return nullptr;
  }

  if (!ifs.is_open()) {
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

// TEST_F(TestLiteInference, Net) {
//  auto msGraph = std::make_shared<lite::GraphDefT>();
//  msGraph->name = "graph";
//  auto msSubgraph = std::make_unique<lite::SubGraphDefT>();
//  msSubgraph->name = "subGraph";
//
//  auto node = std::make_unique<lite::OpDefT>();
//  node->inputIndex = {0, 1};
//  node->outputIndex = {2};
//  node->attr.type = lite::OpT_Add;
//  node->attr.value = new lite::AddT;
//  node->name = "Add";
//  node->fmkType = lite::FmkType_CAFFE;
//  msSubgraph->nodes.emplace_back(std::move(node));
//
//  msSubgraph->inputIndex = {0};
//  msSubgraph->outputIndex = {2};
//
//  auto input0 = std::make_unique<lite::TensorDefT>();
//  input0->refCount = lite::MSCONST_WEIGHT_REFCOUNT;
//  input0->format = lite::Format_NCHW;
//  input0->dataType = TypeId::kNumberTypeFloat;
//  input0->dims = {1, 1, 2, 2};
//  input0->offset = -1;
//  msSubgraph->allTensors.emplace_back(std::move(input0));
//
//  auto input1 = std::make_unique<lite::TensorDefT>();
//  input1->refCount = lite::MSCONST_WEIGHT_REFCOUNT;
//  input1->format = lite::Format_NCHW;
//  input1->dataType = TypeId::kNumberTypeFloat;
//  input1->dims = {1, 1, 2, 2};
//  input1->offset = -1;
//  input1->data.resize(16);
//  msSubgraph->allTensors.emplace_back(std::move(input1));
//
//  auto output = std::make_unique<lite::TensorDefT>();
//  output->refCount = 0;
//  output->format = lite::Format_NCHW;
//  output->dims = {1, 1, 2, 2};
//  output->offset = -1;
//  msSubgraph->allTensors.emplace_back(std::move(output));
//  msGraph->subgraphs.emplace_back(std::move(msSubgraph));
//
//  flatbuffers::FlatBufferBuilder builder(1024);
//  auto offset = lite::GraphDef::Pack(builder, msGraph.get());
//  builder.Finish(offset);
//  int size = builder.GetSize();
//  auto *content = builder.GetBufferPointer();
//  mindspore::lite::Context context;
//  context.allocator = nullptr;
//  context.deviceCtx.type = mindspore::lite::DeviceType::DT_CPU;
// #if 0
//    auto graph = mindspore::lite::inference::LoadModel((char *)content, size);
//
//    auto session = mindspore::lite::inference::Session::CreateSession(&context);
//
//    std::vector<float> z1 = {1.1, 2.1, 3.1, 4.1};
//    std::vector<inference::MSTensor *> inputs;
//    auto t1 = inference::MSTensor::CreateTensor(TypeId::kNumberTypeFloat32, std::vector<int>({1, 1, 2, 2}));
//    memcpy_s(t1->MutableData(), z1.size() * sizeof(float), z1.data(), z1.size() * sizeof(float));
//
//    auto t2 = inference::MSTensor::CreateTensor(TypeId::kNumberTypeFloat32, std::vector<int>({1, 1, 2, 2}));
//    memcpy_s(t2->MutableData(), z1.size() * sizeof(float), z1.data(), z1.size() * sizeof(float));
//
//    inputs.push_back(t1);
//    inputs.push_back(t1);
//    //    VectorRef *outputs = new VectorRef();
//    auto outputs = session->RunGraph(inputs);
// #else
//  auto file = "./efficientnet_b0.ms";
//  size_t model_size;
//
//  char *modelbuf = ReadModelFile(file, &model_size);
//  auto graph = mindspore::lite::inference::LoadModel(modelbuf, model_size);
//  auto session = mindspore::lite::inference::Session::CreateSession(&context);
//  session->CompileGraph(graph);
//  std::vector<inference::MSTensor *> inputs;
//  auto t1 = inference::MSTensor::CreateTensor(TypeId::kNumberTypeFloat32, std::vector<int>({1, 244, 244, 3}));
//
//  inputs.push_back(t1);
//  auto outputs = session->RunGraph(inputs);
// #endif
// }

// TEST_F(TestLiteInference, Conv) {
//   auto msGraph = std::make_shared<lite::GraphDefT>();
//   msGraph->name = "graph";
//   auto msSubgraph = std::make_unique<lite::SubGraphDefT>();
//   msSubgraph->name = "subGraph";
//
//   auto node = std::make_unique<lite::OpDefT>();
//   node->inputIndex = {0, 1};
//   node->outputIndex = {2};
//   node->attr.type = lite::OpT_Conv2D;
//   auto attr = new lite::Conv2DT;
//   attr->padMode = lite::PadMode_SAME;
//   attr->channelIn = 1;
//   attr->channelOut = 1;
//   attr->format = lite::Format_NHWC;
//   attr->strideH = 1;
//   attr->strideW = 1;
//   attr->kernelH = 2;
//   attr->kernelW = 2;
//
//   node->attr.value = attr;
//   node->name = "Conv2D";
//   node->fmkType = lite::FmkType_CAFFE;
//   msSubgraph->nodes.emplace_back(std::move(node));
//
//   msSubgraph->inputIndex = {0};
//   msSubgraph->outputIndex = {2};
//   // MS_LOG(ERROR) << "OutData";
//
//   auto input0 = std::make_unique<lite::TensorDefT>();
//   input0->refCount = lite::MSCONST_WEIGHT_REFCOUNT;
//   input0->format = lite::Format_NCHW;
//   input0->dataType = TypeId::kNumberTypeFloat;
//   input0->dims = {1, 1, 5, 5};
//   // input0->data.resize(sizeof(float) * 25);
//   // std::vector<float> input_data = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
//   // memcpy(input0->data.data(), input_data.data(), sizeof(int) * 25);
//   input0->offset = -1;
//   msSubgraph->allTensors.emplace_back(std::move(input0));
//
//   auto weight = std::make_unique<lite::TensorDefT>();
//   weight->refCount = lite::MSCONST_WEIGHT_REFCOUNT;
//   weight->format = lite::Format_KHWC;
//   weight->dataType = TypeId::kNumberTypeFloat;
//   weight->dims = {1, 2, 2, 1};
//   weight->data.resize(sizeof(float) * 4);
//   std::vector<float> weight_data = {1, 2, 3, 4};
//   memcpy(weight->data.data(), weight_data.data(), sizeof(int) * 4);
//   weight->offset = -1;
//   msSubgraph->allTensors.emplace_back(std::move(weight));
//
//   auto output = std::make_unique<lite::TensorDefT>();
//   output->refCount = 0;
//   output->format = lite::Format_NCHW;
//   output->dims = {1, 1, 5, 5};
//   output->offset = -1;
//   msSubgraph->allTensors.emplace_back(std::move(output));
//   msGraph->subgraphs.emplace_back(std::move(msSubgraph));
//
//   flatbuffers::FlatBufferBuilder builder(1024);
//   auto offset = lite::GraphDef::Pack(builder, msGraph.get());
//   builder.Finish(offset);
//   int size = builder.GetSize();
//   auto *content = builder.GetBufferPointer();
//   mindspore::lite::Context context;
//   context.allocator = nullptr;
//   context.deviceCtx.type = mindspore::lite::DeviceType::DT_CPU;
//   auto graph = mindspore::lite::inference::LoadModel((char *)content, size);
//   auto session = mindspore::lite::inference::Session::CreateSession(&context);
//   session->CompileGraph(graph);
//   std::vector<inference::MSTensor *> inputs;
//   auto t1 = inference::MSTensor::CreateTensor(TypeId::kNumberTypeFloat32, std::vector<int>({1, 3, 244, 244}));
//
//   inputs.push_back(t1);
//   auto outputs = session->RunGraph(inputs);
// }

}  // namespace mindspore
