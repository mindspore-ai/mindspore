/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "gtest/gtest.h"
#include "common/common_test.h"
#include "include/errorcode.h"
#include "tools/converter/converter.h"
#include "src/lite_session.h"
#include "src/lite_kernel.h"
#include "include/api/types.h"
#include "include/api/graph.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/cell.h"

namespace mindspore {
class GraphTest : public mindspore::CommonTest {
 public:
  GraphTest() {}
};

TEST_F(GraphTest, UserSetGraphOutput1) {
  size_t size = 0;
  char *graph_buf = lite::ReadFile("./mindrtParallel/mindrt_parallel_model_split.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  ASSERT_NE(model, nullptr);

  auto context = std::make_shared<lite::Context>();
  ASSERT_NE(context, nullptr);

  session::LiteSession *session = session::LiteSession::CreateSession(context.get());
  ASSERT_NE(session, nullptr);

  int benchmark_ret = session->CompileGraph(model.get());
  ASSERT_EQ(benchmark_ret, lite::RET_OK);

  /* set input data */
  auto inputs = session->GetInputs();
  auto in = inputs[0];
  auto in_data = in->MutableData();
  char *bin_buf = lite::ReadFile("./mindrtParallel/mindrt_parallel_model.bin", &size);
  memcpy(in_data, bin_buf, in->Size());

  /* set output data */
  std::map<string, void *> out_datas;
  auto outputs = session->GetOutputs();
  for (auto &out_tensor_pair : outputs) {
    string out_name = out_tensor_pair.first;
    tensor::MSTensor *out_tensor = out_tensor_pair.second;

    void *out_data = malloc(out_tensor->Size());
    out_datas.insert(std::make_pair(out_name, out_data));

    out_tensor->set_data(out_data);
    out_tensor->set_allocator(nullptr);
  }

  /* run graph */
  benchmark_ret = session->RunGraph(nullptr, nullptr);
  ASSERT_EQ(benchmark_ret, lite::RET_OK);
  delete session;

  static int output_count = 0;
  /* output data control by users */
  for (auto out_data : out_datas) {
    string name = out_data.first;
    void *data = out_data.second;
    float *fp32_data = reinterpret_cast<float *>(data);
    if (name == "output") {
      output_count++;
      ASSERT_LE(fabs(fp32_data[0] - (0.115831)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (0.113074)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (0.245932)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (0.346307)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (-0.15687)), 0.01);
    }
    if (name == "output2") {
      output_count++;
      ASSERT_LE(fabs(fp32_data[0] - (0.06387864)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (0.22883008)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (0.23308714)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (0.04586578)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (0.06820235)), 0.01);
    }
    if (name == "output3") {
      output_count++;
      ASSERT_LE(fabs(fp32_data[0] - (-0.1617176)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (-0.3828573)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (0.08357508)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (0.10217983)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (-0.3480084)), 0.01);
    }
    free(data);
  }
  ASSERT_EQ(output_count, 3);
}

TEST_F(GraphTest, UserSetGraphOutput2) {
  size_t size = 0;
  char *model_buf = lite::ReadFile("./mindrtParallel/mindrt_parallel_model_split.ms", &size);
  ASSERT_NE(model_buf, nullptr);

  Graph graph;
  Status load_ret = Serialization::Load(model_buf, size, kMindIR, &graph);
  ASSERT_EQ(load_ret == kSuccess, true);

  auto context = std::make_shared<Context>();
  ASSERT_NE(context, nullptr);

  auto &device_list = context->MutableDeviceInfo();

  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_list.push_back(device_info);

  GraphCell graph_cell(graph);
  Model *model = new Model();
  ASSERT_NE(model, nullptr);
  Status build_ret = model->Build(graph_cell, context);
  ASSERT_EQ(build_ret == kSuccess, true);

  /* set input data */
  std::vector<MSTensor> inputs = model->GetInputs();
  auto in = inputs[0];
  auto in_data = in.MutableData();
  char *bin_buf = lite::ReadFile("./mindrtParallel/mindrt_parallel_model.bin", &size);
  memcpy(in_data, bin_buf, in.DataSize());

  /* set output data */
  std::vector<void *> out_datas;
  auto outputs = model->GetOutputs();
  for (MSTensor &out_tensor : outputs) {
    void *out_data = malloc(out_tensor.DataSize());
    out_datas.push_back(out_data);

    out_tensor.SetData(out_data);
    out_tensor.SetAllocator(nullptr);
  }

  /* run graph */
  Status predict_ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(predict_ret == kSuccess, true);
  delete model;

  /* output data control by users */
  for (int i = 0; i < 3; i++) {
    void *out_data = out_datas[i];
    float *fp32_data = reinterpret_cast<float *>(out_data);
    if (i == 0) {
      ASSERT_LE(fabs(fp32_data[0] - (0.115831)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (0.113074)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (0.245932)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (0.346307)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (-0.15687)), 0.01);
    }
    if (i == 1) {
      ASSERT_LE(fabs(fp32_data[0] - (0.06387864)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (0.22883008)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (0.23308714)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (0.04586578)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (0.06820235)), 0.01);
    }
    if (i == 2) {
      ASSERT_LE(fabs(fp32_data[0] - (-0.1617176)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (-0.3828573)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (0.08357508)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (0.10217983)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (-0.3480084)), 0.01);
    }
    free(out_data);
  }
}

}  // namespace mindspore
