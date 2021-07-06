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
#include "tools/benchmark/benchmark.h"
#include "src/mindrt_executor.h"
#include "src/lite_session.h"
#include "src/lite_kernel.h"

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
  //  std::string bin_file = ;
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

  /* output data control by users */
  for (auto out_data : out_datas) {
    string name = out_data.first;
    void *data = out_data.second;
    float *fp32_data = reinterpret_cast<float *>(data);
    if (name == "Conv2D-7") {
      ASSERT_LE(fabs(fp32_data[0] - (-0.01506812)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (0.007832255)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (-0.00440396)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (0.000382302)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (0.001282413)), 0.01);
    }
    if (name == "Conv2D-8") {
      ASSERT_LE(fabs(fp32_data[0] - (0.019412944)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (-0.01643771)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (0.001904978)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (-0.00486740)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (0.009935631)), 0.01);
    }
    if (name == "Conv2D-9") {
      ASSERT_LE(fabs(fp32_data[0] - (-0.012825339)), 0.01);
      ASSERT_LE(fabs(fp32_data[1] - (-0.012769699)), 0.01);
      ASSERT_LE(fabs(fp32_data[2] - (-0.004285028)), 0.01);
      ASSERT_LE(fabs(fp32_data[3] - (-0.002383671)), 0.01);
      ASSERT_LE(fabs(fp32_data[4] - (-0.005860286)), 0.01);
    }
    free(data);
  }
}
}  // namespace mindspore
