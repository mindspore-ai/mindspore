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

#define USE_DEPRECATED_API
#include "gtest/gtest.h"
#include "common/common_test.h"
#include "include/errorcode.h"
#include "tools/converter/converter.h"
#include "tools/benchmark/run_benchmark.h"
#include "src/litert/mindrt_executor.h"
#include "src/litert/lite_session.h"
#include "src/litert/kernel_exec.h"
#include "src/common/file_utils.h"
#include "include/converter.h"

namespace mindspore {
class MindrtParallelTest : public mindspore::CommonTest {
 public:
  MindrtParallelTest() {}
};

int CheckOffline1(lite::LiteSession *session) {
  /* -----------  start check -------------- */
  lite::LiteSession *lite_session = reinterpret_cast<lite::LiteSession *>(session);
  auto kernels = lite_session->get_kernels();
  if (kernels.size() != 4) {
    return -1;
  }

  /* sub-graph-0 */
  kernel::SubGraphKernel *subgraph0 = reinterpret_cast<kernel::SubGraphKernel *>(kernels[0]);
  std::vector<kernel::KernelExec *> nodes0 = subgraph0->nodes();
  if (nodes0.size() != 1) {
    return -2;
  }
  if (nodes0[0]->type() != schema::PrimitiveType_SplitWithOverlap) {
    return -3;
  }

  /* sub-graph-1 */
  kernel::SubGraphKernel *subgraph1 = reinterpret_cast<kernel::SubGraphKernel *>(kernels[1]);
  std::vector<kernel::KernelExec *> nodes1 = subgraph1->nodes();
  if (nodes1.size() != 3) {
    return -4;
  }
  if (nodes1[0]->type() != schema::PrimitiveType_Conv2DFusion ||
      nodes1[1]->type() != schema::PrimitiveType_Conv2DFusion ||
      nodes1[2]->type() != schema::PrimitiveType_Conv2DFusion) {
    return -5;
  }

  /* sub-graph-2 */
  kernel::SubGraphKernel *subgraph2 = reinterpret_cast<kernel::SubGraphKernel *>(kernels[2]);
  std::vector<kernel::KernelExec *> nodes2 = subgraph2->nodes();
  if (nodes2.size() != 3) {
    return -6;
  }
  if (nodes2[0]->type() != schema::PrimitiveType_Conv2DFusion ||
      nodes2[1]->type() != schema::PrimitiveType_Conv2DFusion ||
      nodes2[2]->type() != schema::PrimitiveType_Conv2DFusion) {
    return -7;
  }

  /* sub-graph-3 */
  kernel::SubGraphKernel *subgraph3 = reinterpret_cast<kernel::SubGraphKernel *>(kernels[3]);
  std::vector<kernel::KernelExec *> nodes3 = subgraph3->nodes();
  if (nodes3.size() != 12) {
    return -8;
  }
  if (nodes3[0]->type() != schema::PrimitiveType_Concat) {
    return -9;
  }

  return lite::RET_OK;
}

int CheckRuntime1(lite::LiteSession *session) {
  lite::LiteSession *lite_session = reinterpret_cast<lite::LiteSession *>(session);
  auto kernels = lite_session->get_kernels();
  if (kernels.size() != 6) {
    return -1;
  }
  return lite::RET_OK;
}

TEST_F(MindrtParallelTest, offline1) {
  mindspore::Converter converter(converter::kFmkTypeTflite, "./mindrtParallel/mindrt_parallel_model.tflite",
                                 "./mindrtParallel/mindrt_parallel_model_split");
  converter.SetConfigFile("./mindrtParallel/mindrt_parallel_model.config");

  auto status = converter.Convert();
  ASSERT_EQ(status, kSuccess);

  size_t size = 0;
  char *graph_buf = lite::ReadFile("./mindrtParallel/mindrt_parallel_model_split.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  ASSERT_NE(model, nullptr);

  auto context = std::make_shared<lite::InnerContext>();
  ASSERT_NE(context, nullptr);
  context->enable_parallel_ = true;

  lite::LiteSession *session = lite::LiteSession::CreateSession(context);
  ASSERT_NE(session, nullptr);

  int benchmark_ret = session->CompileGraph(model.get());
  ASSERT_EQ(benchmark_ret, lite::RET_OK);

  ASSERT_EQ(CheckOffline1(session), lite::RET_OK);

  auto inputs = session->GetInputs();
  for (auto in : inputs) {
    in->MutableData();
  }

  benchmark_ret = session->RunGraph(nullptr, nullptr);
  ASSERT_EQ(benchmark_ret, lite::RET_OK);

  delete session;
}

TEST_F(MindrtParallelTest, runtime1) {
  mindspore::Converter converter(converter::kFmkTypeTflite, "./mindrtParallel/mindrt_parallel_model.tflite",
                                 "./mindrtParallel/mindrt_parallel_model");
  converter.SetConfigFile("./mindrtParallel/mindrt_parallel_model.config");

  auto status = converter.Convert();
  ASSERT_EQ(status, kSuccess);

  size_t size = 0;
  char *graph_buf = lite::ReadFile("./mindrtParallel/mindrt_parallel_model.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  ASSERT_NE(model, nullptr);

  auto context = std::make_shared<lite::InnerContext>();
  ASSERT_NE(context, nullptr);
  context->enable_parallel_ = true;

  lite::LiteSession *session = lite::LiteSession::CreateSession(context);
  ASSERT_NE(session, nullptr);

  int benchmark_ret = session->CompileGraph(model.get());
  ASSERT_EQ(benchmark_ret, lite::RET_OK);

  ASSERT_EQ(CheckRuntime1(session), lite::RET_OK);

  auto inputs = session->GetInputs();
  for (auto in : inputs) {
    in->MutableData();
  }
  benchmark_ret = session->RunGraph(nullptr, nullptr);
  ASSERT_EQ(benchmark_ret, lite::RET_OK);

  delete session;
}
}  // namespace mindspore
