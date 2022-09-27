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

#include "ut/src/runtime/kernel/opencl/common.h"
#include "include/errorcode.h"
#include "src/litert/mindrt_executor.h"
#include "src/litert/lite_session.h"
#include "src/litert/kernel_exec.h"

class MindrtRuntimeTest : public mindspore::CommonTest {
 public:
  MindrtRuntimeTest() = default;
};

int CheckRuntime(mindspore::lite::LiteSession *session) {
  auto kernels = session->get_kernels();

  int cpu_kernel_count = 0;
  int gpu_kernel_count = 0;
  for (auto kernel : kernels) {
    if (kernel->subgraph_type() == mindspore::kernel::kGpuFp32SubGraph) {
      gpu_kernel_count++;
    }
    if (kernel->subgraph_type() == mindspore::kernel::kCpuFP32SubGraph) {
      cpu_kernel_count++;
    }
  }

  if (kernels.size() != 6) {
    return -1;
  }
  if (cpu_kernel_count != 4) {
    return -2;
  }
  if (gpu_kernel_count != 2) {
    return -3;
  }

  return 0;
}

TEST_F(MindrtRuntimeTest, Runtime) {
  size_t size = 0;
  char *graph_buf = mindspore::lite::ReadFile("./test_data/mindrt_parallel/parallel.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<mindspore::lite::Model>(mindspore::lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  ASSERT_NE(model, nullptr);

  auto context = std::make_shared<mindspore::lite::InnerContext>();
  ASSERT_NE(context, nullptr);
  context->enable_parallel_ = true;
  mindspore::lite::DeviceContext gpu_device_ctx;
  gpu_device_ctx.device_type_ = mindspore::lite::DT_GPU;
  gpu_device_ctx.device_info_.gpu_device_info_.enable_float16_ = false;
  mindspore::lite::DeviceContext cpu_device_ctx;
  cpu_device_ctx.device_type_ = mindspore::lite::DT_CPU;
  gpu_device_ctx.device_info_.cpu_device_info_.enable_float16_ = false;
  context->device_list_.clear();
  context->device_list_.push_back(gpu_device_ctx);
  context->device_list_.push_back(cpu_device_ctx);

  mindspore::lite::LiteSession *session = mindspore::lite::LiteSession::CreateSession(context);
  ASSERT_NE(session, nullptr);

  int benchmark_ret = session->CompileGraph(model.get());
  ASSERT_EQ(benchmark_ret, mindspore::lite::RET_OK);

  ASSERT_EQ(CheckRuntime(session), 0);

  auto inputs = session->GetInputs();
  for (auto in : inputs) {
    in->MutableData();
  }
  benchmark_ret = session->RunGraph(nullptr, nullptr);
  ASSERT_EQ(benchmark_ret, mindspore::lite::RET_OK);

  delete session;
}

int CheckRuntime2(mindspore::lite::LiteSession *session) {
  auto kernels = session->get_kernels();

  for (auto kernel : kernels) {
    if (kernel->subgraph_type() != mindspore::kernel::kCpuFP16SubGraph) {
      return -1;
    }
  }

  if (kernels.size() != 6) {
    return -2;
  }

  return 0;
}

TEST_F(MindrtRuntimeTest, RuntimeFp16) {
  size_t size = 0;
  char *graph_buf = mindspore::lite::ReadFile("./test_data/mindrt_parallel/parallel.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<mindspore::lite::Model>(mindspore::lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  ASSERT_NE(model, nullptr);

  auto context = std::make_shared<mindspore::lite::InnerContext>();
  ASSERT_NE(context, nullptr);
  context->enable_parallel_ = true;
  auto &cpu_device_ctx = context->device_list_[0];
  cpu_device_ctx.device_info_.cpu_device_info_.enable_float16_ = true;

  mindspore::lite::LiteSession *session = mindspore::lite::LiteSession::CreateSession(context);
  ASSERT_NE(session, nullptr);

  int benchmark_ret = session->CompileGraph(model.get());
  ASSERT_EQ(benchmark_ret, mindspore::lite::RET_OK);

  ASSERT_EQ(CheckRuntime2(session), 0);

  auto inputs = session->GetInputs();
  for (auto in : inputs) {
    in->MutableData();
  }
  benchmark_ret = session->RunGraph(nullptr, nullptr);
  ASSERT_EQ(benchmark_ret, mindspore::lite::RET_OK);

  delete session;
}
