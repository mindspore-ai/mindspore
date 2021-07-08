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
#include "src/mindrt_executor.h"
#include "src/lite_session.h"
#include "src/lite_kernel.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_MindrtParallel : public CommonTest {};

int CheckRuntime(mindspore::session::LiteSession *session) {
  mindspore::lite::LiteSession *lite_session = reinterpret_cast<mindspore::lite::LiteSession *>(session);
  auto kernels = lite_session->get_kernels();

  int cpu_kernel_count = 0;
  int gpu_kernel_count = 0;
  for (auto kernel : kernels) {
    if (kernel->subgraph_type() == mindspore::kernel::kGpuSubGraph) {
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

TEST_F(TestOpenCL_MindrtParallel, Runtime) {
  size_t size = 0;
  char *graph_buf = lite::ReadFile("./test_data/mindrt_parallel/mindrt_parallel_model.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<lite::Model>(mindspore::lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  ASSERT_NE(model, nullptr);

  auto context = std::make_shared<lite::Context>();
  ASSERT_NE(context, nullptr);
  context->enable_parallel_ = true;
  DeviceContext gpu_device_ctx{DT_GPU, {false}};
  gpu_device_ctx.device_info_.gpu_device_info_.enable_float16_ = false;
  context->device_list_.push_back(gpu_device_ctx);

  session::LiteSession *session = session::LiteSession::CreateSession(context.get());
  ASSERT_NE(session, nullptr);

  int benchmark_ret = session->CompileGraph(model.get());
  ASSERT_EQ(benchmark_ret, lite::RET_OK);

  ASSERT_EQ(CheckRuntime(session), 0);

  auto inputs = session->GetInputs();
  for (auto in : inputs) {
    in->MutableData();
  }
  benchmark_ret = session->RunGraph(nullptr, nullptr);
  ASSERT_EQ(benchmark_ret, lite::RET_OK);

  delete session;
}
}  // namespace mindspore::lite::opencl::test
