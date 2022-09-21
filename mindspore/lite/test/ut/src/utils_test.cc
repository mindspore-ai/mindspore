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

#include <cmath>
#include <memory>
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "mindspore/lite/src/litert/kernel_exec.h"
#include "mindspore/lite/src/litert/kernel_exec_util.h"

namespace mindspore {
class UtilsTest : public mindspore::CommonTest {
 public:
  UtilsTest() {}
};

TEST_F(UtilsTest, TestSubgraph) {
  auto kernel0 = std::make_shared<kernel::KernelExec>();
  auto kernel1 = std::make_shared<kernel::KernelExec>();
  auto kernel2 = std::make_shared<kernel::KernelExec>();

  auto tensor0 = std::make_shared<lite::Tensor>();
  auto tensor1 = std::make_shared<lite::Tensor>();
  auto tensor2 = std::make_shared<lite::Tensor>();
  auto tensor3 = std::make_shared<lite::Tensor>();
  auto tensor4 = std::make_shared<lite::Tensor>();

  kernel0->AddOutKernel(kernel1.get());
  kernel1->AddInKernel(kernel0.get());
  kernel1->AddOutKernel(kernel2.get());
  kernel2->AddInKernel(kernel1.get());

  kernel0->set_in_tensors({tensor0.get(), tensor1.get()});
  kernel0->set_out_tensors({tensor2.get()});
  kernel1->set_in_tensors({tensor2.get()});
  kernel1->set_out_tensors({tensor3.get()});
  kernel2->set_in_tensors({tensor3.get()});
  kernel2->set_out_tensors({tensor4.get()});

  std::vector<kernel::KernelExec *> kernels = {kernel0.get(), kernel1.get(), kernel2.get()};

  auto input_kernels = kernel::KernelExecUtil::SubgraphInputNodes(kernels);
  ASSERT_EQ(input_kernels.size(), 1);
  auto output_kernels = kernel::KernelExecUtil::SubgraphOutputNodes(kernels);
  ASSERT_EQ(output_kernels.size(), 1);
  auto input_tensors = kernel::KernelExecUtil::SubgraphInputTensors(kernels);
  ASSERT_EQ(input_tensors.size(), 2);
  auto output_tensors = kernel::KernelExecUtil::SubgraphOutputTensors(kernels);
  ASSERT_EQ(output_tensors.size(), 1);
}
}  // namespace mindspore
