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

#ifndef MINDSPORE_LITE_SRC_BACKEND_OPENCL_SUBGRAPH_OPENCL_KENEL_H_
#define MINDSPORE_LITE_SRC_BACKEND_OPENCL_SUBGRAPH_OPENCL_KENEL_H_

#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "src/runtime/opencl/opencl_allocator.h"

namespace mindspore::kernel {

struct SubGraphOpenCLParameter {
  OpParameter op_parameter;
  int input_size;
  int output_size;
};

class SubGraphOpenCLKernel : public SubGraphKernel {
 public:
  explicit SubGraphOpenCLKernel(const std::vector<lite::tensor::Tensor *> inputs,
                                const std::vector<lite::tensor::Tensor *> outputs,
                                const std::vector<kernel::LiteKernel *> inKernels,
                                const std::vector<kernel::LiteKernel *> outKernels,
                                const std::vector<kernel::LiteKernel *> nodes)
      : SubGraphKernel(inputs, outputs, inKernels, outKernels, nodes, nullptr, nullptr) {}
  ~SubGraphOpenCLKernel() override;

  int Init() override;
  int InferShape() override;
  int ReSize() override;
  int Run() override;
  int UnInit();

 private:
  SubGraphOpenCLParameter *subgraph_ocl_parameter_;
  lite::opencl::OpenCLAllocator *allocator_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_OPENCL_SUBGRAPH_OPENCL_KERNEL_H_

