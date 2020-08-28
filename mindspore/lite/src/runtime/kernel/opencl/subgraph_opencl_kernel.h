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

 protected:
  int MallocTensorWithReuse();
  int GenToFormatOp(const std::vector<lite::tensor::Tensor *> &in_tensors,
                    const std::vector<std::vector<kernel::LiteKernel *>> in_kernels,
                    std::vector<lite::tensor::Tensor *> *out_tensors,
                    std::vector<OpenCLToFormatParameter *> *out_parameters, std::vector<LiteKernel *> *out_convert_ops,
                    OpenCLMemType mem_type);
  int GetKernelFromToTensor(const std::vector<lite::tensor::Tensor *> &in_tensors,
                            const std::vector<kernel::LiteKernel *> &in_kernels,
                            std::vector<std::vector<kernel::LiteKernel *>> *out_kernels, bool is_from);

 private:
  lite::opencl::OpenCLAllocator *allocator_;
  std::vector<lite::tensor::Tensor *> in_convert_tensors_;
  std::vector<lite::tensor::Tensor *> out_convert_tensors_;
  std::vector<OpenCLToFormatParameter *> in_parameters_;
  std::vector<OpenCLToFormatParameter *> out_parameters_;
  std::vector<LiteKernel *> in_convert_ops_;
  std::vector<LiteKernel *> out_convert_ops_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_OPENCL_SUBGRAPH_OPENCL_KERNEL_H_
