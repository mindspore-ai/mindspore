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

#ifndef MINDSPORE_LITE_SRC_LITE_KERNEL_UTIL_H_
#define MINDSPORE_LITE_SRC_LITE_KERNEL_UTIL_H_
#include <vector>
#include <set>
#include "src/lite_kernel.h"

namespace mindspore::kernel {

class LiteKernelUtil {
 public:
  static std::vector<kernel::LiteKernel *> SubgraphInputNodes(const std::vector<kernel::LiteKernel *> &kernels);
  static std::vector<kernel::LiteKernel *> SubgraphOutputNodes(const std::vector<kernel::LiteKernel *> &kernels);
  static std::vector<lite::Tensor *> SubgraphInputTensors(const std::vector<kernel::LiteKernel *> &kernels);
  static std::vector<lite::Tensor *> SubgraphOutputTensors(const std::vector<kernel::LiteKernel *> &kernels);
  static int TopologicalSortKernels(std::vector<kernel::LiteKernel *> *kernels);
  static void InitTensorInitRefCount(const std::vector<kernel::LiteKernel *> &kernels);
  static int SetInput(const LiteKernel &kernelMod, const std::vector<lite::Tensor *> &inputs);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  static bool IsSwitchCall(kernel::LiteKernel *kernel);
#endif
  static kernel::LiteKernel *GetInputsSpecificNode(const kernel::LiteKernel *kernel,
                                                   const schema::PrimitiveType &primitive_type);
  static bool InputsContainsSpecificNode(const kernel::LiteKernel *kernel, const schema::PrimitiveType &primitive_type);
  // find in_kernels_ and out_kernels of kernel, sub_graph and nodes_ in sub_graph
  static void FindAllInoutKernels(const std::vector<kernel::LiteKernel *> &kernels);

 private:
  static std::set<lite::Tensor *> AllOutTensor(const std::vector<kernel::LiteKernel *> &kernels);
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITE_KERNEL_UTIL_H_
