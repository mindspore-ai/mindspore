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
#include "src/kernel_exec.h"
#include "src/sub_graph_kernel.h"
#include "src/inner_context.h"

namespace mindspore::kernel {

class KernelExecUtil {
 public:
  static std::vector<KernelExec *> SubgraphInputNodes(const std::vector<KernelExec *> &kernels);
  static std::vector<KernelExec *> SubgraphOutputNodes(const std::vector<KernelExec *> &kernels);
  static std::vector<lite::Tensor *> SubgraphInputTensors(const std::vector<KernelExec *> &kernels);
  static std::vector<lite::Tensor *> SubgraphOutputTensors(const std::vector<KernelExec *> &kernels);
  static int TopologicalSortKernels(std::vector<KernelExec *> *kernels);
  static void InitTensorInitRefCount(const std::vector<KernelExec *> &kernels);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  static bool IsSwitchTypeCall(KernelExec *kernel);
  static bool IsNonTailCall(KernelExec *node);
  static bool IsTailCall(KernelExec *node);
  static std::vector<KernelExec *> GetCallInputPartials(KernelExec *call_node);
  static KernelExec *GetPartialOutputCall(KernelExec *partial_node);
  static bool IsNonTailCallSubGraph(KernelExec *kernel);
  static bool IsTailCallSubGraph(KernelExec *kernel);
  static std::vector<KernelExec *> GetCallInputPartialsCorrespondingOutputSubgraph(KernelExec *call_node);
#endif
  static KernelExec *GetInputsSpecificNode(const KernelExec *kernel, const schema::PrimitiveType &primitive_type);
  static bool InputsContainsSpecificNode(const KernelExec *kernel, const schema::PrimitiveType &primitive_type);
  // find in_kernels_ and out_kernels of kernel, sub_graph and nodes_ in sub_graph
  static void FindAllInoutKernels(const std::vector<KernelExec *> &kernels);
  static void FindAllInoutKernelsInSubgraphKernel(const std::vector<KernelExec *> &kernels);
  static SubGraphKernel *CreateSubGraphKernel(const std::vector<KernelExec *> &kernels,
                                              const std::vector<lite::Tensor *> *in_tensors,
                                              const std::vector<lite::Tensor *> *out_tensors, SubGraphType type,
                                              const lite::InnerContext &context, int schema_version);
  static int ReplaceSubGraphNodesInTensor(KernelExec *kernel, const lite::Tensor *old_tensor, lite::Tensor *new_tensor);
  static int ReplaceSubGraphNodesOutTensor(KernelExec *kernel, const lite::Tensor *old_tensor,
                                           lite::Tensor *new_tensor);
  static bool IsOutputSubGraph(KernelExec *subgraph_kernel);
  static SubGraphKernel *BelongToWhichSubGraph(const std::vector<KernelExec *> &subgraphs, KernelExec *kernel);

 private:
  static std::set<lite::Tensor *> AllOutTensor(const std::vector<KernelExec *> &kernels);
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITE_KERNEL_UTIL_H_
