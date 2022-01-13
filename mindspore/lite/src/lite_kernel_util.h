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
#include "src/sub_graph_kernel.h"
#include "src/inner_context.h"

namespace mindspore::kernel {

class LiteKernelUtil {
 public:
  static std::vector<LiteKernel *> SubgraphInputNodes(const std::vector<LiteKernel *> &kernels);
  static std::vector<LiteKernel *> SubgraphOutputNodes(const std::vector<LiteKernel *> &kernels);
  static std::vector<lite::Tensor *> SubgraphInputTensors(const std::vector<LiteKernel *> &kernels);
  static std::vector<lite::Tensor *> SubgraphOutputTensors(const std::vector<LiteKernel *> &kernels);
  static int TopologicalSortKernels(std::vector<LiteKernel *> *kernels);
  static void InitTensorInitRefCount(const std::vector<LiteKernel *> &kernels);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  static bool IsSwitchTypeCall(LiteKernel *kernel);
  static bool IsNonTailCall(LiteKernel *node);
  static bool IsTailCall(LiteKernel *node);
  static std::vector<LiteKernel *> GetCallInputPartials(LiteKernel *call_node);
  static LiteKernel *GetPartialOutputCall(LiteKernel *partial_node);
  static bool IsNonTailCallSubGraph(LiteKernel *kernel);
  static bool IsTailCallSubGraph(LiteKernel *kernel);
  static std::vector<LiteKernel *> GetCallInputPartialsCorrespondingOutputSubgraph(LiteKernel *call_node);
#endif
  static LiteKernel *GetInputsSpecificNode(const LiteKernel *kernel, const schema::PrimitiveType &primitive_type);
  static bool InputsContainsSpecificNode(const LiteKernel *kernel, const schema::PrimitiveType &primitive_type);
  // find in_kernels_ and out_kernels of kernel, sub_graph and nodes_ in sub_graph
  static void FindAllInoutKernels(const std::vector<LiteKernel *> &kernels);
  static void FindAllInoutKernelsInSubgraphKernel(const std::vector<LiteKernel *> &kernels);
  static SubGraphKernel *CreateSubGraphKernel(const std::vector<LiteKernel *> &kernels,
                                              const std::vector<lite::Tensor *> *in_tensors,
                                              const std::vector<lite::Tensor *> *out_tensors, SubGraphType type,
                                              const lite::InnerContext &context, int schema_version);
  static int ReplaceSubGraphNodesInTensor(LiteKernel *kernel, const lite::Tensor *old_tensor, lite::Tensor *new_tensor);
  static int ReplaceSubGraphNodesOutTensor(LiteKernel *kernel, const lite::Tensor *old_tensor,
                                           lite::Tensor *new_tensor);
  static bool IsOutputSubGraph(LiteKernel *subgraph_kernel);
  static SubGraphKernel *BelongToWhichSubGraph(const std::vector<LiteKernel *> &subgraphs, LiteKernel *kernel);

 private:
  static std::set<lite::Tensor *> AllOutTensor(const std::vector<LiteKernel *> &kernels);
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITE_KERNEL_UTIL_H_
