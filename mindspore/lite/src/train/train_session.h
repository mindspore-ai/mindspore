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
#ifndef MINDSPORE_LITE_SRC_TRAIN_TRAIN_SESSION_H_
#define MINDSPORE_LITE_SRC_TRAIN_TRAIN_SESSION_H_
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "include/context.h"
#include "backend/session/session_basic.h"
#include "backend/session/kernel_graph.h"
#include "mindspore/lite/src/train/lite_kernel_runtime.h"
// #include "backend/session/session_factory.h"
namespace mindspore {
namespace lite::tensor {
class Tensor;
}
namespace session {
struct KernelRelation {
  std::string node_full_name;
  std::vector<lite::tensor::Tensor *> input_tensor;
  std::vector<lite::tensor::Tensor *> output_tensor;
  CNodePtr cnode;
};

class TrainSession {
 public:
  explicit TrainSession(lite::Context * context) { Init(context); }
  ~TrainSession() = default;

  GraphId CompileGraph(NotNull<FuncGraphPtr> func_graph);

  void RunGraph(const GraphId &graph_id, const std::vector<lite::tensor::Tensor *> &inputs,
                std::vector<lite::tensor::Tensor *> *outputs);

  // void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) override
  // {};
 protected:
  void Init(lite::Context *context);
  std::shared_ptr<lite::Context> context_ = nullptr;
  std::unordered_map<GraphId, std::shared_ptr<KernelGraph>> graphs_;
  static GraphId graph_sum_;
  KernelGraphPtr NewKernelGraph();

 private:
  // GraphId CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  // GraphId CompileGraph(const char *model_buf, size_t size);
  std::shared_ptr<KernelGraph> ConstructKernelGraph(const FuncGraphPtr &func_graph);
  int BuildKernelInputAndOutputFromFuncGraph(const KernelGraphPtr &kernel_graph);

  lite::tensor::Tensor *GetTensorForAnfNode(const AnfNodePtr anf_node);

  void SetKernelInfo(const KernelGraph *kernel_graph);
  int BuildKernel(const KernelGraph *kernel_graph);
  lite::LiteInferKernelRuntime runtime_;
  std::map<std::string, KernelRelation> kernel_relation_infos_;
  FuncGraphPtr func_graph_ = NULL;
  std::map<AnfNodePtr, lite::tensor::Tensor *> tensors_;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_SESSION_H_
