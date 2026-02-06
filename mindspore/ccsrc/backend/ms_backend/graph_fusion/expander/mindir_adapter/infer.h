/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_ADAPTER_INFER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_ADAPTER_INFER_H_

#include "backend/ms_backend/graph_fusion/expander/mindir_adapter/anf_node_holder.h"
#include "backend/ms_backend/graph_fusion/model/node.h"
#include "utils/hash_map.h"

namespace mindspore::graphkernel::expander {
class MindirInfer {
 public:
  MindirInfer() = default;
  virtual ~MindirInfer() = default;
  virtual void InferOp(const NodePtr &node, const PrimitivePtr &prim, const NodePtrList &args) = 0;
  virtual void SetValue(const NodePtr &node) = 0;
  virtual void HandleInputs(const NodePtrList &) {}
};

class InferByHostInfo : public MindirInfer {
 public:
  void InferOp(const NodePtr &node, const PrimitivePtr &prim, const NodePtrList &args) override;
  void SetValue(const NodePtr &node) override { node->as<AnfNodePtr>()->set_abstract(node->GetValue()->ToAbstract()); }
};

class InferByDeviceInfo : public MindirInfer {
 public:
  void InferOp(const NodePtr &node, const PrimitivePtr &prim, const NodePtrList &args) override;
  void SetValue(const NodePtr &node) override;
  void HandleInputs(const NodePtrList &inputs) override;

 protected:
  HashMap<NodePtr, inner::NodePtr> inner_node_cache_;
};
}  // namespace mindspore::graphkernel::expander
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_ADAPTER_INFER_H_
