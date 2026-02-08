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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_ADAPTER_ANF_NODE_HOLDER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_ADAPTER_ANF_NODE_HOLDER_H_

#include <string>
#include "backend/ms_backend/graph_fusion/expander/base/node.h"
#include "ir/anf.h"

namespace mindspore::graphkernel::expander {
class AnfNodeHolder : public Node {
 public:
  explicit AnfNodeHolder(const AnfNodePtr &node) : node_(node) {}
  ~AnfNodeHolder() override = default;
  ValuePtr GetValue() override;
  const void *obj() override { return static_cast<void *>(&node_); }

 protected:
  AnfNodePtr node_;
};

class AnfNodeHolderWithHostInfo : public AnfNodeHolder {
 public:
  using AnfNodeHolder::AnfNodeHolder;
  BaseShapePtr GetShapePtr() override;
  ShapeVector GetShape() override;
  TypePtr GetDtype() override;
  std::string GetFormat() override;
};

class AnfNodeHolderWithDeviceInfo : public AnfNodeHolder {
 public:
  using AnfNodeHolder::AnfNodeHolder;
  BaseShapePtr GetShapePtr() override;
  ShapeVector GetShape() override;
  TypePtr GetDtype() override;
  std::string GetFormat() override;
};
}  // namespace mindspore::graphkernel::expander
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_MINDIR_ADAPTER_ANF_NODE_HOLDER_H_
