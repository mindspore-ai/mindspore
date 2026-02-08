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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_NODE_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_NODE_H_

#include <memory>
#include <vector>
#include <string>
#include "abstract/dshape.h"

namespace mindspore::graphkernel::expander {
using abstract::BaseShapePtr;
class Node {
 public:
  Node() = default;
  virtual ~Node() = default;

  virtual BaseShapePtr GetShapePtr() = 0;
  virtual ShapeVector GetShape() = 0;
  virtual TypePtr GetDtype() = 0;
  virtual std::string GetFormat() = 0;
  // virtual std::vector<std::string> GetFormats() = 0;
  virtual ValuePtr GetValue() = 0;

  // Get the real object of Node.
  virtual const void *obj() = 0;
  template <typename T>
  const T &as() {
    return *(static_cast<const T *>(obj()));
  }
};
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
using NodePtrDict = HashMap<std::string, NodePtr>;
}  // namespace mindspore::graphkernel::expander
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_NODE_H_
