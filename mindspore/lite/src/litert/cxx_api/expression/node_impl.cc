/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/cxx_api/expression/node_impl.h"
#include <vector>
#include "include/api/net.h"
#include "src/expression/ops.h"

namespace mindspore {
Node *NodeImpl::Connect(lite::Node *lnode) {
  auto node = std::make_unique<Node>();
  if (node == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate node";
    return nullptr;
  }
  if (lnode == nullptr) {
    MS_LOG(ERROR) << "lite node is null";
    return nullptr;
  }
  auto pnode = node.release();
  auto impl = GetImpl(pnode);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "missing implementation";
    return nullptr;
  }
  impl->set_node(lnode);
  lnode->set_impl(impl);
  return pnode;
}
namespace NN {
std::unique_ptr<Node> Input(std::vector<int> dims, DataType data_type, int fmt) {
  auto type = static_cast<TypeId>(data_type);
  auto lite_node = lite::NN::Input(dims, type, fmt);
  return std::unique_ptr<Node>(NodeImpl::Connect(lite_node));
}
}  // namespace NN
}  // namespace mindspore
