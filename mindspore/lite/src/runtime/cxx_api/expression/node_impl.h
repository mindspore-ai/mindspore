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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_EXPRESSION_NODE_IMPL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_EXPRESSION_NODE_IMPL_H_

#include <algorithm>
#include <set>
#include <memory>
#include <vector>
#include "include/api/net.h"
#include "include/api/cfg.h"
#include "include/api/data_type.h"
#include "src/expression/node.h"
#include "src/expression/expr.h"

namespace mindspore {
using lite::EXPR;
class NodeSet {
 public:
  std::set<lite::Node *> set_;
};

class Expr : public EXPR {
 public:
  static std::vector<EXPR *> convert(const std::vector<Expr *> &input) {
    std::vector<EXPR *> vec(input.size());
    (void)std::transform(input.begin(), input.end(), vec.begin(), [](Expr *e) { return reinterpret_cast<EXPR *>(e); });
    return vec;
  }
  static std::vector<Expr *> convert(const std::vector<EXPR *> &input) {
    std::vector<Expr *> vec(input.size());
    (void)std::transform(input.begin(), input.end(), vec.begin(), [](EXPR *e) { return reinterpret_cast<Expr *>(e); });
    return vec;
  }
};

class MS_API NodeImpl {
 public:
  std::vector<Expr *> operator()(const std::vector<Expr *> &inputs) {
    auto in = Expr::convert(inputs);
    auto out = (*node_)(in);
    return Expr::convert(out);
  }
  lite::Node *node() { return node_; }
  void set_node(lite::Node *node) { node_ = node; }
  void set_pnode(Node *node) { pnode_ = node; }
  Node *pnode() { return pnode_; }
  static Node *Connect(lite::Node *lnode);
  static std::shared_ptr<NodeImpl> &GetImpl(Node *node) { return node->impl_; }

 private:
  Node *pnode_{nullptr};
  lite::Node *node_{nullptr};
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_EXPRESSION_NODE_IMPL_H_
