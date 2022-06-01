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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_EXPR_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_EXPR_H_

#include <vector>
#include <list>
#include <memory>
#include <map>
#include <functional>
#include <string>
#include "include/api/format.h"
#include "mindapi/base/type_id.h"

namespace mindspore {
namespace lite {
class Node;

class EXPR {
 public:
  explicit EXPR(Node *node) : node_(node) { SetSize(1); }
  static void PrintDot(std::vector<EXPR *> vec);
  static void Replace(const std::vector<EXPR *> &vec, std::vector<EXPR *> *old, std::vector<EXPR *> *n);
  static void CreateOutputMap(std::vector<EXPR *> vec, std::map<EXPR *, std::list<EXPR *>> *outmap);
  static void Clear(std::vector<EXPR *> vec);
  void Travers(std::function<bool(EXPR *e, EXPR *itr)> cb);
  std::string name();
  EXPR *GetInput(int idx) { return params_.at(idx); }
  void set_node(Node *node) { node_ = node; }
  Node *node() { return node_; }
  bool visited = false;
  void set_params(std::vector<EXPR *> params) { params_ = params; }
  void set_params(int idx, EXPR *expr) { params_[idx] = expr; }
  void add_params(EXPR *e) { params_.push_back(e); }
  std::vector<EXPR *> &params() { return params_; }
  EXPR *params(int i) { return params_[i]; }
  void SetSize(int n) { params_.resize(n); }
  void SetDims(std::vector<int> dims) { dims_ = dims; }
  std::vector<int> &dims() { return dims_; }
  void set_format(int fmt) { format_ = fmt; }
  int format() { return format_; }
  void set_data_type(TypeId data_type) { data_type_ = data_type; }
  TypeId data_type() { return data_type_; }

 private:
  void Replace(EXPR **old, EXPR **n, std::vector<Node *> *to_delete);
  std::vector<EXPR *> params_;
  Node *node_{nullptr};
  void Clear();
  std::vector<int> dims_;
  int format_ = NHWC;
  TypeId data_type_ = kNumberTypeFloat32;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_EXPR_H_
