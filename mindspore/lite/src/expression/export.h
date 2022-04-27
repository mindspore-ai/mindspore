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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_EXPORT_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_EXPORT_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <list>
#include <map>
#include <iostream>
#include "src/expression/expr.h"

namespace mindspore {
namespace schema {
struct MetaGraphT;
}
namespace lite {
class ExportSession {
 public:
  explicit ExportSession(std::map<EXPR *, std::list<EXPR *>> &outmap) : outmap_(outmap) {}
  int Init(const std::string model_name, std::string version);
  void UpdateOutput(EXPR *expr, int id) { output_tensors_[expr] = id; }
  int GetOutput(EXPR *expr) { return output_tensors_.at(expr); }
  schema::MetaGraphT *&meta_graph() { return meta_graph_; }
  int SetInputOutput(const std::vector<EXPR *> &inputs, const std::vector<EXPR *> &outputs);
  bool IsToDependOnly(EXPR *expr);

 private:
  schema::MetaGraphT *meta_graph_{nullptr};
  std::unordered_map<EXPR *, int> output_tensors_;  // output tensors per EXPR
  std::map<EXPR *, std::list<EXPR *>> &outmap_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXPRESSION_EXPORT_H_
