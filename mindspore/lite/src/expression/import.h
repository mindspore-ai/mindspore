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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_IMPORT_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_IMPORT_H_

#include <string>
#include <unordered_map>
#include <memory>
#include "nnacl/op_base.h"
#include "src/expression/net.h"

namespace mindspore {
namespace lite {
using import_func = std::function<Node *()>;

template <typename T>
Node *ReturnNode() {
  return new (std::nothrow) T();
}

class ImportReg {
 public:
  explicit ImportReg(mindspore::schema::PrimitiveType type, import_func func) { import_map_[type] = func; }
  static import_func GetImportFunc(mindspore::schema::PrimitiveType type);

 private:
  static std::unordered_map<mindspore::schema::PrimitiveType, import_func> import_map_;
};

class Import {
 private:
  int8_t *buffer_ = nullptr;
  OpParameter *GetAttr(const schema::Primitive *prim);
  std::unique_ptr<Node> CreateNode(const schema::CNode *cnode);

 public:
  Net *ImportMs(const schema::MetaGraph *meta_graph);
  Net *ImportMs(std::string file_name);
  ~Import() {
    delete[] buffer_;
    buffer_ = nullptr;
  }
};
}  // namespace lite
}  // namespace mindspore

#endif  //  MINDSPORE_LITE_SRC_EXPRESSION_IMPORT_H_
