/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_LIST_OPERATION_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_LIST_OPERATION_H_

#include <string>
#include <memory>
#include <vector>

#include "ir/meta_func_graph.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
class ListAppend : public MetaFuncGraph {
 public:
  explicit ListAppend(const std::string &name) : MetaFuncGraph(name) {}
  ~ListAppend() override = default;
  MS_DECLARE_PARENT(ListAppend, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) override;
  friend std::ostream &operator<<(std::ostream &os, const ListAppend &list_append) {
    os << list_append.name_;
    return os;
  }
  friend bool operator==(const ListAppend &lhs, const ListAppend &rhs) { return lhs.name_ == rhs.name_; }
};
using ListAppendPtr = std::shared_ptr<ListAppend>;

class ListInsert : public MetaFuncGraph {
 public:
  explicit ListInsert(const std::string &name) : MetaFuncGraph(name) {}
  ~ListInsert() override = default;
  MS_DECLARE_PARENT(ListInsert, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) override;
  friend std::ostream &operator<<(std::ostream &os, const ListInsert &list_insert) {
    os << list_insert.name_;
    return os;
  }
  friend bool operator==(const ListInsert &lhs, const ListInsert &rhs) { return lhs.name_ == rhs.name_; }
};
using ListInsertPtr = std::shared_ptr<ListInsert>;

class ListPop : public MetaFuncGraph {
 public:
  explicit ListPop(const std::string &name) : MetaFuncGraph(name) {}
  ~ListPop() override = default;
  MS_DECLARE_PARENT(ListPop, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const ListPop &list_pop) {
    os << list_pop.name_;
    return os;
  }
  friend bool operator==(const ListPop &lhs, const ListPop &rhs) { return lhs.name_ == rhs.name_; }
};
using ListPopPtr = std::shared_ptr<ListPop>;

class ListClear : public MetaFuncGraph {
 public:
  explicit ListClear(const std::string &name) : MetaFuncGraph(name) {}
  ~ListClear() override = default;
  MS_DECLARE_PARENT(ListClear, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const ListClear &list_clear) {
    os << list_clear.name_;
    return os;
  }
  friend bool operator==(const ListClear &lhs, const ListClear &rhs) { return lhs.name_ == rhs.name_; }
};
using ListClearPtr = std::shared_ptr<ListClear>;

class ListExtend : public MetaFuncGraph {
 public:
  explicit ListExtend(const std::string &name) : MetaFuncGraph(name) {}
  ~ListExtend() override = default;
  MS_DECLARE_PARENT(ListExtend, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const ListExtend &list_extend) {
    os << list_extend.name_;
    return os;
  }
  friend bool operator==(const ListExtend &lhs, const ListExtend &rhs) { return lhs.name_ == rhs.name_; }

 private:
  static void AddNodeToElems(const AbstractBasePtr &arg, const FuncGraphPtr &ret, std::vector<AnfNodePtr> *elems);
};
using ListExtendPtr = std::shared_ptr<ListExtend>;

class ListReverse : public MetaFuncGraph {
 public:
  explicit ListReverse(const std::string &name) : MetaFuncGraph(name) {}
  ~ListReverse() override = default;
  MS_DECLARE_PARENT(ListReverse, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const ListReverse &list_reverse) {
    os << list_reverse.name_;
    return os;
  }
  friend bool operator==(const ListReverse &lhs, const ListReverse &rhs) { return lhs.name_ == rhs.name_; }
};
using ListReversePtr = std::shared_ptr<ListReverse>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_LIST_OPERATION_H_
