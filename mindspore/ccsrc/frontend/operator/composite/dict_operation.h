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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_DICT_OPERATION_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_DICT_OPERATION_H_

#include <string>
#include <unordered_map>
#include <memory>

#include "ir/meta_func_graph.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
class DictClear : public MetaFuncGraph {
 public:
  explicit DictClear(const std::string &name) : MetaFuncGraph(name) {}
  ~DictClear() override = default;
  MS_DECLARE_PARENT(DictClear, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const DictClear &dict_clear) {
    os << dict_clear.name_;
    return os;
  }
  friend bool operator==(const DictClear &lhs, const DictClear &rhs) { return lhs.name_ == rhs.name_; }
};
using DictClearPtr = std::shared_ptr<DictClear>;

class DictHasKey : public MetaFuncGraph {
 public:
  explicit DictHasKey(const std::string &name) : MetaFuncGraph(name) {}
  ~DictHasKey() override = default;
  MS_DECLARE_PARENT(DictHasKey, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const DictHasKey &dict_has_key) {
    os << dict_has_key.name_;
    return os;
  }
  friend bool operator==(const DictHasKey &lhs, const DictHasKey &rhs) { return lhs.name_ == rhs.name_; }
};
using DictHasKeyPtr = std::shared_ptr<DictHasKey>;

class DictUpdate : public MetaFuncGraph {
 public:
  explicit DictUpdate(const std::string &name) : MetaFuncGraph(name) {}
  ~DictUpdate() override = default;
  MS_DECLARE_PARENT(DictUpdate, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const DictUpdate &dict_update) {
    os << dict_update.name_;
    return os;
  }
  friend bool operator==(const DictUpdate &lhs, const DictUpdate &rhs) { return lhs.name_ == rhs.name_; }
  void AddNodeToLists(const AbstractBasePtr &arg, const FuncGraphPtr &ret, AnfNodePtrList *keys, AnfNodePtrList *values,
                      std::unordered_map<std::string, size_t> *hash_map);
};
using DictUpdatePtr = std::shared_ptr<DictUpdate>;

class DictFromKeys : public MetaFuncGraph {
 public:
  explicit DictFromKeys(const std::string &name) : MetaFuncGraph(name) {}
  ~DictFromKeys() override = default;
  MS_DECLARE_PARENT(DictFromKeys, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const DictFromKeys &dict_from_keys) {
    os << dict_from_keys.name_;
    return os;
  }
  friend bool operator==(const DictFromKeys &lhs, const DictFromKeys &rhs) { return lhs.name_ == rhs.name_; }
  abstract::AbstractBasePtrList ParseIterableObject(const abstract::AbstractBasePtr &arg_key);
};
using DictFromKeysPtr = std::shared_ptr<DictFromKeys>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_DICT_OPERATION_H_
