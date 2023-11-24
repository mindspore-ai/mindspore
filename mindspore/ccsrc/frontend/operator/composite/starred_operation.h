/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_STARRED_OPERATION_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_STARRED_OPERATION_H_

#include <string>
#include <map>
#include <set>
#include <memory>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "utils/misc.h"
#include "utils/any.h"
#include "ir/dtype.h"
#include "ir/meta_func_graph.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using AbstractBasePtr = abstract::AbstractBasePtr;
using AbstractBasePtrList = abstract::AbstractBasePtrList;
using AbstractTuplePtr = abstract::AbstractTuplePtr;

class StarredGetItem : public MetaFuncGraph {
 public:
  explicit StarredGetItem(const std::string &name) : MetaFuncGraph(name) {}
  ~StarredGetItem() override = default;
  MS_DECLARE_PARENT(StarredGetItem, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend std::ostream &operator<<(std::ostream &os, const StarredGetItem &op) {
    os << op.name_;
    return os;
  }
  friend bool operator==(const StarredGetItem &lhs, const StarredGetItem &rhs) { return lhs.name_ == rhs.name_; }
};
using StarredGetItemPtr = std::shared_ptr<StarredGetItem>;

class StarredUnpack : public MetaFuncGraph {
 public:
  explicit StarredUnpack(const std::string &name) : MetaFuncGraph(name) {}
  ~StarredUnpack() override = default;
  MS_DECLARE_PARENT(StarredUnpack, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend std::ostream &operator<<(std::ostream &os, const StarredUnpack &op) {
    os << op.name_;
    return os;
  }
  friend bool operator==(const StarredUnpack &lhs, const StarredUnpack &rhs) { return lhs.name_ == rhs.name_; }
};
using StarredUnpackPtr = std::shared_ptr<StarredUnpack>;

class StarredUnpackMerge : public MetaFuncGraph {
 public:
  explicit StarredUnpackMerge(const std::string &name) : MetaFuncGraph(name) {}
  ~StarredUnpackMerge() override = default;
  MS_DECLARE_PARENT(StarredUnpackMerge, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend std::ostream &operator<<(std::ostream &os, const StarredUnpackMerge &op) {
    os << op.name_;
    return os;
  }
  friend bool operator==(const StarredUnpackMerge &lhs, const StarredUnpackMerge &rhs) {
    return lhs.name_ == rhs.name_;
  }
  std::pair<std::vector<int64_t>, int64_t> GetStarredUnpackMergeFlags(const AbstractBasePtrList &args_abs_list);
};
using StarredUnpackMergePtr = std::shared_ptr<StarredUnpackMerge>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_STARRED_OPERATION_H_
