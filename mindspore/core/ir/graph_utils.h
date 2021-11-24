/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_GRAPH_UTILS_H_
#define MINDSPORE_CORE_IR_GRAPH_UTILS_H_

#include <utility>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <functional>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "utils/label.h"

namespace mindspore {
enum IncludeType { FOLLOW, NOFOLLOW, EXCLUDE };

using IncludeFunc = std::function<IncludeType(const AnfNodePtr &)>;
using FilterFunc = std::function<bool(const AnfNodePtr &)>;
using SuccFunc = std::function<std::vector<AnfNodePtr>(AnfNodePtr)>;
using SearchFunc = std::function<std::vector<AnfNodePtr>(const AnfNodePtr &, const IncludeFunc &)>;
using MatchFunc = std::function<bool(const CNodePtr &)>;
using NodeVisitFunc = std::function<void(const AnfNodePtr &)>;

std::vector<AnfNodePtr> SuccDeeper(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccDeeperSimple(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccIncoming(const AnfNodePtr &node);
std::vector<AnfNodePtr> SuccIncludeFV(const FuncGraphPtr &fg, const AnfNodePtr &node);

const std::vector<AnfNodePtr> &GetInputs(const AnfNodePtr &node);

inline IncludeType AlwaysInclude(const AnfNodePtr &) { return FOLLOW; }
IncludeType IncludeBelongGraph(const FuncGraphPtr &fg, const AnfNodePtr &node);

std::vector<AnfNodePtr> DeepScopedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include = AlwaysInclude);

std::vector<AnfNodePtr> DeepLinkedGraphSearch(const AnfNodePtr &root, const IncludeFunc &include = AlwaysInclude);

std::vector<AnfNodePtr> DeepScopedGraphSearchWithFilter(const AnfNodePtr &root, const IncludeFunc &include,
                                                        const FilterFunc &filter);

std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &root, const SuccFunc &succ = SuccIncoming,
                                 const IncludeFunc &include = AlwaysInclude);

std::vector<CNodePtr> BroadFirstSearchGraphCNodes(const CNodePtr &start);
std::vector<FuncGraphPtr> BroadFirstSearchGraphUsed(const FuncGraphPtr &root);

CNodePtr BroadFirstSearchFirstOf(const std::vector<CNodePtr> &starts, const MatchFunc &match_predicate);

}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_GRAPH_UTILS_H_
