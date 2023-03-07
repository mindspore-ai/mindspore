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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_SPLIT_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_SPLIT_MODEL_H_

#include <vector>
#include <list>
#include <memory>
#include <set>
#include <utility>
#include "backend/common/graph_kernel/model/lite_graph.h"
#include "backend/common/graph_kernel/split_model/area.h"
#include "backend/common/graph_kernel/split_model/fuse_pattern.h"

namespace mindspore::graphkernel::inner {
class ReachTable : public CircleChecker {
 public:
  explicit ReachTable(size_t size);
  virtual ~ReachTable() = default;
  bool HasCircle(const AreaPtr &a, const AreaPtr &b) const override;

  // Link area from `from` to `to`.
  void Link(size_t from, size_t to);

  // Fuse the area `target` and `other`. After that, the `other` area will be discarded.
  void FuseArea(size_t target, size_t other);

 private:
  // check the reachability from `from` to `to`
  bool Reachable(size_t from, size_t to) const { return reach_[from][to]; }

  size_t size_;
  std::vector<std::vector<bool>> reach_;
  std::set<size_t> alive_;
};

class SplitModel {
 public:
  void Run(const LiteGraphPtr &litegraph);
  const std::list<AreaPtr> &areas() const { return areas_; }
  SplitModel() = default;
  virtual ~SplitModel() = default;

 protected:
  // transform the litegraph to areas, and initialize inner tables.
  void InitGraph(const LiteGraphPtr &litegraph);
  // Push leading "1" to shapes to facilitate pattern match.
  void AlignShape(const LiteGraphPtr &litegraph) const;
  // initialize fusion pattern list.
  virtual void InitFusePatterns() = 0;
  bool RunOnePattern(const FusePatternPtr &pattern);
  // fuse areas by pattern
  void RunFusePatterns();
  // set default area mode when the area has only one node.
  void SetDefaultAreaMode(const AreaPtr &area) const { area->SetMode(GetDefaultAreaMode(area->dom())); }
  // get default area mode of the dominant node
  virtual AreaMode GetDefaultAreaMode(const PrimOpPtr &node) const = 0;
  // add new pattern
  void AddPattern(const std::shared_ptr<FusePattern> &pn, bool enable = true);
  // fuse areas
  void FuseAreas(const AreaPtr &dom, const std::vector<AreaPtr> &areas, FuseDirection direction);
  // create new area
  AreaPtr NewArea(const PrimOpPtr &op, bool is_output);
  // limit the area's size
  void LimitAreaSize(const AreaPtr &dom, std::vector<AreaPtr> *areas, size_t max_size = 200) const;

  std::list<AreaPtr> areas_;  // use std::list to accelerate the "erase"
  std::shared_ptr<ReachTable> reach_table_{nullptr};
  HashMap<NodePtr, AreaPtr> node_area_map_;

 private:
  size_t cur_area_id_{0};
  std::vector<std::pair<FusePatternPtr, bool>> patterns_;
};
using SplitModelPtr = std::shared_ptr<SplitModel>;
}  // namespace mindspore::graphkernel::inner
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_SPLIT_MODEL_H_
