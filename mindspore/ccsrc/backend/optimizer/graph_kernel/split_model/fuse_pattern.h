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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_FUSE_PATTERN_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_FUSE_PATTERN_H_

#include <string>
#include <vector>
#include <memory>
#include "backend/optimizer/graph_kernel/split_model/area.h"

namespace mindspore::graphkernel::inner {
class CircleChecker {
 public:
  // whether it will form a circle if the two areas are fused.
  virtual bool HasCircle(const AreaPtr &a, const AreaPtr &b) const = 0;
};
using CircleCheckerPtr = std::shared_ptr<CircleChecker>;

enum class FuseDirection {
  FORWARD,  // fuse with inputs
  BACKWARD  // fuse with outputs
};

// the base class of fusion patterns
class FusePattern {
 public:
  explicit FusePattern(const std::string &name) : name_(name) {}
  virtual ~FusePattern() = default;
  // Run the pattern
  bool Run(const AreaPtr &dom) {
    Reset();
    return Check(dom) && Match(dom);
  }
  std::string ToString() const;
  // Bind the circle checker
  void SetCircleChecker(const CircleCheckerPtr &c) { circle_checker_ = c; }

  std::string name() const { return name_; }
  FuseDirection direction() const { return direction_; }
  std::vector<AreaPtr> &fused_areas() { return fused_areas_; }

 protected:
  void Reset() { fused_areas_.clear(); }
  // Check whether the pattern can handle this area
  virtual bool Check(const AreaPtr &) { return true; }
  // Match the ADJACENT areas of `dom`
  virtual bool Match(const AreaPtr &dom) = 0;
  // whether it will form a circle if the two areas are fused.
  bool HasCircle(const AreaPtr &a, const AreaPtr &b) const {
    MS_EXCEPTION_IF_NULL(circle_checker_);
    return circle_checker_->HasCircle(a, b);
  }

  std::string name_;
  std::vector<AreaPtr> fused_areas_;
  FuseDirection direction_{FuseDirection::FORWARD};
  CircleCheckerPtr circle_checker_{nullptr};
};
using FusePatternPtr = std::shared_ptr<FusePattern>;

/* some common patterns are defined below */

// bind the virtual nodes to their inputs
class FuseVirtualNode : public FusePattern {
 public:
  FuseVirtualNode() : FusePattern("bind_virtual_node") { direction_ = FuseDirection::FORWARD; }
  ~FuseVirtualNode() = default;

 protected:
  bool Check(const AreaPtr &area) override { return area->pattern() == NodePattern::VIRTUAL; }
  bool Match(const AreaPtr &area) override;
};
}  // namespace mindspore::graphkernel::inner
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SPLIT_MODEL_FUSE_PATTERN_H_
