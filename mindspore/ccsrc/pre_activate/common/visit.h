/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_COMMON_VISIT_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_COMMON_VISIT_H_

#include <unordered_map>
#include <stdexcept>
#include <list>
#include <vector>
#include <string>
#include <memory>

#include "ir/base.h"
#include "utils/base_ref.h"

// namespace to support utils definition
namespace mindspore {
using VisitFn = std::function<BaseRef(const BaseRef &)>;

class Visitor {
 public:
  virtual void SetFn(VisitFn fn) = 0;
  virtual bool Visit(const BaseRef &e, BaseRef *out) const = 0;
  virtual bool Visit(const VectorRef &e, BaseRef *out) const = 0;
  virtual ~Visitor() = default;
};

class DefaultVisitor : public Visitor {
 public:
  DefaultVisitor() : fn_(nullptr) {}
  ~DefaultVisitor() override = default;
  void SetFn(VisitFn fn) override { fn_ = fn; };
  bool Visit(const VectorRef &e, BaseRef *out) const override;
  bool Visit(const BaseRef &e, BaseRef *out) const override;
  void Visit(const AnfNodePtr &node, const VisitFn &fn, AnfNodePtr *output) const;
  void Visit(const CNodePtr &cnode, const VisitFn &fn, AnfNodePtr *output) const;
  void Visit(const ValueNodePtr &vnode, const VisitFn &fn, AnfNodePtr *output) const;

  VisitFn fn_;
};

std::shared_ptr<VectorRef> ExpandList(const std::vector<BaseRef> &list);
bool CheckIfNeedExpand(const std::vector<BaseRef> &list);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_COMMON_VISIT_H_
