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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_VISIT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_VISIT_H_

#include <unordered_map>
#include <stdexcept>
#include <list>
#include <vector>
#include <string>
#include <memory>

#include "base/base.h"
#include "base/base_ref.h"

// namespace to support utils definition
namespace mindspore {
class Visitor {
 public:
  Visitor() {}
  ~Visitor() = default;
  bool Visit(const VectorRef &e, VectorRef *const values_ref, BaseRef *out) const;
  bool Visit(const BaseRef &e, VectorRef *const values_ref, BaseRef *out) const;
  void Visit(const AnfNodePtr &node, VectorRef *const values_ref, AnfNodePtr *output) const;
  void Visit(const CNodePtr &cnode, VectorRef *const values_ref, AnfNodePtr *output) const;
  void Visit(const ValueNodePtr &vnode, VectorRef *const values_ref, AnfNodePtr *output) const;
};

std::shared_ptr<VectorRef> ExpandList(const std::vector<BaseRef> &list);
bool CheckIfNeedExpand(const std::vector<BaseRef> &list);
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_VISIT_H_
