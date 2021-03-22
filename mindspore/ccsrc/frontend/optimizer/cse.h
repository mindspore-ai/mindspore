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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CSE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CSE_H_

#include <vector>
#include <unordered_map>
#include <memory>
#include "ir/anf.h"
#include "ir/manager.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
// Common subexpression elimination.
class CSE {
 public:
  CSE() = default;
  virtual ~CSE() = default;

  virtual bool CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node, bool check_side_effect = true) const;

  virtual bool CheckRandomEffect(const AnfNodePtr &main, const AnfNodePtr &node) const;

  bool Cse(const FuncGraphPtr root, const FuncGraphManagerPtr manager) const;

 private:
  bool BuildOrderGroupAndDoReplace(const FuncGraphManagerPtr manager) const;
  bool DoReplace(const FuncGraphManagerPtr manager, const std::vector<std::size_t> &order_group,
                 std::unordered_map<std::size_t, std::vector<AnfNodePtr>> *groups) const;
};

BasePtr AbsOf(const AnfNodePtr &node, bool ignore_fg_abs_tracking_id = false);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CSE_H_
