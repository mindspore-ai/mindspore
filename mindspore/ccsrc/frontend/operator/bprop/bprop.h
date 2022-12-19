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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_BPROP_BPROP_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_BPROP_BPROP_H_

#include <map>
#include <vector>
#include <utility>
#include "ir/anf.h"
#include "frontend/operator/bprop/bprop_irbuilder.h"
#include "include/common/visible.h"

namespace mindspore {
using DoutUserType = std::vector<std::pair<CNodePtr, int>>;
// deprecated
void BuildBprop(const CNodePtr &cnode, CNodePtrList *outputs, DoutUserType *dout_user);

using UserType = std::map<AnfNodePtr, std::vector<std::pair<CNodePtr, int>>>;
bool BuildBprop(const CNodePtr &cnode, CNodePtrList *outputs, UserType *users);

#ifdef _MSC_VER
class WinBpropRegister {
 public:
  WinBpropRegister() { expander::bprop::RegBpropExpanders(); }
  ~WinBpropRegister() {}
  void EmptyFunc() const {}
};
#endif
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_BPROP_BPROP_H_
