/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/tbe/tbe_kernel_select/tbe_property_checker.h"
#include <map>
#include <string>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/parallel/ops_info/ops_utils.h"

namespace mindspore {
namespace kernel {
using CheckSupportFun = bool (*)(const CNodePtr &cnode);

constexpr char kAttrStrides[] = "strides";

static bool CheckStridedSlice(const CNodePtr &cnode) {
  // check stride[-1] != 1 TODO
  if (AnfAlgo::HasNodeAttr(kAttrStrides, cnode)) {
    auto strides = AnfAlgo::GetNodeAttr<std::vector<int>>(cnode, kAttrStrides);
    if (!strides.empty() && strides[strides.size() - 1] == 1) {
      return true;
    }
  }
  // last tensor TODO
  return true;
}

bool TbePropertyChecker::CheckTbeProperties(const mindspore::CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  static std::map<std::string, CheckSupportFun> tbe_property_checker = {{parallel::KStridedSlice, CheckStridedSlice}};
  auto cnode_type = AnfAlgo::GetCNodeName(cnode);
  auto find_iter = tbe_property_checker.find(cnode_type);
  if (find_iter != tbe_property_checker.end()) {
    return find_iter->second(cnode);
  }
  return true;
}

}  // namespace kernel
}  // namespace mindspore
