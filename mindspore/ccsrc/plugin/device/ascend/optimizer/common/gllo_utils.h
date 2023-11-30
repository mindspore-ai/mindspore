/**
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_GLLO_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_GLLO_UTILS_H_
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ir/anf.h"
#include "base/base_ref.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
using STATUS = int;
constexpr int RET_OK = 0;
constexpr int RET_ERROR = -1;
constexpr int RET_NULL_PTR = -2;  // NULL pointer returned.

bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type);
bool IsParamNode(const BaseRef &n);
bool IsMarkedTrainOp(const CNodePtr &cnode);

#define MS_CHECK_TRUE_RET(value, errcode) \
  do {                                    \
    if (!(value)) {                       \
      return errcode;                     \
    }                                     \
  } while (0)

class ReturnCode {
 public:
  static ReturnCode *GetSingleReturnCode() {
    static ReturnCode return_code;
    return &return_code;
  }
  void UpdateReturnCode(STATUS status) {
    if (status_code_ == RET_OK) {
      status_code_ = status;
    }
  }

 private:
  ReturnCode() = default;
  virtual ~ReturnCode() = default;
  int status_code_ = RET_OK;
};

template <const PrimitivePtr *prim = nullptr>
inline bool IsSpecifiedNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, *prim);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_GLLO_UTILS_H_
