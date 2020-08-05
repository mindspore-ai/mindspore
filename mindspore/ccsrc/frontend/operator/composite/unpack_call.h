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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_UNPACK_CALL_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_UNPACK_CALL_H_

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <map>
#include <set>
#include <memory>

#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/misc.h"
#include "utils/any.h"
#include "ir/dtype.h"
#include "ir/meta_func_graph.h"
#include "utils/ms_utils.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
// Expand the tuple and dict parameters generated when parsing the function call,
// and generate positional parameters and key-value pairs for function.
class UnpackCall : public MetaFuncGraph {
 public:
  explicit UnpackCall(const std::string &name) : MetaFuncGraph(name) {}
  ~UnpackCall() override = default;
  MS_DECLARE_PARENT(UnpackCall, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const UnpackCall &lhs, const UnpackCall &rhs) { return lhs.name_ == rhs.name_; }
};
using UnpackCallPtr = std::shared_ptr<UnpackCall>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_UNPACK_CALL_H_
