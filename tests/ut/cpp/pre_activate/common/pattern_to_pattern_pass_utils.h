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

#ifndef MINDSPORE_TESTS_UT_CPP_PRE_ACTIVATE_COMMON_PATTERN_TO_PATTERN_PASS_UTILS_H_
#define MINDSPORE_TESTS_UT_CPP_PRE_ACTIVATE_COMMON_PATTERN_TO_PATTERN_PASS_UTILS_H_

#include <vector>
#include <memory>
#include "common/common_test.h"
#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"

#define private public
#define protected public
#include "backend/common/optimizer/pattern_to_pattern.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class CheckPattern {
 public:
  CheckPattern()
      : m_(std::make_shared<PatternMap>()),
        src_pattern_(SrcPattern(m_)),
        pattern_engine_(PatternEngine(std::make_shared<Visitor>())),
        primitive_vars_(std::make_shared<PrimitiveVarMap>()),
        equiv_(std::make_shared<Equiv>()){};
  bool build_pattern_map(const AnfNodePtr &node) {
    VarPtr root_g = std::make_shared<Var>("RootG");
    auto src_pattern_root = SexpToNode(src_pattern_.GetRoot(), root_g, primitive_vars_.get(), multigraph_);
    auto primitive = GetCNodePrimitive(src_pattern_root);
    if (IsPrimitiveCNode(node, primitive)) {
      MS_EXCEPTION_IF_NULL(primitive_vars_);
      MS_EXCEPTION_IF_NULL(equiv_);
      equiv_->clear();
      EquivPtr equiv = pattern_engine_.Match(src_pattern_root, node, *primitive_vars_, equiv_);
      if (equiv != nullptr && !equiv->empty()) {
        return src_pattern_.build_pattern_map(node, equiv);
      }
    }
    return false;
  }
  PatternMapPtr m_;
  SrcPattern src_pattern_;
  PatternEngine pattern_engine_;
  PrimitiveVarMapPtr primitive_vars_;
  EquivPtr equiv_;
  bool multigraph_ = true;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_TESTS_UT_CPP_PRE_ACTIVATE_COMMON_PATTERN_TO_PATTERN_PASS_UTILS_H_
