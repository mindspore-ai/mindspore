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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_DROPOUT_FOR_GE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_DROPOUT_FOR_GE_H_

#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
class DropoutForGE : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &opt, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override { is_match_ = true; }

  void Reset() { is_match_ = false; }

 private:
  bool is_match_{false};
};

class DropoutGradForGE : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &opt, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override { is_match_ = true; }

  void Reset() { is_match_ = false; }

 private:
  bool is_match_{false};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_DROPOUT_FOR_GE_H_
