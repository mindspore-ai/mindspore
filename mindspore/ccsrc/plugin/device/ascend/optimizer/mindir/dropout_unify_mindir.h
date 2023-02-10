/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_DROPOUT_UNIFY_MINDIR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_DROPOUT_UNIFY_MINDIR_H_

#include <memory>
#include <string>
#include <vector>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pattern_to_pattern.h"

namespace mindspore {
namespace opt {
class DropoutAndDropoutGradUnifyMindIR : public PatternProcessPass {
 public:
  explicit DropoutAndDropoutGradUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("dropout_and_dropoutgrad_unify_mindir", multigraph) {
    grad_input_ = std::make_shared<Var>();
  }
  ~DropoutAndDropoutGradUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr grad_input_;
};

class DropoutUnifyMindIR0 : public PatternProcessPass {
 public:
  explicit DropoutUnifyMindIR0(bool multigraph = true) : PatternProcessPass("dropout_unify_mindir0", multigraph) {}
  ~DropoutUnifyMindIR0() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class DropoutUnifyMindIR1 : public PatternProcessPass {
 public:
  explicit DropoutUnifyMindIR1(bool multigraph = true) : PatternProcessPass("dropout_unify_mindir1", multigraph) {}
  ~DropoutUnifyMindIR1() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class DropoutGradUnifyMindIR : public PatternToPatternPass {
 public:
  DropoutGradUnifyMindIR() : PatternToPatternPass("dropoutgrad_unify_mindir", true) {}
  ~DropoutGradUnifyMindIR() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_DROPOUT_UNIFY_MINDIR_H_
