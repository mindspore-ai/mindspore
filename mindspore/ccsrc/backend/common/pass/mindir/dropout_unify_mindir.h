/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/common/pass_manager/pattern_to_pattern.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT DropoutUnifyMindIR1 : public PatternProcessPass {
 public:
  explicit DropoutUnifyMindIR1(const std::string &name = "dropout_unify_mindir1", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {}
  ~DropoutUnifyMindIR1() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  void EnableKeepProb() { enable_keep_prob_ = true; }

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
  bool enable_keep_prob_ = false;
};

class BACKEND_COMMON_EXPORT DropoutGradUnifyMindIR : public PatternToPatternPass {
 public:
  DropoutGradUnifyMindIR() : PatternToPatternPass("dropoutgrad_unify_mindir", true) {}
  ~DropoutGradUnifyMindIR() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
};

class BACKEND_COMMON_EXPORT DropoutExtUnifyMindIR1 : public DropoutUnifyMindIR1 {
 public:
  explicit DropoutExtUnifyMindIR1(bool multigraph = true)
      : DropoutUnifyMindIR1("dropout_ext_unify_mindir1", multigraph) {
    EnableKeepProb();
  }
  ~DropoutExtUnifyMindIR1() override = default;
  const BaseRef DefinePattern() const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class BACKEND_COMMON_EXPORT DropoutGradExtUnifyMindIR : public PatternProcessPass {
 public:
  explicit DropoutGradExtUnifyMindIR(bool multigraph = true)
      : PatternProcessPass("dropoutgrad_ext_unify_mindir", multigraph) {}
  ~DropoutGradExtUnifyMindIR() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_DROPOUT_UNIFY_MINDIR_H_
