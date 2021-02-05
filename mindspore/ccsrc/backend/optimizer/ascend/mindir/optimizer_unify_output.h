/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_OPTIMIZER_UNIFY_OUTPUT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_OPTIMIZER_UNIFY_OUTPUT_H_

#include <memory>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class FtrlUnifyOutput : public PatternProcessPass {
 public:
  explicit FtrlUnifyOutput(bool multigraph = true) : PatternProcessPass("ftrl_unify_output", multigraph) {}
  ~FtrlUnifyOutput() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class MomentumUnifyOutput : public PatternProcessPass {
 public:
  explicit MomentumUnifyOutput(bool multigraph = true) : PatternProcessPass("momentum_unify_output", multigraph) {}
  ~MomentumUnifyOutput() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class CenteredRMSPropUnifyOutput : public PatternProcessPass {
 public:
  explicit CenteredRMSPropUnifyOutput(bool multigraph = true)
      : PatternProcessPass("centered_rmsprop_unify_output", multigraph) {}
  ~CenteredRMSPropUnifyOutput() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};

class RMSPropUnifyOutput : public PatternProcessPass {
 public:
  explicit RMSPropUnifyOutput(bool multigraph = true) : PatternProcessPass("rmsprop_unify_output", multigraph) {}
  ~RMSPropUnifyOutput() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_OPTIMIZER_UNIFY_OUTPUT_H_
