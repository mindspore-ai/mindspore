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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_OPTIMIZER_UNIFY_OUTPUT_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_OPTIMIZER_UNIFY_OUTPUT_H_

#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pattern_to_pattern.h"

namespace mindspore {
namespace opt {
class BuildTupleGetFunc {
 public:
  explicit BuildTupleGetFunc(const size_t output_size) : output_size_(output_size) {}
  AnfNodePtr operator()(const PatternMap &m, const AnfNodePtr &get_item) const;
  size_t output_size_;
};
class FtrlUnifyOutput : public PatternToPatternPass {
 public:
  FtrlUnifyOutput() : PatternToPatternPass("ftrl_unify_output", true) {}
  ~FtrlUnifyOutput() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override;
};
class MomentumUnifyOutput : public PatternToPatternPass {
 public:
  MomentumUnifyOutput() : PatternToPatternPass("momentum_unify_output", true) {}
  ~MomentumUnifyOutput() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override;
};
class CenteredRMSPropUnifyOutput : public PatternToPatternPass {
 public:
  CenteredRMSPropUnifyOutput() : PatternToPatternPass("centered_rmsprop_unify_output", true) {}
  ~CenteredRMSPropUnifyOutput() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override;
};
class RMSPropUnifyOutput : public PatternToPatternPass {
 public:
  RMSPropUnifyOutput() : PatternToPatternPass("rmsprop_unify_output", true) {}
  ~RMSPropUnifyOutput() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_MINDIR_OPTIMIZER_UNIFY_OUTPUT_H_
