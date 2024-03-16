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
#include "tools/graph_kernel/converter/split_model_ascend_cce.h"
#include <memory>
#include "utils/ms_context.h"

namespace mindspore::graphkernel::inner {

class FuseAddReshapeTranspose : public FusePattern {
 public:
  FuseAddReshapeTranspose() : FusePattern("FuseAddReshapeTranspose") { direction_ = FuseDirection::BACKWARD; }
  ~FuseAddReshapeTranspose() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return dom->dom()->op() == "Add" && dom->size() == 1; }
  bool Match(const AreaPtr &dom) override {
    auto reshape = dom->users();
    if (reshape.size() == 0 || reshape[0]->dom()->op() != "Reshape") {
      return false;
    }
    auto transpose = reshape[0]->users();
    if (transpose.size() == 0 || transpose[0]->dom()->op() != "Transpose") {
      return false;
    }
    fused_areas_ = {reshape[0], transpose[0]};
    return !fused_areas_.empty();
  }
};

class FuseBufferFusionOps : public FusePattern {
 public:
  FuseBufferFusionOps() : FusePattern("FuseBufferFusionOps") { direction_ = FuseDirection::BACKWARD; }
  ~FuseBufferFusionOps() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return (dom->dom()->op() == "StridedSliceV2"); }
  bool Match(const AreaPtr &dom) override {
    auto elemwise = dom->users();
    if (elemwise.size() == 0) {
      return false;
    }
    if (elemwise[0]->dom()->op() == "Mul" || elemwise[0]->dom()->op() == "FastGeLU" ||
        elemwise[0]->dom()->op() == "StridedSliceV2") {
      fused_areas_.emplace_back(elemwise[0]);
    }
    return !fused_areas_.empty();
  }
};

class FuseBMMReshapeTranspose : public FusePattern {
 public:
  FuseBMMReshapeTranspose() : FusePattern("FuseBMMReshapeTranspose") { direction_ = FuseDirection::BACKWARD; }
  ~FuseBMMReshapeTranspose() = default;

 protected:
  bool Check(const AreaPtr &dom) override { return dom->dom()->op() == "BatchMatMul" && dom->size() == 1; }
  bool Match(const AreaPtr &dom) override {
    auto reshape = dom->users();
    if (reshape.size() == 0 || reshape[0]->dom()->op() != "Reshape") {
      return false;
    }
    auto transpose = reshape[0]->users();
    if (transpose.size() == 0 || transpose[0]->dom()->op() != "Transpose") {
      return false;
    }
    fused_areas_ = {reshape[0], transpose[0]};
    return !fused_areas_.empty();
  }
};

void SplitModelAscendCCE::InitDefaultAreaOps() {
  default_area_op_ = {"MatMul", "BatchMatMul", "PagedAttention", "PagedAttentionMask", "ReshapeAndCache"};
}

void SplitModelAscendCCE::AddCceOpPattern(std::string &&pattern_name, FusePatterns &&patterns) {
  auto cce_lib_splitter_pattern = graphkernel::GraphKernelFlags::GetInstance().cce_lib_splitter_pattern;
  auto it = std::find(cce_lib_splitter_pattern.begin(), cce_lib_splitter_pattern.end(), pattern_name);
  if (it != cce_lib_splitter_pattern.end()) {
    for (auto &p : patterns) {
      AddPattern(p, true);
    }
    MS_LOG(INFO) << "Add cce op pattern <<" << pattern_name;
  }
}

void SplitModelAscendCCE::InitFusePatterns() {
  AddCceOpPattern("FuseAddReshapeTranspose", {std::make_shared<FuseAddReshapeTranspose>()});
  // call FuseBufferFusionOps twice to fuse two StridedSlice
  AddCceOpPattern("FuseBufferFusionOps",
                  {std::make_shared<FuseBufferFusionOps>(), std::make_shared<FuseBufferFusionOps>()});
  AddCceOpPattern("FuseBMMReshapeTranspose", {std::make_shared<FuseBMMReshapeTranspose>()});
}

AreaMode SplitModelAscendCCE::GetDefaultAreaMode(const PrimOpPtr &node) const {
  if (node != nullptr && default_area_op_.find(node->op()) != default_area_op_.end()) {
    return AreaMode::COMPOSITE;
  }
  return AreaMode::BASIC;
}
}  // namespace mindspore::graphkernel::inner
