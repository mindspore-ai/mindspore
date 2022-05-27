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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DEFORMABLE_OFFSETS_FISSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DEFORMABLE_OFFSETS_FISSION_H_

#include <vector>
#include "backend/common/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class DeformableOffsetsFission : public PatternProcessPass {
 public:
  explicit DeformableOffsetsFission(bool multigraph = true)
      : PatternProcessPass("deformable_offsets_fission", multigraph) {}
  ~DeformableOffsetsFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  ValueNodePtr CreateHelperNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                const std::vector<size_t> &offset_shape, const std::vector<int64_t> &kernel_sizes,
                                const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                                const std::vector<int64_t> &dilations, const size_t axis_h, const size_t axis_w,
                                const size_t axis_c) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_DEFORMABLE_OFFSETS_FISSION_H_
