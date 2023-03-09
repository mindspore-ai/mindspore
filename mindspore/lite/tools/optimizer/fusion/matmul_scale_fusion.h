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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MATMUL_SCALE_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MATMUL_SCALE_FUSION_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "include/backend/optimizer/optimizer.h"
#include "tools/optimizer/fusion/scale_base_fusion.h"
#include "ops/mat_mul.h"

namespace mindspore::opt {
class MatMulScaleFusion : public ScaleBaseFusion {
 public:
  explicit MatMulScaleFusion(bool multigraph = true) : ScaleBaseFusion("MatMulScaleFusion", multigraph) {}
  ~MatMulScaleFusion() override = default;
  const BaseRef DefinePattern() const override;

 protected:
  bool CheckPrevCnodeProper(const CNodePtr &prev_cnode) const override;
  PrimitiveCPtr BuildNewPrimitive(const CNodePtr &curr_cnode, const CNodePtr &prev_cnode) const override;
  int CalNewBiasImpl(float *curr_weight_data, float *curr_bias_data, std::vector<int64_t> prev_bias_shape,
                     float *prev_bias_data) const override;
  int CalNewScaleImpl(float *curr_weight_data, std::vector<int64_t> prev_weight_shape, float *prev_weight_data,
                      const AnfNodePtr &prim) const override;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_MATMUL_SCALE_FUSION_H_
