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

#ifndef MINDSPORE_CORE_OPS_SPARSE_SEGMENT_SUM_GRAD_H_
#define MINDSPORE_CORE_OPS_SPARSE_SEGMENT_SUM_GRAD_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseSegmentSumGrad = "SparseSegmentSumGrad";
class OPS_API SparseSegmentSumGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseSegmentSumGrad);
  SparseSegmentSumGrad() : BaseOperator(kNameSparseSegmentSumGrad) {
    InitIOName({"grad", "indices", "segment_ids", "output_dim0"}, {"output"});
  }
};

OPS_API abstract::AbstractBasePtr SparseSegmentSumGradInfer(const abstract::AnalysisEnginePtr &,
                                                            const PrimitivePtr &primitive,
                                                            const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimSparseSegmentSumGradPtr = std::shared_ptr<SparseSegmentSumGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_SEGMENT_SQRT_N_GRAD_H_
