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

#ifndef MINDSPORE_CORE_OPS_SPARSE_SEGMENT_MEAN_H_
#define MINDSPORE_CORE_OPS_SPARSE_SEGMENT_MEAN_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseSegmentMean = "SparseSegmentMean";
/// \brief Computes the mean along sparse segments of a tensor.
/// Refer to Python API @ref mindspore.ops.SparseSegmentMean for more details.
class OPS_API SparseSegmentMean : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseSegmentMean);
  /// \brief Constructor.
  SparseSegmentMean() : BaseOperator(kNameSparseSegmentMean) { InitIOName({"x", "indices", "segment_ids"}, {"y"}); }
};

AbstractBasePtr SparseSegmentMeanInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_SEGMENT_MEAN_H_
