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

#ifndef MINDSPORE_CORE_OPS_SPLIT_WITH_SIZE_H_
#define MINDSPORE_CORE_OPS_SPLIT_WITH_SIZE_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
/// \brief Splits the input tensor into output_num of tensors along the given axis and output numbers.
/// Refer to Python API @ref mindspore.ops.Split for more details.
class MIND_API SplitWithSize : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SplitWithSize);
  /// \brief Constructor.
  SplitWithSize() : BaseOperator(kNameSplitWithSize) { InitIOName({"input", "split_size", "axis"}, {"output"}); }
};
MIND_API abstract::AbstractBasePtr SplitWithSizeInfer(const abstract::AnalysisEnginePtr &,
                                                      const PrimitivePtr &primitive,
                                                      const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimSplitWithSize = std::shared_ptr<SplitWithSize>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SPLIT_WITH_SIZE_H_
