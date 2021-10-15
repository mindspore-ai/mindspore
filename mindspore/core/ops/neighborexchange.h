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

#ifndef MINDSPORE_CORE_OPS_NEIGHBOREXCHANGE_H_
#define MINDSPORE_CORE_OPS_NEIGHBOREXCHANGE_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNeighborExchange = "NeighborExchange";
/// \brief NeighborExchange sends data from the local rank to ranks in the send_rank_ids.
/// Refer to Python API @ref mindspore.ops.NeighborExchange for more details.
class MS_CORE_API NeighborExchange : public PrimitiveC {
 public:
  /// \brief Constructor.
  NeighborExchange() : PrimitiveC(kNameNeighborExchange) {}
  /// \brief Destructor.
  ~NeighborExchange() = default;
  MS_DECLARE_PARENT(NeighborExchange, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.NeighborExchange for the inputs.
  void Init() {}
};
AbstractBasePtr NeighborExchangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args);
using PrimNeighborExchangePtr = std::shared_ptr<NeighborExchange>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NEIGHBOREXCHANGE_H_
