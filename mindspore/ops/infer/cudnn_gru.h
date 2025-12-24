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

#ifndef MINDSPORE_CORE_OPS_CUDNN_GRU_H_
#define MINDSPORE_CORE_OPS_CUDNN_GRU_H_

#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCudnnGRU = "CudnnGRU";
class OPS_API CudnnGRU : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(CudnnGRU);

  /// \brief Constructor.
  CudnnGRU() : BaseOperator(kNameCudnnGRU) { InitIOName({"input", "h", "w"}, {"output", "h_n", "reserve", "state"}); }
};

using PrimCudnnGRUPtr = std::shared_ptr<CudnnGRU>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CUDNN_GRU_H_
