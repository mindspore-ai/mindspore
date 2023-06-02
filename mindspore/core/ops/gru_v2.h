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

#ifndef MINDSPORE_CORE_OPS_GRU_V2_H_
#define MINDSPORE_CORE_OPS_GRU_V2_H_

#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "mindspore/core/ops/op_name.h"
#include "ops/base_operator.h"
#include "ops/primitive_c.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGRUV2 = "GRUV2";
/// \brief rnn net.
/// Refer to Python API @ref mindspore.ops.GRUV2 for more details.
class MIND_API GRUV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GRUV2);
  /// \brief Constructor.
  GRUV2() : BaseOperator(kNameGRUV2) {
    InitIOName({"input", "h", "w", "seq_lengths"}, {"output", "h_n", "reserve", "state"});
  }

  void set_bidirectional(bool bidirectional) { (void)AddAttr(kBidirectional, api::MakeValue(bidirectional)); }

  bool get_bidirectional() const {
    auto value_ptr = this->GetAttr(kBidirectional);
    return GetValue<bool>(value_ptr);
  }

  int64_t get_input_size() const { return GetValue<int64_t>(GetAttr(kInput_size)); }
  int64_t get_hidden_size() const { return GetValue<int64_t>(GetAttr(kHidden_size)); }
  int64_t get_num_layers() const { return GetValue<int64_t>(GetAttr(kNumLayers)); }
  bool get_has_bias() const {
    auto value_ptr = this->GetAttr(kHasBias);
    return GetValue<bool>(value_ptr);
  }
};
AbstractBasePtr GRUV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);
using PrimGRUV2Ptr = std::shared_ptr<GRUV2>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRU_V2_H_
