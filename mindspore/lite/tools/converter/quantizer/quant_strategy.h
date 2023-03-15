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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H_

#include <cstddef>
#include <utility>
#include <set>
#include <string>
#include "ir/anf.h"
#include "mindspore/core/ops/core_ops.h"
#include "utils/check_convert_utils.h"
#include "ir/manager.h"
#include "tools/converter/quantizer/quant_params.h"

namespace mindspore::lite::quant {
class QuantStrategy {
 public:
  QuantStrategy(size_t min_quant_weight_size, size_t min_quant_weight_channel, std::set<std::string> skip_node,
                TargetDevice target_device = CPU)
      : min_quant_weight_size_(min_quant_weight_size),
        min_quant_weight_channel_(min_quant_weight_channel),
        skip_node_(std::move(skip_node)),
        target_device_(target_device) {}

  ~QuantStrategy() = default;

  bool CanOpFullQuantized(const FuncGraphManagerPtr &manager, const CNodePtr &cnode,
                          const std::set<PrimitivePtr> &support_int8_ops,
                          const std::set<PrimitivePtr> &skip_check_dtype_ops,
                          const std::set<mindspore::ActivationType> &support_activation);

  bool CanTensorQuantized(const CNodePtr &cnode, const AnfNodePtr &input_node, int preferred_dim);

  bool IsSkipOp(const std::string &skip_node_name);

 private:
  bool CheckAscendSpec(const FuncGraphManagerPtr &manager, const CNodePtr &cnode, TypeId type_id,
                       int min_quant_weight_channel);

  size_t min_quant_weight_size_ = 0;
  size_t min_quant_weight_channel_ = 0;
  std::set<std::string> skip_node_;
  TargetDevice target_device_ = CPU;
};
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H_
