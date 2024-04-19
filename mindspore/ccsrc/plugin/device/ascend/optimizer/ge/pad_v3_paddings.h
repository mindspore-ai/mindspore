/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_PAD_V3_PADDINGS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_PAD_V3_PADDINGS_H_

#include <vector>
#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class PadV3Paddings : public PatternProcessPass {
 public:
  explicit PadV3Paddings(bool multi_graph = true) : PatternProcessPass("pad_v3_paddings", multi_graph) {}
  ~PadV3Paddings() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

  bool HasDynPaddings(const CNodePtr &) const;

  const CNodePtr CreateStridedSlice(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, int64_t index) const;
  const CNodePtr CreateConcatNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &concat_input_vec,
                                  std::string concat_node_name) const;
  const CNodePtr ProcessSliceNConcat(const FuncGraphPtr &func_graph, const AnfNodePtr &pad_node,
                                     const AnfNodePtr &input_node, const int64_t &padding_dst_length,
                                     const int64_t &padding_src_length) const;

  template <typename T, TypeId type_id>
  const AnfNodePtr OptimizePaddingsValue(const FuncGraphPtr &, const AbstractBasePtr &, const bool &, const size_t &,
                                         bool force_length8) const;

  const AnfNodePtr CreatePaddingsNode(const FuncGraphPtr &graph, const AbstractBasePtr &ori_paddings,
                                      const bool &paddings_contiguous, const size_t &dst_length,
                                      const TypeId &type_id) const {
    if (type_id == kNumberTypeInt32) {
      return OptimizePaddingsValue<int32_t, kNumberTypeInt32>(graph, ori_paddings, paddings_contiguous, dst_length,
                                                              false);
    }
    return OptimizePaddingsValue<int64_t, kNumberTypeInt64>(graph, ori_paddings, paddings_contiguous, dst_length,
                                                            false);
  }
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_PAD_V3_PADDINGS_H_
