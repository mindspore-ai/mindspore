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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_ADAPTER_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_ADAPTER_H

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include "nlohmann/json.hpp"
#include "base/base.h"
#include "kernel/oplib/opinfo.h"
#include "kernel/oplib/super_bar.h"
#include "kernel/kernel_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
// Note: This file is mainly used to adapt the ME front-end operator description and
//       the TBE back-end operator implementation difference
namespace mindspore::kernel::tbe {
enum FusionDataType { kFusionNormal = 0, kFusionAddN, kFusionReLUGradV2, kFusionAdd };

class TbeAdapter {
 public:
  TbeAdapter() = default;
  ~TbeAdapter() = default;
  template <typename T>
  static void InputOrderPass(const std::shared_ptr<AnfNode> &anf_node, std::vector<T> const &inputs_list,
                             std::vector<T> *inputs_json) {
    MS_EXCEPTION_IF_NULL(inputs_json);
    auto op_name = common::AnfAlgo::GetCNodeName(anf_node);
    auto orders = kernel::SuperBar::GetKernelIdxToGraphIdx(op_name);
    if (!orders.has_value()) {
      (void)std::copy(inputs_list.begin(), inputs_list.end(), std::back_inserter((*inputs_json)));
    } else {
      auto kernel_idx_to_graph_idx = orders.value();
      for (size_t i = 0; i < kernel_idx_to_graph_idx.size(); ++i) {
        (void)inputs_json->push_back(inputs_list.at(kernel_idx_to_graph_idx[i]));
      }
    }
  }
  static void FusionDescJsonPass(const AnfNodePtr &node, nlohmann::json *output_desc,
                                 const std::map<const AnfNodePtr, tbe::FusionDataType> &spec_data_input);
  static std::string GetRealOpType(const std::string &origin_type);
  static std::string FormatPass(const std::string &format, const size_t &origin_shape_size);
  static bool GetSpecDataInput(const FusionScopeInfo &fusion_scope_info,
                               std::map<const AnfNodePtr, tbe::FusionDataType> *spec_data_input);
  static bool IsPlaceHolderInput(const AnfNodePtr &node, const OpIOInfoPtr &input_ptr);
  static void CastAttrJsonPrePass(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *op_info_attrs,
                                  nlohmann::json *attrs_json);
  static void CastAttrJsonPost(const AnfNodePtr &anf_node, nlohmann::json *attrs_json);
  static void LayerNormAttrJsonPost(const AnfNodePtr &anf_node, nlohmann::json *attrs_json);

 private:
  static bool IsSpecialFusionComputeNode(const std::vector<mindspore::AnfNodePtr> &compute_nodes);
  static bool GetSpecInputLayers(const std::string &op_name, const std::vector<mindspore::AnfNodePtr> &reorder_layer,
                                 std::map<const AnfNodePtr, FusionDataType> *spec_data_input);

  static std::unordered_set<std::string> input_order_adjusted_ops_;
};
}  // namespace mindspore::kernel::tbe
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_ADAPTER_H
