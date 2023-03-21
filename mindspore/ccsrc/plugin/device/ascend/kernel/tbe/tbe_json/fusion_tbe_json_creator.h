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
#ifndef MINDSPORE_FUSION_TBE_JSON_CREATOR_H
#define MINDSPORE_FUSION_TBE_JSON_CREATOR_H
#include <map>
#include <vector>
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_creator.h"
namespace mindspore::kernel {

using ANodeFusionDataTypeMap = std::map<const AnfNodePtr, tbe::FusionDataType>;
class FusionBuildTbeJsonCreator : public TbeJsonCreator {
 public:
  FusionBuildTbeJsonCreator() : TbeJsonCreator(), optional_index_(0) {}
  ~FusionBuildTbeJsonCreator() override = default;
  bool GenJson(const FusionScopeInfo &fusion_scope_info, nlohmann::json *fusion_json) override;
  bool GenInputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) override;
  bool GenOutputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) override;

 protected:
  bool GenOpListJson(const FusionScopeInfo &fusion_scope_info, std::vector<nlohmann::json> *fusion_json);
  std::vector<size_t> GetDescOutputIndex(const std::vector<int64_t> &output_used_nums) const;
  void GenReusedOutputDesc(const AnfNodePtr &anf_node, size_t index, size_t output_index, nlohmann::json *output_desc,
                           size_t out_size) const;
  void GenDataJson(const std::vector<AnfNodePtr> &compute_nodes, const std::vector<nlohmann::json> &compute_json,
                   std::vector<nlohmann::json> *op_list_json, const ANodeFusionDataTypeMap &spec_data_input) const;
  bool AttrsJsonPostProcessing(const AnfNodePtr &, const OpInfoPtr &, nlohmann::json *) override;
  void GenOtherJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) override;

 private:
  AnfNodePtr GetInputCNode(const AnfNodePtr &node, const nlohmann::json &input_desc) const;
  bool CheckDynamicInput(const CNodePtr &cnode) const;
  bool CheckInput(const FusionScopeInfo &fusion_scope_info) const;
  size_t optional_index_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_FUSION_TBE_JSON_CREATOR_H
