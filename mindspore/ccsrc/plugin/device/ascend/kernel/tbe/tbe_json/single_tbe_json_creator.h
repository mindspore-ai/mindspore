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

#ifndef MINDSPORE_SINGLE_TBE_JSON_CREATOR_H
#define MINDSPORE_SINGLE_TBE_JSON_CREATOR_H
#include <vector>
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_creator.h"
namespace mindspore::kernel {

class SingleTbeJsonCreator : public TbeJsonCreator {
 public:
  SingleTbeJsonCreator() = default;
  ~SingleTbeJsonCreator() override = default;
  bool GenJson(const AnfNodePtr &anf_node, nlohmann::json *kernel_json) override;
  bool GenInputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) override;

 protected:
  bool GenOpListJson(const AnfNodePtr &anf_node, std::vector<nlohmann::json> *op_list_json);
  void OpListPostProcessing(const AnfNodePtr &anf_node, std::vector<nlohmann::json> *op_list_json);
  void GenDataJson(const AnfNodePtr &anf_node, const nlohmann::json &compute_json,
                   std::vector<nlohmann::json> *op_list_json) const;
  virtual void GenInputDescJson(const AnfNodePtr &anf_node, size_t real_input_index, nlohmann::json *input_desc);
  bool AssignInputsJson(const AnfNodePtr &anf_node, const std::vector<nlohmann::json> &inputs_desc,
                        const std::vector<size_t> &inputs_tensor_num, const std::vector<OpIOInfoPtr> &inputs_ptr,
                        std::vector<nlohmann::json> *inputs_json) const;
  void GenOutputDescJson(const AnfNodePtr &anf_node, size_t node_out_idx, nlohmann::json *output_desc);
  bool GenOutputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) override;
  bool AssignOutputsJson(const AnfNodePtr &anf_node, const std::vector<nlohmann::json> &outputs_desc,
                         const std::vector<size_t> &outputs_tensor_num, const std::vector<OpIOInfoPtr> &outputs_ptr,
                         std::vector<nlohmann::json> *outputs_json) const;
  void GenOtherJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) override;
};
class SelectTbeJsonCreator : public SingleTbeJsonCreator {
 public:
  SelectTbeJsonCreator() = default;
  ~SelectTbeJsonCreator() override = default;

 protected:
  bool AttrsJsonPostProcessing(const AnfNodePtr &anf_node, const OpInfoPtr &op_info_ptr,
                               nlohmann::json *attrs_json) override;
  void GenDescJson(const AnfNodePtr &anf_node, size_t node_out_idx, size_t desc_output_idx,
                   nlohmann::json *output_desc) override;
  void GenInputDescJson(const AnfNodePtr &anf_node, size_t real_input_index, nlohmann::json *input_desc) override;
};
class BuildTbeJsonCreator : public SingleTbeJsonCreator {
 public:
  BuildTbeJsonCreator() = default;
  ~BuildTbeJsonCreator() override = default;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_SINGLE_TBE_JSON_CREATOR_H
