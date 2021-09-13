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

#ifndef MINDSPORE_TBE_JSON_CREATOR_H_
#define MINDSPORE_TBE_JSON_CREATOR_H_
#include <string>
#include <unordered_map>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "ir/dtype.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/kernel_fusion.h"
#include "backend/kernel_compiler/oplib/oplib.h"
#include "backend/kernel_compiler/tbe/tbe_adapter.h"

namespace mindspore::kernel {
enum class TypeID {
  kIntID = 0,
  kInt64ID,
  kStrID,
  kBoolID,
  kFloatID,
  kListIntID,
  kListFloatID,
  kListUInt64ID,
  kListListIntID
};
class TbeJsonCreator {
 public:
  TbeJsonCreator() = default;
  virtual ~TbeJsonCreator() = default;
  virtual bool GenJson(const AnfNodePtr &anf_node, nlohmann::json *kernel_json) { return false; }
  virtual bool GenJson(const FusionScopeInfo &fusion_scope_info, nlohmann::json *fusion_json) { return false; }
  std::string GetJsonName() { return json_name_; }
  size_t GetJsonHash() { return json_hash_; }

 protected:
  bool GenComputeJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json);
  virtual bool GenInputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) { return false; }
  virtual bool GenOutputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) { return false; }
  bool GenOutputDataDescJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json);
  void GenComputeCommonJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json);
  virtual void GenOtherJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {}
  bool GenAttrsDescJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json);
  bool GenAttrsJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info_ptr, nlohmann::json *attrs_json);
  bool AttrsJsonPreProcessing(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *attrs_ptr,
                              nlohmann::json *attrs_json);
  virtual bool AttrsJsonPostProcessing(const AnfNodePtr &anf_node, const OpInfoPtr &op_info_ptr,
                                       nlohmann::json *attrs_json);
  virtual void GenDescJson(const AnfNodePtr &anf_node, size_t node_out_idx, size_t desc_output_idx,
                           nlohmann::json *output_desc);
  void GenDesJsonCommon(nlohmann::json *output_desc);
  void GenInputConstValue(const AnfNodePtr &anf_node, size_t real_input_index, nlohmann::json *input_desc);
  size_t GenJsonHash(nlohmann::json tbe_json);
  void DeleteDescName(nlohmann::json *desc_json);
  void AddOpNameForComputeNode(nlohmann::json *kernel_json);
  void GenFusionOpName(nlohmann::json *kernel_json, std::string prefix = "");

 private:
  std::string json_name_;
  size_t json_hash_;
};

}  // namespace mindspore::kernel
#endif  // MINDSPORE_TBE_JSON_CREATOR_H_
