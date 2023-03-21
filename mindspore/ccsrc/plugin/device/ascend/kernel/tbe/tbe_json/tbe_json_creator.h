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
#include "kernel/kernel.h"
#include "kernel/kernel_fusion.h"
#include "kernel/oplib/oplib.h"
#include "plugin/device/ascend/kernel/tbe/tbe_adapter.h"

namespace mindspore::kernel {
enum ATTR_DTYPE {
  ATTR_INT8 = 0,
  ATTR_UINT8 = 1,
  ATTR_INT16 = 2,
  ATTR_UINT16 = 3,
  ATTR_INT32 = 4,
  ATTR_UINT32 = 5,
  ATTR_INT64 = 6,
  ATTR_UINT64 = 7,
  ATTR_FLOAT32 = 8,
  ATTR_DOUBLE = 9,
  ATTR_BOOL = 10,
  ATTR_STR = 11,
  ATTR_LIST_INT8 = 12,
  ATTR_LIST_UINT8 = 13,
  ATTR_LIST_INT16 = 14,
  ATTR_LIST_UINT16 = 15,
  ATTR_LIST_INT32 = 16,
  ATTR_LIST_UINT32 = 17,
  ATTR_LIST_INT64 = 18,
  ATTR_LIST_UINT64 = 19,
  ATTR_LIST_FLOAT32 = 20,
  ATTR_LIST_DOUBLE = 21,
  ATTR_LIST_BOOL = 22,
  ATTR_LIST_STR = 23,
  ATTR_LIST_LIST_INT64 = 24,
  ATTR_LIST_LIST_FLOAT = 25,

  // illegal type which can't be fused
  ATTR_MAX,
};

class TbeJsonCreator {
 public:
  TbeJsonCreator() = default;
  virtual ~TbeJsonCreator() = default;
  virtual bool GenJson(const AnfNodePtr &anf_node, nlohmann::json *kernel_json) { return false; }
  virtual bool GenJson(const FusionScopeInfo &fusion_scope_info, nlohmann::json *fusion_json) { return false; }
  std::string GetJsonName() { return json_name_; }
  size_t GetJsonHash() const { return json_hash_; }
  virtual bool GenInputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) { return false; }
  virtual bool GenOutputsJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) { return false; }

 protected:
  static std::string GetCoreType(const AnfNodePtr &node);
  bool GenComputeJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json);
  void GenOutputDataDescJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) const;
  void GenComputeCommonJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) const;
  virtual void GenOtherJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json) {}
  void GenAttrsDescJson(const AnfNodePtr &anf_node, nlohmann::json *compute_json);
  void GenAttrsJson(const AnfNodePtr &anf_node, const OpInfoPtr &op_info, nlohmann::json *attrs_json);
  void AttrsJsonPreProcessing(const AnfNodePtr &anf_node, std::vector<OpAttrPtr> *attrs_ptr,
                              nlohmann::json *attrs_json) const;
  virtual bool AttrsJsonPostProcessing(const AnfNodePtr &anf_node, const OpInfoPtr &op_info_ptr,
                                       nlohmann::json *attrs_json);
  virtual void GenDescJson(const AnfNodePtr &anf_node, size_t node_out_idx, size_t desc_output_idx,
                           nlohmann::json *output_desc);
  void GenDesJsonCommon(nlohmann::json *output_desc) const;
  void GenInputConstValue(const AnfNodePtr &anf_node, size_t real_input_index, nlohmann::json *input_desc) const;
  size_t GenJsonHash(nlohmann::json tbe_json) const;
  void DeleteDescName(nlohmann::json *desc_jsons) const;
  void AddOpNameForComputeNode(nlohmann::json *kernel_json) const;
  void GenFusionOpName(nlohmann::json *kernel_json, std::string prefix = "");

 private:
  std::string json_name_;
  size_t json_hash_{0};
};

}  // namespace mindspore::kernel
#endif  // MINDSPORE_TBE_JSON_CREATOR_H_
