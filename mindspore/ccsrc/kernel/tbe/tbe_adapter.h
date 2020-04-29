/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_KERNEL_TBE_TBE_ADAPTER_H
#define MINDSPORE_CCSRC_KERNEL_TBE_TBE_ADAPTER_H

#include <string>
#include <map>
#include <memory>
#include <vector>
#include "nlohmann/json.hpp"
#include "ir/base.h"
#include "kernel/oplib/opinfo.h"
// Note: This file is mainly used to adapt the ME front-end operator description and
//       the TBE back-end operator implementation difference
namespace mindspore {
namespace kernel {
enum kCreaterType : int { SINGLE_BUILD = 0, PREBUILD, OP_SELECT_FORMAT, CHECK_SUPPORTED };
namespace tbe {
using FAttrsPass = void (*)(const AnfNodePtr &anf_node, const std::vector<std::shared_ptr<OpAttr>> &op_info_attrs,
                            nlohmann::json *attrs_json);
class TbeAdapter {
 public:
  TbeAdapter() = default;
  ~TbeAdapter() = default;
  static void NormalizeFuncName(std::string *func_name);
  static void SetTbeAttrsForTransDataOp(const AnfNodePtr &anf_node);
  static void InputOrderPass(const std::string &op_name, std::vector<std::vector<nlohmann::json>> const &inputs_list,
                             nlohmann::json *inputs_json);
  static bool RunAttrPass(const AnfNodePtr &anf_node, const std::vector<std::shared_ptr<OpAttr>> &op_info_attrs,
                          nlohmann::json *attrs_json);
  static void GenTopKV2IndicesTensorInfo(const std::shared_ptr<AnfNode> &anf_node, size_t real_input_index,
                                         std::vector<nlohmann::json> *input_list, kCreaterType creater_type);

  static void FusionInputOrderPass(const std::string &op_name, const std::vector<nlohmann::json> &inputs_list,
                                   std::vector<nlohmann::json> *inputs_json);
  static void FusionDataOrderPass(const std::string &op_name, const std::vector<AnfNodePtr> &data_layer,
                                  std::vector<AnfNodePtr> *reorder_data_layer);

 private:
  static void MaximumGradAttrJsonPass(const AnfNodePtr &anf_node,
                                      const std::vector<std::shared_ptr<OpAttr>> &op_info_attrs,
                                      nlohmann::json *attrs_json);
  static void MinimumGradAttrJsonPass(const AnfNodePtr &anf_node,
                                      const std::vector<std::shared_ptr<OpAttr>> &op_info_attrs,
                                      nlohmann::json *attrs_json);

  static void CastAttrJsonPass(const AnfNodePtr &anf_node, const std::vector<std::shared_ptr<OpAttr>> &op_info_attrs,
                               nlohmann::json *attrs_json);

  static std::map<std::string, FAttrsPass> build_json_attr_pass_map_;
};
}  // namespace tbe
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_TBE_TBE_ADAPTER_H
