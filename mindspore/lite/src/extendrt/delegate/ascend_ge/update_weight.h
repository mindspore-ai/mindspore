/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_UPDATE_WEIGHTS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_UPDATE_WEIGHTS_H_
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "include/transform/graph_ir/utils.h"
namespace mindspore {
class UpdateWeight {
 public:
  UpdateWeight() = default;
  ~UpdateWeight() = default;

  bool IsMatchName(const std::string &cnode_name, const std::string &param_name);
  bool ParseUpdateWeightConfig(const std::string &config_path);
  std::vector<std::string> GetVariableParamsName(const FuncGraphPtr &anf_graph);
  bool SetInitDataNames(const std::vector<std::string> &init_data_names);
  bool CreateAddOpNodeForGraph(const FuncGraphPtr &anf_graph);
  bool UpdateConstantTensorData(const std::vector<std::vector<std::shared_ptr<tensor::Tensor>>> &weights,
                                std::vector<std::vector<std::shared_ptr<tensor::Tensor>>> *new_weights);

 private:
  /* note:
   * cnode_name == user_config_file_name
   * add_weight_name == cnode_name + "_add_param"
   *
   * init_data_names_ : need update weight tensor name, set by ge graph executor
   * constant_cnode_name_: user_config_file_name
   * new_weight_param_name_: add parameter node name
   */
  std::vector<std::string> new_weight_param_name_;
  std::vector<std::string> constant_cnode_name_;  // equal matmul node name, user config file name
  std::vector<std::string> init_data_names_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_UPDATE_WEIGHTS_H_
