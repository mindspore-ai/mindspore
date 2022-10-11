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

#ifndef MINDSPORE_DYNAMIC_OBFUSCATION_H
#define MINDSPORE_DYNAMIC_OBFUSCATION_H

#include <vector>
#include <string>
#include "load_mindir/load_model.h"
#include "include/common/visible.h"

namespace mindspore {
class COMMON_EXPORT DynamicObfuscator {
 public:
  DynamicObfuscator(const float obf_ratio, const int obf_password, const int append_password)
      : obf_ratio_(obf_ratio), obf_password_(obf_password), append_password_(append_password) {}

  ~DynamicObfuscator() = default;

  FuncGraphPtr ObfuscateMindIR(const FuncGraphPtr &func_graph);

 private:
  void op_wise_fake_branch(FuncGraphPtr func_graph);
  std::string single_op_obfuscate_type(AnfNodePtr node);
  CNodePtr get_control_node(FuncGraphPtr func_graph, AnfNodePtr prev_node);
  CNodePtr password_mode_control(FuncGraphPtr func_graph);
  CNodePtr custom_op_mode_control(FuncGraphPtr func_graph, AnfNodePtr prev_node);
  void replace_matmul_node(CNodePtr node, FuncGraphPtr func_graph, CNodePtr flag_node);

  const float obf_ratio_ = 0.01;
  const int obf_password_;
  const int append_password_;
  bool has_build_appended_input = false;
  std::vector<bool> customized_func_results_;
  int used_control_node_ = 0;
  bool switch_branch_ = true;
  const std::vector<std::string> obf_target_op = {"MatMul-op", "Add-op", "Mat-op", "Sub-op", "Softmax-op", "Relu-op"};
};
}  // namespace mindspore
#endif  // MINDSPORE_DYNAMIC_OBFUSCATION_H
