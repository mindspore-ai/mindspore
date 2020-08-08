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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_IR_FUSION_INPUT_TO_OUTPUT_REGISTRY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_IR_FUSION_INPUT_TO_OUTPUT_REGISTRY_H_
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "ir/anf.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt {
using PreCheckFunc = std::function<bool(const CNodePtr &node)>;
class InputToOutputRegister {
 public:
  explicit InputToOutputRegister(
    const std::string &op_name = "", const PreCheckFunc &pre_check_func = [](const CNodePtr &node) { return true; })
      : op_name_(op_name), pre_check_func_(pre_check_func) {}
  virtual ~InputToOutputRegister() = default;

  void set_input_indices(const std::vector<size_t> &input_indices) { input_indices_ = input_indices; }

  const std::vector<size_t> &input_indices() const { return input_indices_; }
  const std::string &op_name() const { return op_name_; }

 private:
  std::string op_name_;
  std::vector<size_t> input_indices_;
  PreCheckFunc pre_check_func_;
};

class InputToOutputRegistry {
 public:
  static InputToOutputRegistry &Instance();
  void Register(const InputToOutputRegister &reg);
  void Register(
    const std::string &op_name, const std::vector<size_t> &input_indices,
    const PreCheckFunc &pre_check_func = [](const CNodePtr &node) { return true; });
  bool GetRegisterByOpName(const std::string &op_name, InputToOutputRegister *reg) const;

 private:
  InputToOutputRegistry();
  ~InputToOutputRegistry() = default;
  DISABLE_COPY_AND_ASSIGN(InputToOutputRegistry)
  std::unordered_map<std::string, InputToOutputRegister> op_input_to_output_map_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_IR_FUSION_INPUT_TO_OUTPUT_REGISTRY_H_
