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

#ifndef MINDSPORE_LITE_MICRO_CODER_CONFIG_H
#define MINDSPORE_LITE_MICRO_CODER_CONFIG_H

#include <string>

namespace mindspore::lite::micro {
enum Target { kX86 = 0, kARM32M = 1, kARM32A = 2, kARM64 = 3, kAllTargets = 4, kTargetUnknown = 99 };
enum CodeMode { Code_Normal = 0, Code_Inference = 1, Code_Train = 2, Code_Unknown = 99 };

class Configurator {
 public:
  static Configurator *GetInstance() {
    static Configurator configurator;
    return &configurator;
  }

  void set_module_name(const std::string &module_name) { module_name_ = module_name; }
  std::string module_name() const { return module_name_; }

  void set_code_path(const std::string &code_path) { code_path_ = code_path; }
  std::string code_path() const { return code_path_; }

  void set_subgraph_(const std::string &subgraph) { sub_graph_ = subgraph; }
  std::string sub_graph() { return sub_graph_; }

  void set_target(Target target) { target_ = target; }
  Target target() const { return target_; }

  void set_code_mode(CodeMode code_mode) { code_mode_ = code_mode; }
  CodeMode code_mode() const { return code_mode_; }

  void set_debug_mode(bool debug) { debug_mode_ = debug; }
  bool debug_mode() const { return debug_mode_; }

  void set_is_weight_file(bool flag) { is_weight_file_ = flag; }
  bool is_weight_file() const { return is_weight_file_; }

 private:
  Configurator() = default;
  ~Configurator() = default;

  bool is_weight_file_{false};
  std::string module_name_;
  std::string code_path_;
  std::string sub_graph_;
  Target target_{kTargetUnknown};
  CodeMode code_mode_{Code_Unknown};
  bool debug_mode_{false};
};
}  // namespace mindspore::lite::micro

#endif  // MICRO_CODER_CONFIG_H
