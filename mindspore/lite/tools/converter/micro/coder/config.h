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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_CONFIG_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_CONFIG_H_

#include <string>
#include <map>
#include <vector>

namespace mindspore::lite::micro {
enum Target { kX86 = 0, kCortex_M = 1, kARM32 = 2, kARM64 = 3, kAllTargets = 4, kTargetUnknown = 99 };
enum CodeMode { Inference = 0, Train = 1, Code_Unknown = 99 };

struct MicroParam {
  std::string codegen_mode = "Inference";
  std::string target;
  std::string changeable_weights_name;
  bool enable_micro{false};
  bool support_parallel{false};
  bool debug_mode{false};
  std::string save_path;
  std::string project_name;
  bool is_last_model{false};
  bool keep_original_weight{false};
  std::vector<std::vector<std::string>> graph_inputs_template;
  std::map<std::string, std::vector<std::string>> graph_inputs_origin_info;
  std::vector<std::string> dynamic_symbols;
  std::vector<size_t> dynamic_symbols_num;
  std::map<std::string, std::vector<int>> dynamic_symbols_map;
  std::vector<std::vector<std::vector<int>>> graph_inputs_shape_infos;
  std::map<std::string, std::vector<std::vector<int>>> inputs_shape_by_scenes;
};

class Configurator {
 public:
  static Configurator *GetInstance() {
    static Configurator configurator;
    return &configurator;
  }

  void set_code_path(const std::string &code_path) { code_path_ = code_path; }
  std::string code_path() const { return code_path_; }

  void set_target(Target target) { target_ = target; }
  Target target() const { return target_; }

  void set_code_mode(CodeMode code_mode) { code_mode_ = code_mode; }
  CodeMode code_mode() const { return code_mode_; }

  void set_debug_mode(bool debug) { debug_mode_ = debug; }
  bool debug_mode() const { return debug_mode_; }

  void set_support_parallel(bool parallel) { support_parallel_ = parallel; }
  bool support_parallel() const { return support_parallel_; }

  void set_proj_dir(std::string dir) { proj_dir_ = dir; }
  std::string proj_dir() const { return proj_dir_; }

  void set_keep_original_weight(bool keep_weight) { keep_original_weight_ = keep_weight; }
  bool keep_original_weight() const { return keep_original_weight_; }

  void set_changeable_weights_name(const std::string &weights_name) { changeable_weights_name_ = weights_name; }
  const std::string &changeable_weights_name() const { return changeable_weights_name_; }

  void set_dynamic_shape(bool dynamic_shape) { dynamic_shape_ = dynamic_shape; }
  bool dynamic_shape() const { return dynamic_shape_; }

  void set_dynamic_symbols(const std::vector<std::string> &dynamic_symbols) { dynamic_symbols_ = dynamic_symbols; }
  const std::vector<std::string> &dynamic_symbols() const { return dynamic_symbols_; }

  void set_dynamic_symbols_num(const std::vector<size_t> &dynamic_symbols_num) {
    dynamic_symbols_num_ = dynamic_symbols_num;
  }
  const std::vector<size_t> &dynamic_symbols_num() const { return dynamic_symbols_num_; }

  void set_dynamic_symbols_map(const std::map<std::string, std::vector<int>> &dynamic_symbols_map) {
    dynamic_symbols_map_ = dynamic_symbols_map;
  }
  const std::map<std::string, std::vector<int>> &dynamic_symbols_map() const { return dynamic_symbols_map_; }

  void set_user_graph_inputs_template(const std::vector<std::vector<std::string>> &graph_inputs_template) {
    user_graph_inputs_template_ = graph_inputs_template;
  }
  const std::vector<std::vector<std::string>> &user_graph_inputs_template() const {
    return user_graph_inputs_template_;
  }

  void set_graph_inputs_shape_infos(const std::vector<std::vector<std::vector<int>>> &graph_inputs_shape_infos) {
    graph_inputs_shape_infos_ = graph_inputs_shape_infos;
  }
  const std::vector<std::vector<std::vector<int>>> &graph_inputs_shape_infos() { return graph_inputs_shape_infos_; }

 private:
  Configurator() = default;
  ~Configurator() = default;
  std::string code_path_;
  Target target_{kTargetUnknown};
  CodeMode code_mode_{Code_Unknown};
  bool support_parallel_{false};
  bool debug_mode_{false};
  bool keep_original_weight_{false};
  bool dynamic_shape_{false};
  std::string proj_dir_;
  std::string changeable_weights_name_;
  std::vector<std::string> dynamic_symbols_;
  std::vector<size_t> dynamic_symbols_num_;
  std::map<std::string, std::vector<int>> dynamic_symbols_map_;
  std::vector<std::vector<std::vector<int>>> graph_inputs_shape_infos_;
  std::vector<std::vector<std::string>> user_graph_inputs_template_;
};
}  // namespace mindspore::lite::micro
#endif  // MICRO_CODER_CONFIG_H
