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

namespace mindspore::lite::micro {
enum Target { kX86 = 0, kCortex_M = 1, kARM32 = 2, kARM64 = 3, kAllTargets = 4, kTargetUnknown = 99 };
enum CodeMode { Inference = 0, Train = 1, Code_Unknown = 99 };

struct MicroParam {
  std::string codegen_mode = "Inference";
  std::string target;
  bool enable_micro{false};
  bool support_parallel{false};
  bool debug_mode{false};
  std::string save_path;
  std::string project_name;
  bool is_last_model{false};
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

 private:
  Configurator() = default;
  ~Configurator() = default;
  std::string code_path_;
  Target target_{kTargetUnknown};
  CodeMode code_mode_{Code_Unknown};
  bool support_parallel_{false};
  bool debug_mode_{false};
  std::string proj_dir_;
};
}  // namespace mindspore::lite::micro
#endif  // MICRO_CODER_CONFIG_H
