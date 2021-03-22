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
#include "coder/coder.h"
#include <getopt.h>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include "schema/inner/model_generated.h"
#include "tools/common/flag_parser.h"
#include "coder/session.h"
#include "coder/context.h"
#include "utils/dir_utils.h"
#include "securec/include/securec.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "coder/config.h"
#include "coder/generator/component/component.h"

namespace mindspore::lite::micro {

class CoderFlags : public virtual FlagParser {
 public:
  CoderFlags() {
    AddFlag(&CoderFlags::model_path_, "modelPath", "Input model path", "");
    AddFlag(&CoderFlags::code_path_, "codePath", "Input code path", ".");
    AddFlag(&CoderFlags::target_, "target", "generated code target, x86| ARM32M| ARM32A| ARM64", "x86");
    AddFlag(&CoderFlags::code_mode_, "codeMode", "generated code mode, Inference | Train", "Inference");
    AddFlag(&CoderFlags::support_parallel_, "supportParallel", "whether support parallel launch, true | false", false);
    AddFlag(&CoderFlags::debug_mode_, "debugMode", "dump the tensors data for debugging, true | false", false);
  }

  ~CoderFlags() override = default;

  std::string model_path_;
  bool support_parallel_{false};
  std::string code_path_;
  std::string code_mode_;
  bool debug_mode_{false};
  std::string target_;
};

int Coder::Run(const std::string &model_path) {
  session_ = CreateCoderSession();
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "new session failed while running";
    return RET_ERROR;
  }
  STATUS status = session_->Init(model_path);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Init session failed.";
    return RET_ERROR;
  }

  status = session_->Build();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Set Input resize shapes error";
    return status;
  }
  status = session_->Run();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Generate Code Files error. " << status;
    return status;
  }
  status = session_->GenerateCode();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Generate Code Files error " << status;
  }
  return status;
}

int Configurator::ParseProjDir(std::string model_path) {
  // split model_path to get model file name
  proj_dir_ = model_path;
  size_t found = proj_dir_.find_last_of("/\\");
  if (found != std::string::npos) {
    proj_dir_ = proj_dir_.substr(found + 1);
  }
  found = proj_dir_.find(".ms");
  if (found != std::string::npos) {
    proj_dir_ = proj_dir_.substr(0, found);
  } else {
    MS_LOG(ERROR) << "model file's name must be end with \".ms\".";
    return RET_ERROR;
  }
  if (proj_dir_.size() == 0) {
    proj_dir_ = "net";
    MS_LOG(WARNING) << "parse model's name failed, use \"net\" instead.";
  }
  return RET_OK;
}

int Coder::Init(const CoderFlags &flags) const {
  static const std::map<std::string, Target> kTargetMap = {
    {"x86", kX86}, {"ARM32M", kARM32M}, {"ARM32A", kARM32A}, {"ARM64", kARM64}, {"All", kAllTargets}};
  static const std::map<std::string, CodeMode> kCodeModeMap = {{"Inference", Inference}, {"Train", Train}};
  Configurator *config = Configurator::GetInstance();

  std::vector<std::function<bool()>> parsers;
  parsers.emplace_back([&flags, config]() -> bool {
    if (!FileExists(flags.model_path_)) {
      MS_LOG(ERROR) << "model_path \"" << flags.model_path_ << "\" is not valid";
      return false;
    }
    if (config->ParseProjDir(flags.model_path_) != RET_OK) {
      return false;
    }
    return true;
  });

  parsers.emplace_back([&flags, config]() -> bool {
    auto target_item = kTargetMap.find(flags.target_);
    MS_CHECK_TRUE_RET_BOOL(target_item != kTargetMap.end(), "unsupported target: " + flags.target_);
    config->set_target(target_item->second);
    return true;
  });

  parsers.emplace_back([&flags, config]() -> bool {
    auto code_item = kCodeModeMap.find(flags.code_mode_);
    MS_CHECK_TRUE_RET_BOOL(code_item != kCodeModeMap.end(), "unsupported code mode: " + flags.code_mode_);
    config->set_code_mode(code_item->second);
    return true;
  });

  parsers.emplace_back([&flags, config]() -> bool {
    if (flags.support_parallel_ && config->target() == kARM32M) {
      MS_LOG(ERROR) << "arm32M cannot support parallel.";
      return false;
    }
    config->set_support_parallel(flags.support_parallel_);
    return true;
  });

  parsers.emplace_back([&flags, config]() -> bool {
    config->set_debug_mode(flags.debug_mode_);
    return true;
  });

  parsers.emplace_back([&flags, config]() -> bool {
    const std::string slash = std::string(kSlash);
    if (!flags.code_path_.empty() && !DirExists(flags.code_path_)) {
      MS_LOG(ERROR) << "code_gen code path " << flags.code_path_ << " is not valid";
      return false;
    }
    config->set_code_path(flags.code_path_);
    if (flags.code_path_.empty()) {
      std::string path = ".." + slash + config->proj_dir();
      config->set_code_path(path);
    } else {
      if (flags.code_path_.substr(flags.code_path_.size() - 1, 1) != slash) {
        std::string path = flags.code_path_ + slash + config->proj_dir();
        config->set_code_path(path);
      } else {
        std::string path = flags.code_path_ + config->proj_dir();
        config->set_code_path(path);
      }
    }
    return InitProjDirs(flags.code_path_, config->proj_dir()) != RET_ERROR;
  });

  if (!std::all_of(parsers.begin(), parsers.end(), [](auto &parser) -> bool { return parser(); })) {
    if (!flags.help) {
      std::cerr << flags.Usage() << std::endl;
      return 0;
    }
    return RET_ERROR;
  }
  auto print_parameter = [](auto name, auto value) {
    MS_LOG(INFO) << std::setw(20) << std::left << name << "= " << value;
  };

  print_parameter("modelPath", flags.model_path_);
  print_parameter("projectName", config->proj_dir());
  print_parameter("target", config->target());
  print_parameter("codePath", config->code_path());
  print_parameter("codeMode", config->code_mode());
  print_parameter("debugMode", config->debug_mode());

  return RET_OK;
}

int RunCoder(int argc, const char **argv) {
  CoderFlags flags;
  Option<std::string> err = flags.ParseFlags(argc, argv, false, false);
  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << flags.Usage() << std::endl;
    return RET_ERROR;
  }

  if (flags.help) {
    std::cerr << flags.Usage() << std::endl;
    return RET_OK;
  }

  Coder code_gen;
  STATUS status = code_gen.Init(flags);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Coder init Error";
    return status;
  }
  status = code_gen.Run(flags.model_path_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Coder Run Error.";
    return status;
  }
  MS_LOG(INFO) << "end of Coder";
  return RET_OK;
}

}  // namespace mindspore::lite::micro
