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
#include "tools/converter/micro/coder/coder.h"
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include "tools/converter/micro/coder/session.h"
#include "tools/converter/micro/coder/train/train_session.h"
#include "utils/dir_utils.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "tools/converter/micro/coder/config.h"
#include "tools/converter/micro/coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
namespace {
std::shared_ptr<CoderSession> CreateCoderSession() {
  std::shared_ptr<CoderSession> session;
  auto code_mode = Configurator::GetInstance()->code_mode();
  if (code_mode == CodeMode::Inference) {
    session = std::make_shared<CoderSession>();
  } else if (code_mode == CodeMode::Train) {
    session = std::make_shared<CoderTrainSession>();
  } else {
    MS_LOG(ERROR) << "unsupported code mode. " << code_mode;
    session = nullptr;
  }
  return session;
}
}  // namespace
int Coder::Run(const void *model_buff, size_t size) {
  session_ = CreateCoderSession();
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "new session failed while running!";
    return RET_ERROR;
  }
  STATUS status = session_->Init(model_buff, size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Init session failed!";
    return RET_ERROR;
  }

  status = session_->Build();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Compile graph failed!";
    return status;
  }
  status = session_->Run();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Generate Code Files error!" << status;
    return status;
  }
  status = session_->GenerateCode();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Generate Code Files error!" << status;
  }
  FreeGlobalVariable();
  FreeThread();
  return status;
}

bool Coder::InitPath(const std::string &output_path) {
  this->save_path_.clear();
  this->model_name_.clear();
  auto pos = output_path.find_last_of('/');
  if (pos == std::string::npos) {
    pos = output_path.find_last_of('\\');
  }
  if (pos == std::string::npos) {
#ifdef _WIN32
    this->save_path_ = ".\\";
#else
    this->save_path_ = "./";
#endif
    this->model_name_ = output_path;
  } else {
    this->save_path_ = output_path.substr(0, pos + 1);
    this->model_name_ = output_path.substr(pos + 1);
  }
  this->save_path_ = RealPath(this->save_path_.c_str());
  if (this->save_path_.empty()) {
    return false;
  }
  auto suffix_pos = this->model_name_.find_last_of('.');
  if (suffix_pos != std::string::npos && this->model_name_.substr(suffix_pos + 1) == "ms") {
    this->model_name_ = this->model_name_.substr(0, suffix_pos);
  }
#ifdef _WIN32
  this->save_path_ = this->save_path_ + "\\";
#else
  this->save_path_ = this->save_path_ + "/";
#endif
  return true;
}

int Coder::MicroSourceCodeGeneration(const schema::MetaGraphT &graph, const std::string &output_path,
                                     const std::string &codegen_mode, const std::string &device, bool support_parallel,
                                     bool debug_mode) {
  flatbuffers::FlatBufferBuilder builder(kFlatbuffersBuilderInitSize);
  auto offset = schema::MetaGraph::Pack(builder, &graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  micro::Coder code_gen;
  if (!code_gen.InitPath(output_path)) {
    MS_LOG(ERROR) << "Init path failed";
    return RET_ERROR;
  }
  // codegeneration for micro
  STATUS status = code_gen.Init(codegen_mode, device, support_parallel, debug_mode);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Codegen init Error";
    return RET_ERROR;
  }
  status = code_gen.Run(builder.GetBufferPointer(), size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Codegen Run Error";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "end of Codegen";
  return RET_OK;
}

int Coder::Init(const std::string &code_mode, const std::string &target, bool support_parallel, bool debug_mode) const {
  static const std::map<std::string, Target> kTargetMap = {
    {"x86", kX86}, {"Cortex-M", kCortex_M}, {"ARM32", kARM32}, {"ARM64", kARM64}, {"All", kAllTargets}};
  static const std::map<std::string, CodeMode> kCodeModeMap = {{"Inference", Inference}, {"Train", Train}};
  Configurator *config = Configurator::GetInstance();

  auto target_item = kTargetMap.find(target);
  MS_CHECK_TRUE_MSG(target_item != kTargetMap.end(), RET_ERROR, "unsupported target: " + target);
  config->set_target(target_item->second);

  auto code_item = kCodeModeMap.find(code_mode);
  MS_CHECK_TRUE_MSG(code_item != kCodeModeMap.end(), RET_ERROR, "unsupported code mode: " + code_mode);
  config->set_code_mode(code_item->second);
  if (code_item->second == CodeMode::Train && config->target() == kCortex_M) {
    MS_LOG(ERROR) << "Cortex-M cannot support train.";
    return RET_ERROR;
  }

  if (support_parallel && config->target() == kCortex_M) {
    MS_LOG(ERROR) << "Cortex-M cannot support parallel.";
    return RET_ERROR;
  }
  config->set_support_parallel(support_parallel);
  config->set_debug_mode(debug_mode);

  config->set_proj_dir(model_name_);

  const std::string slash = std::string(kSlash);
  if (!save_path_.empty() && !DirExists(save_path_)) {
    MS_LOG(ERROR) << "code_gen code path " << save_path_ << " is not valid";
    return RET_ERROR;
  }

  if (save_path_.substr(save_path_.size() - 1, 1) != slash) {
    std::string path = save_path_ + slash + model_name_;
    config->set_code_path(path);
  } else {
    std::string path = save_path_ + model_name_;
    config->set_code_path(path);
  }

  if (InitProjDirs(save_path_, model_name_) != RET_OK) {
    return RET_ERROR;
  }

  auto print_parameter = [](auto name, auto value) {
    MS_LOG(INFO) << std::setw(20) << std::left << name << "= " << value;
  };

  print_parameter("projectName", config->proj_dir());
  print_parameter("target", config->target());
  print_parameter("codePath", config->code_path());
  print_parameter("codeMode", config->code_mode());
  print_parameter("debugMode", config->debug_mode());
  return RET_OK;
}
}  // namespace mindspore::lite::micro
