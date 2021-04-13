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
#include "coder/generator/generator.h"
#include <sys/stat.h>
#include <set>
#include <fstream>
#include "coder/generator/component/component.h"
#include "coder/generator/component/cmake_component.h"
#include "coder/generator/component/weight_component.h"
#include "coder/generator/component/common_component.h"
#include "coder/generator/component/const_blocks/cmake_lists.h"
#include "coder/generator/component/const_blocks/debug_utils.h"
#include "coder/generator/component/const_blocks/load_input.h"
#include "coder/generator/component/const_blocks/msession.h"
#include "coder/generator/component/const_blocks/mtensor.h"
#include "coder/generator/component/const_blocks/mstring.h"
#include "coder/generator/component/const_blocks/model.h"
#include "coder/generator/component/const_blocks/thread_pool.h"
#include "coder/generator/component/const_blocks/benchmark.h"
#include "coder/generator/component/const_blocks/license.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
int WriteContentToFile(const std::string &file, const std::string &content) {
  std::ofstream of(file);
  if (of.bad()) {
    MS_LOG(ERROR) << "open file error " << file;
    return RET_ERROR;
  }
  MS_LOG(INFO) << "write " << file;
  of << content;
  of.close();
  return RET_OK;
}

Generator::Generator(std::unique_ptr<CoderContext> ctx) {
  ctx_ = std::move(ctx);
  this->config_ = Configurator::GetInstance();
  this->net_inc_hfile_ = "net.h";
  this->net_src_cfile_ = "net.c";
  this->net_weight_hfile_ = "weight.h";
  this->net_src_file_path_ = config_->code_path() + kSourcePath;
  this->net_main_file_path_ = config_->code_path() + kBenchmarkPath;
  origin_umask_ = umask(user_umask_);
  MS_LOG(DEBUG) << "origin umask: " << origin_umask_ << ", user umask: " << user_umask_;
}

Generator::~Generator() { (void)umask(origin_umask_); }

void Generator::CodeNetRunFunc(std::ofstream &ofs) {
  // generate net inference code
  ofs << "void Inference() {\n";
  if (config_->support_parallel()) {
    ofs << gThreadNum << " = GetCurrentThreadNum(" << gThreadPool << ");\n ";
  }
  for (const auto &block : ctx_->code_blocks()) {
    ofs << "  {\n" << block << "  }\n";
  }
  ofs << "}\n";
}

int Generator::CodeSourceCMakeFile() {
  std::string src_cmake_file = net_src_file_path_ + cmake_file_name_;
  std::ofstream ofs(src_cmake_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << src_cmake_file;
  CodeCMakeNetLibrary(ofs, ctx_, config_);
  ofs.close();
  return RET_OK;
}

int Generator::CodeStaticContent() {
  std::vector<std::pair<std::string, std::string>> const_blocks = {
    {net_main_file_path_ + "load_input.h", load_input_h},
    {net_main_file_path_ + "load_input.c", load_input_c},
    {config_->code_path() + "/" + "CMakeLists.txt", bench_cmake_lists_txt},
    {net_main_file_path_ + "benchmark.cc", benchmark_source},
    {net_src_file_path_ + "CMakeLists.txt", src_cmake_lists_txt},
    {net_src_file_path_ + "session.h", session_header},
    {net_src_file_path_ + "tensor.h", tensor_header},
    {net_src_file_path_ + "tensor.cc", tensor_source},
    {net_src_file_path_ + "string.cc", string_source},
    {net_src_file_path_ + "model.h", model_header}};
  if (config_->support_parallel()) {
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "thread_pool.h", thread_header));
  }
  if (config_->debug_mode()) {
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "debug_utils.h", debug_utils_h));
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "debug_utils.c", debug_utils_c));
  }
  for (const auto &static_block : const_blocks) {
    std::string file_name = static_block.first;
    std::string content = static_block.second;
    MS_CHECK_RET_CODE(WriteContentToFile(file_name, content), "write file failed");
  }
  return RET_OK;
}

int Generator::CodeSessionImplement() {
  std::string cfile = net_src_file_path_ + "session.cc";
  std::ofstream ofs(cfile);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << cfile;
  ofs << g_hwLicense;
  ofs << "#include \"session.h\"\n";
  ofs << "#include \"model.h\"\n";
  ofs << "#include \"net.h\"\n";
  ofs << "#include <new>\n\n";
  CodeSessionCompileGraph(ofs, ctx_, config_);
  ofs << session_source;
  CodeCreateSessionImplement(ofs, config_);
  return RET_OK;
}

int Generator::CodeWeightFile() {
  // weight header file
  std::string hfile = net_src_file_path_ + net_weight_hfile_;
  std::ofstream hofs(hfile);
  MS_CHECK_TRUE(!hofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << hfile;
  CodeWeightFileHeader(hofs, ctx_);

  // weight source file
  std::string cfile = net_src_file_path_ + "weight.c";
  std::ofstream cofs(cfile);
  MS_CHECK_TRUE(!cofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << cfile;
  cofs << g_hwLicense;
  cofs << "#include \"" << net_weight_hfile_ << "\"\n\n";
  cofs << "int  " << gThreadNum << " = 1; \n";
  cofs << "unsigned char * " << ctx_->buffer_name() << " = 0; \n";

  if (config_->target() != kARM32M) {
    std::string net_file = net_src_file_path_ + "net.bin";
    SaveDataToNet(ctx_->saved_weights(), net_file);
    CodeModelParamsForNet(hofs, cofs, ctx_);
    CodeWeightInitFunc(cofs, ctx_);
  } else {
    CodeModelParamsState(hofs, ctx_->saved_weights());
    CodeModelParamsData(cofs, ctx_->saved_weights());
  }
  hofs.close();
  cofs.close();
  return RET_OK;
}

int Generator::GenerateCode() {
  MS_CHECK_RET_CODE(CodeNetHFile(), "code net h file failed.");
  MS_CHECK_RET_CODE(CodeNetCFile(), "code net c file failed.");
  MS_CHECK_RET_CODE(CodeWeightFile(), "code weight file failed.");
  MS_CHECK_RET_CODE(CodeSourceCMakeFile(), "code net cmake file failed.");
  MS_CHECK_RET_CODE(CodeStaticContent(), "code static content failed.");
  MS_CHECK_RET_CODE(CodeSessionImplement(), "code session file failed.");
  return RET_OK;
}
}  // namespace mindspore::lite::micro
