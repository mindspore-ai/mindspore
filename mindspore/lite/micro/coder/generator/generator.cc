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
#include <map>
#include <set>
#include <fstream>
#include "coder/generator/utils/generator_utils.h"
#include "coder/generator/const_blocks/cmake_lists_code.h"
#include "coder/generator/const_blocks/bench_debug_utils.h"
#include "coder/generator/const_blocks/bench_load_input.h"
#include "coder/generator/const_blocks/micro_tensor.h"
#include "coder/generator/const_blocks/license.h"
#include "micro/coder/log.h"

namespace mindspore::lite::micro {

Generator::Generator(std::unique_ptr<CoderContext> ctx) {
  ctx_ = std::move(ctx);
  this->config_ = Configurator::GetInstance();
  std::string module_name = config_->module_name();
  this->net_inc_hfile_ = module_name + ".h";
  this->net_src_cfile_ = module_name + ".c";
  this->net_weight_hfile_ = module_name + "_weight.h";
  this->net_main_cfile_ = module_name + "_benchmark.c";

  this->net_src_file_path_ = config_->code_path() + "/src/";
  this->net_inc_file_path_ = config_->code_path() + "/include/";
  this->net_main_file_path_ = config_->code_path() + "/benchmark/";
  origin_umask_ = umask(user_umask_);
  MS_LOG(DEBUG) << "origin umask: " << origin_umask_ << ", user umask: " << user_umask_;
}

Generator::~Generator() { (void)umask(origin_umask_); }

int Generator::CodeGraphInOutQuanArgs(std::ofstream &ofs) {
  std::vector<Tensor *> graph_inputs = ctx_->graph_inputs();
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "this graph has no input tensor";
    return RET_ERROR;
  }
  Tensor *in_tensor = graph_inputs.at(kInputIndex);
  MS_CHECK_PTR(in_tensor);
  std::vector<Tensor *> graph_outputs = ctx_->graph_outputs();
  if (graph_outputs.empty()) {
    MS_LOG(ERROR) << "this graph has no output tensor";
    return RET_ERROR;
  }
  Tensor *out_tensor = graph_outputs.at(kOutputIndex);
  MS_CHECK_PTR(out_tensor);
  std::vector<QuantArg> in_quant_args = in_tensor->quant_params();
  std::vector<QuantArg> out_quant_args = out_tensor->quant_params();
  if (in_quant_args.empty() || out_quant_args.empty()) {
    MS_LOG(WARNING) << "in_quant_args or out_quant_args is empty";
    return RET_OK;
  }
  ofs << "GraphQuantArgs " << config_->module_name() << "_GetInOutQuantArgs() {\n"
      << "\t\t"
      << "GraphQuantArgs quan_args = { " << in_quant_args.at(0).scale << ", " << out_quant_args.at(0).scale << ", "
      << in_quant_args.at(0).zeroPoint << ", " << out_quant_args.at(0).zeroPoint << "};\n"
      << "\t\t"
      << "return quan_args;\n"
      << "}\n";
  return RET_OK;
}

int Generator::CodeNetFileInputOutput(std::ofstream &ofs) {
  // input tensors
  ofs << "\n// set input tensors\n";
  std::vector<Tensor *> inputs = ctx_->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    ofs << "\nstatic const unsigned char *" << ctx_->input_name() + std::to_string(i) << " = 0;\n";
  }
  size_t size = inputs.size();
  ofs << "int " << config_->module_name() << "_SetInputs(const void **inputs, int num) {\n"
      << "if (inputs == NULL) {\n"
         "\treturn RET_ERROR;\n"
         "\t}\n"
      << "\tif (num !=" << size << ") { return RET_ERROR;}\n";
  for (size_t i = 0; i < size; ++i) {
    ofs << "\t" << ctx_->input_name() + std::to_string(i) << " = inputs[" << i << "];\n";
  }
  ofs << "\treturn RET_OK;\n}\n";

  // output tensors
  ofs << "\n// output tensors\n";
  std::vector<Tensor *> outputs = ctx_->graph_outputs();
  size_t output_num = outputs.size();
  std::string output_name = ctx_->output_name();

  ofs << "const MicroTensorList* " << config_->module_name() << "_GetOutputs() {\n"
      << "  static MicroTensor " << output_name << "[" << output_num << "] ;\n";

  if (PrintMicroTensors(ofs, outputs, output_name, ctx_->tensors_map()) != RET_OK) {
    return RET_ERROR;
  }
  ofs << "  static MicroTensorList  " << config_->module_name() << "_TensorArray;\n"
      << "  " << config_->module_name() << "_TensorArray.num = " << output_num << ";\n"
      << "  " << config_->module_name() << "_TensorArray.tensor = &" << output_name << "[0];\n"
      << "  return  &" << config_->module_name() << "_TensorArray; \n}\n";
  return RET_OK;
}

void Generator::CodeNetFileMembuffer(std::ofstream &ofs) {
  // memory buffer
  ofs << "\n// Get MemBuffer Size\n"
      << "unsigned int " << config_->module_name() << "_GetBufferSize() {\n"
      << "\t return " << ctx_->total_buffer_size() << "; \n}\n";

  ofs << "\n// set Membuffer address\n";
  ofs << "int " << config_->module_name() << "_SetBuffer( void *buffer) { \n";
  ofs << "\tif (buffer == NULL) {\n"
         "\t\tMICRO_ERROR(\"memory buffer is NULL\");\n"
         "\t\treturn RET_ERROR;\n"
         "\t}\n";
  ofs << "\t" << ctx_->buffer_name()
      << "= buffer; \n"
         "\treturn RET_OK;";
  ofs << "}\n";
}

void Generator::CodeNetFileInclude(std::ofstream &ofs) {
  ofs << g_hwLicense;
  // need copy head file of microtensor ro dst'dirs
  ofs << "#include \"microtensor.h\"\n";
  // copy debug head files to cmake include files
  ofs << "#include \"" << net_weight_hfile_ << "\"\n"
      << "#include \"" << net_inc_hfile_ << "\"\n";
  if (config_->debug_mode()) {
    ofs << "#include \"../benchmark/debug_utils.h\"\n";
  }
}

void Generator::CodeNetRunFunc(std::ofstream &ofs) {
  // generate net predict code
  ofs << "void " << config_->module_name() << "_Inference() {\n";
  if (config_->code_mode() == CodeMode::Code_Android) {
    ofs << "int thread_num = GetCurrentThreadNum(THREAD_POOL_DEFAULT);\n";
  }
  for (const auto &codeBlock : ctx_->code_blocks()) {
    ofs << "\t{\n";
    ofs << codeBlock;
    ofs << "\t}\n";
  }
  ofs << "}\n";
}

int Generator::CodeTestCMakeFile() {
  std::string net_main_cmake_file_path = net_main_file_path_;
  std::string test_cmake_file = net_main_cmake_file_path + "benchmark.cmake";
  std::ofstream of(test_cmake_file);
  if (of.bad()) {
    MS_LOG(ERROR) << "open file error " << test_cmake_file;
    return RET_ERROR;
  }

  MS_LOG(INFO) << "write " << test_cmake_file;
  of << "include_directories(${CMAKE_CURRENT_SOURCE_DIR})\n";
  of << "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/)\n";
  of << "set(SRC_FILES\n";
  of << "\t\t" << config_->module_name() + "_benchmark.c\n";
  of << "\t\tload_input.c\n";
  of << "\t\tdebug_utils.c\n";
  of << ")\n";
  of.close();
  return RET_OK;
}

int Generator::CodeCMakeExecutableFile(std::ofstream &ofs) const {
  ofs << "include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/)\n";
  if (config_->target() == kARM32M) {
    IncludeCmsisDirectories(ofs);
  }
  ofs << "set(OP_SRC\n";
  for (const std::string &c_file : ctx_->c_files()) {
    ofs << "    " << c_file << ".o\n";
  }
  ofs << "    " << config_->module_name() << "_weight.c.o\n";
  ofs << "    " << config_->module_name() << ".c.o\n";
  ofs << ")\n";

  std::set<std::string> kernel_cmake_asm_set_files = ctx_->asm_files();
  if (!kernel_cmake_asm_set_files.empty()) {
    ofs << "set(ASSEMBLY_SRC\n";
    for (const std::string &asm_file : kernel_cmake_asm_set_files) {
      ofs << "    " << asm_file << ".o\n";
    }
    ofs << ")\n";
    ofs << "set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)\n";
    ofs << "list(APPEND OP_SRC ${ASSEMBLY_SRC})\n";
  }

  ofs << "file(GLOB NET_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.c)\n";
  ofs << "add_library(${PROJ_NAME} STATIC ${NET_SRC})\n";
  return RET_OK;
}

int Generator::CodeCMakeFile() {
  std::string src_cmake_file = net_src_file_path_ + cmake_file_name_;
  std::ofstream of(src_cmake_file);
  if (of.bad()) {
    MS_LOG(ERROR) << "open file error " << src_cmake_file;
    return RET_ERROR;
  }
  MS_LOG(INFO) << "write " << src_cmake_file.c_str();
  if (CodeCMakeExecutableFile(of) != RET_OK) {
    of.close();
    return RET_ERROR;
  }
  of.close();
  return RET_OK;
}

int Generator::CodeStaticContent() {
  const std::vector<std::pair<std::string, std::string>> static_blocks = {
    {net_inc_file_path_ + "microtensor.h", micro_tensor_h},
    {net_src_file_path_ + "CMakeLists.txt", src_cmake_lists_txt},
    {net_main_file_path_ + "debug_utils.h", debug_utils_h},
    {net_main_file_path_ + "debug_utils.c", debug_utils_c},
    {net_main_file_path_ + "load_input.h", load_input_h},
    {net_main_file_path_ + "load_input.c", load_input_c},
    {net_main_file_path_ + "CMakeLists.txt", bench_cmake_lists_txt}};
  for (const auto &static_block : static_blocks) {
    std::string file_name = static_block.first;
    std::string content = static_block.second;
    if (WriteContentToFile(file_name, content) != RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void Generator::CodeWeightInitFunc(const std::map<std::string, Tensor *> &address_map, std::ofstream &ofs) {
  ofs << "int " << config_->module_name() << "_Init(void *weight_buffer, int weight_size) {\n"
      << "\tif (weight_buffer == NULL) {\n"
         "\t\tMICRO_ERROR(\"weight buffer is NULL\");\n"
      << "\t\treturn RET_ERROR;\n"
      << "\t}\n";
  CodeReadModelParams(ctx_->saved_weights(), ctx_->tensors_map(), ofs);
  for (const auto &block : ctx_->init_contents()) {
    ofs << "{\n" << block << "\n}\n";
  }
  ofs << "return RET_OK;";
  ofs << "}\n\n";
}

void Generator::CodeFreeResource(const std::map<std::string, Tensor *> &address_map, std::ofstream &ofs) const {
  ofs << "\tvoid *allocated[] = {";
  size_t num = 0;
  for (const auto &item : address_map) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (tensor->data_c() != nullptr && tensor->category() != Tensor::Category::CONST_TENSOR) {
      ofs << name << ", ";
      num++;
    }
  }
  ofs << "\t};\n";

  ofs << "\tfor (int i = 0; i < " << num << "; ++i) {\n";
  ofs << "\t\tfree(allocated[i]);\n";
  ofs << "\t\tallocated[i] = NULL;\n";
  ofs << "\t}\n";
}

int Generator::CodeWeightFile() {
  // weight header file
  std::string hfile = net_src_file_path_ + net_weight_hfile_;
  std::ofstream hofs(hfile);
  if (hofs.bad()) {
    MS_LOG(ERROR) << "open file error" << hfile;
    return RET_ERROR;
  }
  hofs << g_hwLicense;
  for (const auto &h_file : ctx_->h_files()) {
    hofs << "#include \"" << h_file << "\"\n";
  }
  hofs << "#include <stdlib.h>\n";
  hofs << "#include <string.h>\n";
  hofs << "#include \"microtensor.h\"\n\n";
  hofs << "extern unsigned char *" << ctx_->buffer_name() << ";\n";

  // weight source file
  std::string cfile = net_src_file_path_ + config_->module_name() + "_weight.c";
  std::ofstream cofs(cfile);
  if (cofs.bad()) {
    MS_LOG(ERROR) << "open file error" << cfile;
    return RET_ERROR;
  }
  cofs << g_hwLicense;
  cofs << "#include \"" << net_weight_hfile_ << "\"\n\n";
  cofs << "unsigned char * " << ctx_->buffer_name() << " = 0 ; \n";

  // reverse key and value of tensors_map
  std::map<std::string, Tensor *> address_map;
  for (const auto &item : ctx_->tensors_map()) {
    address_map.insert(std::make_pair(item.second, item.first));
  }

  if (config_->is_weight_file()) {
    std::string net_file = net_src_file_path_ + config_->module_name() + ".net";
    if (SaveDataToNet(ctx_->saved_weights(), net_file) != RET_OK) {
      hofs.close();
      cofs.close();
      return RET_ERROR;
    }
    CodeModelParamsDefine(address_map, hofs, cofs);
    CodeWeightInitFunc(address_map, cofs);
  } else {
    CodeModelParamsDefineAndData(ctx_->saved_weights(), hofs, cofs);
  }

  hofs.close();
  cofs.close();
  return RET_OK;
}

int Generator::GenerateCode() {
  MS_CHECK_RET_CODE(CodeNetHFile(), "code net h file failed.");
  MS_CHECK_RET_CODE(CodeNetCFile(), "code net c file failed.");
  MS_CHECK_RET_CODE(CodeWeightFile(), "code weight file failed.");
  MS_CHECK_RET_CODE(CodeCMakeFile(), "code net cmake file failed.");
  MS_CHECK_RET_CODE(CodeTestFile(), "code test file failed.");
  MS_CHECK_RET_CODE(CodeTestCMakeFile(), "code test cmake file failed.");
  MS_CHECK_RET_CODE(CodeStaticContent(), "code static content failed.");
  return RET_OK;
}
}  // namespace mindspore::lite::micro
