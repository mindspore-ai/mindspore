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
#include "coder/generator/component/const_blocks/calib_output.h"
#include "coder/generator/component/const_blocks/msession.h"
#include "coder/generator/component/const_blocks/mtensor.h"
#include "coder/generator/component/const_blocks/mcontext.h"
#include "coder/generator/component/const_blocks/benchmark.h"
#include "coder/generator/component/const_blocks/benchmark_train.h"
#include "coder/generator/component/const_blocks/license.h"
#include "coder/generator/component/train_component.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/kernel_registry.h"

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

int Generator::CodeSourceCMakeFile() {
  std::string src_cmake_file = net_src_file_path_ + cmake_file_name_;
  std::ofstream ofs(src_cmake_file);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << src_cmake_file;
  CodeCMakeNetLibrary(ofs, ctx_, config_);
  ofs.close();
  return RET_OK;
}

int Generator::CodeDataCFile() {
  std::string cfile = net_main_file_path_ + "data.c";
  std::ofstream cofs(cfile);
  MS_CHECK_TRUE(!cofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << cfile;
  cofs << g_hwLicense;
  cofs << "#include \"data.h\"\n";

  auto inputs_num = ctx_->graph_inputs().size();
  auto outputs_num = ctx_->graph_outputs().size();

  cofs << "#define NET_INPUTS_NUM " << inputs_num << "\n";
  cofs << "#define NET_OUTPUTS_NUM " << outputs_num << "\n";
  std::stringstream data_def;
  std::stringstream calib_data_def;
  std::stringstream input_tensors_def;
  std::stringstream output_tensors_def;
  std::stringstream calib_input_tensors_def;
  std::stringstream calib_output_tensors_def;
  for (size_t i = 0; i < inputs_num; i++) {
    Tensor *tensor = ctx_->graph_inputs()[i];
    cofs << "#define NET_INPUT" << i << "_SIZE " << tensor->ElementsNum() << "\n";
    data_def << "float input" << i << "_data[NET_INPUT" << i << "_SIZE];\n";
    calib_data_def << "float calib_input" << i << "_data[NET_INPUT" << i << "_SIZE] = {};\n";
    input_tensors_def << "  {\n"
                      << "    \"" << tensor->tensor_name() << "\",\n"
                      << "    NET_INPUT" << i << "_SIZE,\n"
                      << "    NET_INPUT" << i << "_SIZE * sizeof(float),\n"
                      << "    input" << i << "_data,\n"
                      << "  },\n";
    calib_input_tensors_def << "  {\n"
                            << "    \"" << tensor->tensor_name() << "\",\n"
                            << "    NET_INPUT" << i << "_SIZE,\n"
                            << "    NET_INPUT" << i << "_SIZE * sizeof(float),\n"
                            << "    calib_input" << i << "_data,\n"
                            << "  },\n";
  }
  for (size_t i = 0; i < outputs_num; i++) {
    Tensor *tensor = ctx_->graph_outputs()[i];
    cofs << "#define NET_OUTPUT" << i << "_SIZE " << tensor->ElementsNum() << "\n";
    data_def << "float output" << i << "_data[NET_OUTPUT" << i << "_SIZE];\n";
    calib_data_def << "float calib_output" << i << "_data[NET_OUTPUT" << i << "_SIZE] = {};\n";
    output_tensors_def << "  {\n"
                       << "    \"" << tensor->tensor_name() << "\",\n"
                       << "    NET_OUTPUT" << i << "_SIZE,\n"
                       << "    NET_OUTPUT" << i << "_SIZE * sizeof(float),\n"
                       << "    output" << i << "_data,\n"
                       << "  },\n";
    calib_output_tensors_def << "  {\n"
                             << "    \"" << tensor->tensor_name() << "\",\n"
                             << "    NET_OUTPUT" << i << "_SIZE,\n"
                             << "    NET_OUTPUT" << i << "_SIZE * sizeof(float),\n"
                             << "    calib_output" << i << "_data,\n"
                             << "  },\n";
  }

  cofs << "\n" << data_def.str() << calib_data_def.str();
  cofs << "\nTensor inputs_tensors[NET_INPUTS_NUM] = {\n" << input_tensors_def.str() << "};\n";
  cofs << "\nTensor outputs_tensors[NET_OUTPUTS_NUM] = {\n" << output_tensors_def.str() << "};\n";
  cofs << "\nTensor calib_inputs_tensors[NET_INPUTS_NUM] = {\n" << calib_input_tensors_def.str() << "};\n";
  cofs << "\nTensor calib_outputs_tensors[NET_OUTPUTS_NUM] = {\n" << calib_output_tensors_def.str() << "};\n";

  cofs << "TensorArray g_inputs = {inputs_tensors, NET_INPUTS_NUM};\n";
  cofs << "TensorArray g_outputs = {outputs_tensors, NET_OUTPUTS_NUM};\n";
  cofs << "TensorArray g_calib_inputs = {calib_inputs_tensors, NET_INPUTS_NUM};\n";
  cofs << "TensorArray g_calib_outputs = {calib_outputs_tensors, NET_OUTPUTS_NUM};\n";

  cofs.close();
  return RET_OK;
}

int Generator::CodeBenchmarkHFile(const std::string &file) {
  std::ofstream of(file);
  if (of.bad()) {
    MS_LOG(ERROR) << "open file error " << file;
    return RET_ERROR;
  }
  MS_LOG(INFO) << "write " << file;
  of << R"RAW(/**
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

#ifndef MINDSPORE_LITE_MICRO_BENCHMARK_H_
#define MINDSPORE_LITE_MICRO_BENCHMARK_H_
#ifdef __cplusplus
extern "C" {
#endif

)RAW";
  size_t shape_size = 0;
  std::vector<Tensor *> inputs = ctx_->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    shape_size += inputs[i]->shape().size();
  }
  std::vector<Tensor *> outputs = ctx_->graph_outputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    shape_size += outputs[i]->shape().size();
  }
  constexpr int kMicroTensorSize = 32;
  size_t workspace_size = UP_ROUND(ctx_->total_buffer_size(), C4NUM) + UP_ROUND(ctx_->weight_buffer_size(), C4NUM) +
                          shape_size * sizeof(int64_t) +
                          (kMicroTensorSize) * (ctx_->graph_inputs().size() + ctx_->graph_outputs().size());

  of << "#define WORK_SPACE_SIZE " << workspace_size << "\n";
  of << R"RAW(

int benchmark();

#ifdef __cplusplus
}
#endif
#endif //MINDSPORE_LITE_MICRO_BENCHMARK_H_

)RAW";
  of.close();
  return RET_OK;
}

int Generator::CodeStaticContent() {
  std::string bench_cmake_lists_txt = bench_cmake_lists;
  std::string calib_header_txt = calib_header;
  std::string calib_source_txt = calib_source;
  std::string load_input_h_txt = load_input_h;
  std::string load_input_c_txt = load_input_c;
  std::string benchmark_source_txt = benchmark_source;
  std::string src_cmake_lists_txt = src_cmake_lists;
  std::string context_header_txt = context_header;
  std::string context_source_txt = context_source;
  std::string tensor_header_txt = tensor_header;
  std::string tensor_source_txt = tensor_source;
  if (config_->code_mode() == CodeMode::Train) {
    benchmark_source_txt = benchmark_train_source;
  }
  if (config_->target() == kCortex_M) {
    bench_cmake_lists_txt = bench_cmake_lists_cortex;
    calib_header_txt = calib_header_cortex;
    calib_source_txt = calib_source_cortex;
    load_input_h_txt = load_input_h_cortex;
    load_input_c_txt = load_input_c_cortex;
    benchmark_source_txt = benchmark_source_cortex;
    context_source_txt = context_source_cortex;
  } else if (config_->support_parallel() == false) {
    context_source_txt = context_source_no_parallel;
  }

  std::vector<std::pair<std::string, std::string>> const_blocks = {
    {config_->code_path() + "/" + "CMakeLists.txt", bench_cmake_lists_txt},
    {net_main_file_path_ + "calib_output.h", calib_header_txt},
    {net_main_file_path_ + "calib_output.c", calib_source_txt},
    {net_main_file_path_ + "load_input.h", load_input_h_txt},
    {net_main_file_path_ + "load_input.c", load_input_c_txt},
    {net_main_file_path_ + "benchmark.c", benchmark_source_txt},
    {net_src_file_path_ + "CMakeLists.txt", src_cmake_lists_txt},
    {net_src_file_path_ + "context.h", context_header_txt},
    {net_src_file_path_ + "context.c", context_source_txt},
    {net_src_file_path_ + "tensor.h", tensor_header_txt},
    {net_src_file_path_ + "tensor.c", tensor_source_txt}};

  if (config_->debug_mode()) {
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "debug_utils.h", debug_utils_h));
    const_blocks.emplace_back(std::make_pair(net_src_file_path_ + "debug_utils.c", debug_utils_c));
  }
  if (config_->target() == kCortex_M) {
    CodeBenchmarkHFile(net_main_file_path_ + "benchmark.h");
    const_blocks.emplace_back(std::make_pair(net_main_file_path_ + "data.h", data_h_cortex));
    const_blocks.emplace_back(
      std::make_pair(config_->code_path() + "/" + "cortex-m7.toolchain.cmake", cortex_m7_toolchain));
    const_blocks.emplace_back(std::make_pair(config_->code_path() + "/" + "build.sh", cortex_build_sh));
    MS_CHECK_RET_CODE(CodeDataCFile(), "code data c file failed.");
  }
  for (const auto &static_block : const_blocks) {
    std::string file_name = static_block.first;
    std::string content = static_block.second;
    MS_CHECK_RET_CODE(WriteContentToFile(file_name, content), "write file failed");
  }
  return RET_OK;
}

int Generator::CodeMSModelImplement() {
  std::string cfile = net_src_file_path_ + "model.c";
  std::ofstream ofs(cfile);
  MS_CHECK_TRUE(!ofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << cfile;
  ofs << g_hwLicense;
  ofs << "#include \"tensor.h\"\n";
  ofs << "#include \"context.h\"\n";
  ofs << "#include \"c_api/model_c.h\"\n";
  ofs << "#include \"net.h\"\n";
  if (config_->support_parallel()) {
    ofs << "#include \"" << kThreadWrapper << "\"\n";
  }
  ofs << "#include \"weight.h\"\n\n";
  CodeMSTensorHandleArrayDestroyState(ofs, *config_);
  CodeMSModelCreate(ofs, ctx_, *config_);
  CodeMSModelCalcWorkspaceSize(ofs, ctx_, *config_);
  CodeMSModelSetWorkspace(ofs, ctx_, *config_);
  CodeMSModelBuild(ofs, ctx_->GetCurModelIndex(), config_);
  ofs << model_runtime_other_source;
  if (config_->code_mode() == CodeMode::Train) {
    CodeMSModelRunStep(ofs, ctx_);
    CodeMSModelSetTrainMode(ofs, ctx_);
    CodeMSModelExportWeight(ofs, ctx_->GetCurModelIndex());
  } else {
    CodeMSModelPredict(ofs, ctx_);
  }
  CodeMSModelDestory(ofs, ctx_->GetCurModelIndex(), config_);
  return RET_OK;
}

int Generator::CodeWeightFile() {
  // weight header file
  std::string hfile = net_src_file_path_ + net_weight_hfile_;
  std::ofstream hofs(hfile);
  MS_CHECK_TRUE(!hofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << hfile;
  CodeWeightFileHeader(hofs, ctx_, *config_);

  // weight source file
  std::string cfile = net_src_file_path_ + "weight.c";
  std::ofstream cofs(cfile);
  MS_CHECK_TRUE(!cofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << cfile;
  cofs << g_hwLicense;
  cofs << "#include \"" << net_weight_hfile_ << "\"\n\n";
  cofs << "#include <stdio.h>\n\n";
  cofs << "int  " << gThreadNum << " = 1; \n";
  std::vector<Tensor *> inputs = ctx_->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    cofs << "extern const unsigned char *" << ctx_->input_name() + std::to_string(i) << ";\n";
  }
  if (config_->target() != kCortex_M) {
    cofs << "unsigned char *" << ctx_->buffer_name() << " = 0; \n";
    cofs << "unsigned char *" << ctx_->weight_name() << " = 0; \n";
    std::string net_file = net_src_file_path_ + "net.bin";
    SaveDataToNet(ctx_->saved_weights(), net_file);
  } else {
    if (!ctx_->weight_buffer_size_code_blocks().empty()) {
      MS_LOG(ERROR) << "Weight init code generation error ";
      return RET_ERROR;
    }
    cofs << "int __errno; \n";
    cofs << "unsigned char * " << ctx_->buffer_name() << " = NULL; \n";
    cofs << "unsigned char * " << ctx_->weight_name() << " = NULL; \n";
    CodeModelParamsData(cofs, ctx_->saved_weights());
  }
  CodeModelParamsForNet(hofs, cofs, ctx_, *config_);
  CodeInitWeightState(hofs, ctx_->GetCurModelIndex());
  if (config_->code_mode() == CodeMode::Train) {
    CodeWeightInitFuncForTrain(cofs, ctx_);
    CodeExportWeightState(hofs, ctx_->GetCurModelIndex());
    CodeWeightExportFunc(cofs, ctx_, *config_);
  } else {
    CodeWeightInitFunc(cofs, ctx_, *config_);
  }
  hofs.close();
  cofs.close();
  return RET_OK;
}

void Generator::CodeCommonNetH(std::ofstream &ofs) {
  ofs << g_hwLicense;
  ofs << kExternCpp;
  CodeInputState(ofs, ctx_->GetCurModelIndex());
  if (is_get_quant_args_) {
    CodeGraphQuantArgsState(ofs, ctx_->GetCurModelIndex());
  }
  CodeManageResourceState(ofs, ctx_->GetCurModelIndex());
  CodeExecuteState(ofs, ctx_->GetCurModelIndex());
}

void Generator::CodeCommonNetC(std::ofstream &ofs) {
  ofs << g_hwLicense << "\n"
      << "#include \"" << net_weight_hfile_ << "\"\n"
      << "#include \"" << net_inc_hfile_ << "\"\n\n";
  if (config_->support_parallel()) {
    ofs << "#include \"" << kThreadWrapper << "\"\n\n";
  }
  if (config_->debug_mode()) {
    ofs << "#include \"" << kDebugUtils << "\"\n";
  }
  CodeGlobalCodeBlocks(ofs, ctx_);
  CodeInputImplement(ofs, ctx_);
  CodeInitResourceImplement(ofs, ctx_);
  CodeFreeResourceImplement(ofs, ctx_, *config_);
  if (is_get_quant_args_) {
    CodeGraphQuantArgsImplement(ofs, ctx_);
  }
}

int Generator::CodeRegKernelHFile() {
  if (!KernelRegistry::GetInstance()->HasKernelRegistered()) return RET_OK;
  if (!KernelRegistry::GetInstance()->CheckRegistered(schema::PrimitiveType_Custom)) {
    MS_LOG(ERROR) << "Only support custom kernel to register now!";
    return RET_ERROR;
  }

  std::string reg_kernel_header = net_src_file_path_ + "registered_kernel.h";
  std::ofstream cofs(reg_kernel_header);
  MS_CHECK_TRUE(!cofs.bad(), "filed to open file");
  MS_LOG(INFO) << "write " << reg_kernel_header;
  cofs << g_hwLicense;
  cofs << "#include \"nnacl/tensor_c.h\"\n";
  cofs << "#include \"nnacl/custom_parameter.h\"\n\n";
  cofs << KernelRegistry::GetInstance()->GenKernelInterface(kCustomKernelName, kCustomKernelParam) << "\n";
  return RET_OK;
}

int Generator::GenerateCode() {
  MS_CHECK_RET_CODE(CodeNetHFile(), "code net h file failed.");
  MS_CHECK_RET_CODE(CodeNetCFile(), "code net c file failed.");
  MS_CHECK_RET_CODE(CodeWeightFile(), "code weight file failed.");
  MS_CHECK_RET_CODE(CodeSourceCMakeFile(), "code net cmake file failed.");
  MS_CHECK_RET_CODE(CodeStaticContent(), "code static content failed.");
  MS_CHECK_RET_CODE(CodeMSModelImplement(), "code session file failed.");
  MS_CHECK_RET_CODE(CodeRegKernelHFile(), "code registered kernel header file failed.");
  return RET_OK;
}
}  // namespace mindspore::lite::micro
