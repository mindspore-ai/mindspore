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

#include "coder/generator/inference/inference_generator.h"
#include <vector>
#include <map>
#include <set>
#include <string>
#include "coder/generator/const_blocks/license.h"

namespace mindspore::lite::micro {

int InferenceGenerator::CodeNetHFile() {
  std::string net_include_file = net_inc_file_path_ + net_inc_hfile_;
  std::ofstream ofs(net_include_file);
  if (ofs.bad()) {
    MS_LOG(ERROR) << "open file error " << net_include_file.c_str();
    return RET_ERROR;
  }
  ofs << g_hwLicense;
  if (config_->code_mode() == CodeMode::Code_Android) {
    ofs << "#include \"src/runtime/thread_pool.h\"\n";
  }
  ofs << "#include \"microtensor.h\"\n\n";

  ofs << "/**\n"
      << "  * set input tensors\n"
      << "  * @param inputs, the input data ptr's array of the model, the tensors' count of input may be greater than "
         "one.\n"
      << "  * @param num, the input data's number of the model.\n"
      << "  **/\n"
      << "int " << config_->module_name() << "_SetInputs(const void **inputs, int num);\n\n";

  ofs << "/**\n"
      << "  * get output tensor of the model \n"
      << "  **/\n"
      << "const MicroTensorList *" << config_->module_name() << "_GetOutputs();\n\n";

  if (is_get_quant_args_) {
    std::vector<Tensor *> graph_inputs = ctx_->graph_inputs();
    if (graph_inputs.empty()) {
      MS_LOG(ERROR) << "this graph has no input tensor";
      ofs.close();
      return RET_ERROR;
    }
    size_t total_input_size = std::accumulate(
      graph_inputs.begin(), graph_inputs.end(), 0UL,
      [](size_t total_input_size, const Tensor *const tensor) { return total_input_size += tensor->Size(); });
    ofs << "/**\n";
    ofs << "  * get input sizes of the model \n";
    ofs << "  **/\n";
    ofs << "inline int " << config_->module_name() << "_GetInputSizes() {\n"
        << "\t\t"
        << "return " << total_input_size << ";\n"
        << "}\n\n";

    ofs << "/**\n";
    ofs << "  * get input and output QuantArgs of the model \n";
    ofs << "  **/\n";
    ofs << "GraphQuantArgs " << config_->module_name() << "_GetInOutQuantArgs();\n\n";
  }

  if (config_->is_weight_file()) {
    ofs << "/**\n"
        << "  * @param weightBuffer, the ptr of the model's parameters\n"
        << "  * @param weightSize, the size of the model's parameters\n"
        << "  **/\n"
        << "int " << config_->module_name() << "_Init(void *weightBuffer, int weightSize);\n\n";
  }

  ofs << "/**\n"
      << "  * free the memory of packed weights and model's workspace buffer, input address\n"
      << "  **/\n"
      << "void " << config_->module_name() << "_FreeResource();\n";

  ofs << "/**\n"
      << "  * get the memory space size of the inference.\n"
      << "  **/\n"
      << "unsigned int " << config_->module_name() << "_GetBufferSize();\n";

  ofs << "/**\n"
      << "  * set the memory space for the inference\n"
      << "  **/\n"
      << "int " << config_->module_name() << "_SetBuffer(void *buffer);\n\n";

  ofs << "/**\n"
      << "  * net inference function\n"
      << "  **/\n"
      << "void " << config_->module_name() << "_Inference();\n\n";
  return RET_OK;
}

int InferenceGenerator::CodeNetCFile() {
  std::string net_impl_file = net_src_file_path_ + net_src_cfile_;
  std::ofstream ofs(net_impl_file);
  if (ofs.bad()) {
    MS_LOG(ERROR) << "open file error" << net_impl_file.c_str();
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "write " << net_impl_file.c_str();
  CodeNetFileInclude(ofs);
  CodeNetFileMembuffer(ofs);
  if (is_get_quant_args_) {
    if (CodeGraphInOutQuanArgs(ofs) != RET_OK) {
      MS_LOG(ERROR) << "CodeGraphInOutQuanArgs failed";
      ofs.close();
      return RET_ERROR;
    }
  }
  if (CodeNetFileInputOutput(ofs) != RET_OK) {
    ofs.close();
    return RET_ERROR;
  }
  ofs << "void " << config_->module_name() << "_FreeResource() {\n";
  ofs << "\t" << ctx_->buffer_name() << "= NULL;\n";
  std::vector<Tensor *> inputs = ctx_->graph_inputs();
  size_t size = inputs.size();
  for (size_t i = 0; i < size; ++i) {
    ofs << "\t" << ctx_->input_name() + std::to_string(i) << " = NULL;\n";
  }
  std::map<std::string, Tensor *> address_map;
  for (const auto &item : ctx_->tensors_map()) {
    address_map.insert(std::make_pair(item.second, item.first));
  }
  if (config_->is_weight_file()) {
    CodeFreeResource(address_map, ofs);
  }
  ofs << "}\n";
  CodeNetRunFunc(ofs);
  ofs.close();
  return RET_OK;
}

void InferenceGenerator::CodeTestRelevantHeader(std::ofstream &code_test_ofs) {
  code_test_ofs << g_hwLicense;
  code_test_ofs << "#include <stdio.h>\n"
                   "#include <string.h>\n"
                   "#include <stdlib.h>\n"
                   "#include <stdint.h>\n"
                   "#include \"microtensor.h\"\n"
                   "#include \"load_input.h\"\n"
                   "#include \"debug_utils.h\"\n";
  code_test_ofs << "#include \"" << net_inc_hfile_ << "\"\n";

  code_test_ofs << "/**\n"
                   " * mindspore micro params usage:\n"
                   " * args[0]: executable file\n"
                   " * args[1]: inputs .bin file\n"
                   " * args[2]: model weight .net file\n"
                   " * args[3]: loop count for performance testing\n"
                   " * args[4]: runtime thread num\n"
                   " * args[5]: runtime thread bind mode\n"
                   " */\n";

  code_test_ofs << "\n// Warm up. \n"
                << "void " << config_->module_name() << "_WarmUp() {\n"
                << "\tfor (int i = 0; i < " << kWarmUp << "; ++i) {\n"
                << "\t\t" << config_->module_name() << "_Inference();\n"
                << "\t}\n"
                << "}\n";
}

void InferenceGenerator::CodeTestRelevantTile(std::ofstream &code_test_ofs) {
  if (config_->code_mode() == Code_Android) {
    code_test_ofs << "  DestroyThreadPool(THREAD_POOL_DEFAULT);\n";
  }
  code_test_ofs << "  // print model outputs \n";
  code_test_ofs << "  const MicroTensorList *outs = " << config_->module_name() << "_GetOutputs();\n";
  code_test_ofs << "  for (int i = 0; i < outs->num; ++i) {\n"
                   "      MicroTensor *tensor = outs->tensor + i;\n"
                   "      PrintTensorData(tensor);\n"
                   "  }\n";
  code_test_ofs << "  printf(\"" << config_->module_name() << " inference End.\\n\");\n";
  code_test_ofs << "  free(buffer);\n";
  code_test_ofs << "  " << config_->module_name() << "_FreeResource();\n";
  std::vector<Tensor *> inputs = ctx_->graph_inputs();
  size_t inputs_num = inputs.size();
  code_test_ofs << "  // this code_block can be ignore \n";
  code_test_ofs << "  for (int i = 0; i < " << inputs_num
                << "; ++i) {\n"
                   "    free(inputs_binbuf[i]);\n"
                   "  }\n";
  code_test_ofs << "  return 0;\n";
  code_test_ofs << "}\n";
}

int InferenceGenerator::CodeTestFile() {
  std::string net_main_impl_file = net_main_file_path_ + net_main_cfile_;
  std::ofstream ofs(net_main_impl_file);
  if (ofs.bad()) {
    MS_LOG(ERROR) << "open file error " << net_main_impl_file.c_str();
    return RET_ERROR;
  }
  MS_LOG(INFO) << "write " << net_main_impl_file.c_str();
  CodeTestRelevantHeader(ofs);
  ofs << "int main(int argc, char **argv) {\n"
         "  if (argc < 2) { printf(\"There is not input and out file.\\n\"); }\n";
  ofs << "  printf(\"" << config_->module_name() << " inference Start.\\n\");\n";
  std::vector<Tensor *> inputs = ctx_->graph_inputs();
  size_t inputs_num = inputs.size();
  for (size_t i = 0; i < inputs_num; ++i) {
    Tensor *input = inputs[i];
    std::vector<int> shape = input->shape();
    ofs << "  // model's input_shape is [ ";
    for (int sh : shape) {
      ofs << sh << ", ";
    }
    ofs << "];\n";
  }
  ofs << "  void *inputs_binbuf[" << inputs_num << "];\n";
  ofs << "  int inputs_size[" << inputs_num << "] = {";
  for (size_t i = 0; i < inputs_num; ++i) {
    Tensor *input = inputs[i];
    ofs << input->Size() << ", ";
  }
  ofs << "};\n";
  ofs << "  int ret = ReadInputsFile(argv[1], inputs_binbuf, inputs_size, " << inputs_num
      << ");\n"
         "  if (ret != RET_OK) {\n"
         "    MICRO_ERROR(\"read inputs file failed\");\n"
         "    return RET_ERROR;\n"
         "  }\n";
  ofs << "  ret = " << config_->module_name() << "_SetInputs((const void **)inputs_binbuf, " << inputs_num
      << ");\n"
         "  if (ret != RET_OK) {\n"
         "    MICRO_ERROR(\"set inputs failed\");\n"
         "    return RET_ERROR;\n"
         "  }\n";

  ofs << "  unsigned int total_buffer_size = " << config_->module_name() << "_GetBufferSize();\n";
  ofs << "  void *buffer = malloc(total_buffer_size);\n";
  ofs << "  if (buffer == NULL ){\n"
         "     MICRO_ERROR(\"malloc memory buffer failed\");\n"
         "     return RET_ERROR;\n"
         "  }\n";
  ofs << "  ret = " << config_->module_name()
      << "_SetBuffer(buffer);\n"
         "  if (ret != RET_OK) {\n"
         "    MICRO_ERROR(\"set inputs failed\");\n"
         "    return RET_ERROR;"
         "  }\n";
  if (config_->is_weight_file()) {
    ofs << "  int weightSize = 0;\n";
    ofs << "  void *weightBuffer = ReadInputData(argv[2], &weightSize); \n";
    ofs << "  if(" << config_->module_name() << "_Init(weightBuffer, weightSize) != RET_OK) {\n";
    ofs << "    printf(\"model init failed\");\n";
    ofs << "    " << config_->module_name() << "_FreeResource();\n";
    ofs << "    return RET_ERROR;\n";
    ofs << "  }\n";
    ofs << "  free(weightBuffer);\n";
    ofs << "  weightBuffer = NULL;\n";
  }

  if (config_->code_mode() == CodeMode::Code_Android) {
    ofs << "  int thread_num = 4;\n"
           "  BindMode bind_mode = NO_BIND_MODE;\n"
           "  if (argc >= 6) {\n"
           "    thread_num = atoi(argv[4]);\n"
           "    bind_mode = atoi(argv[5]);\n"
           "  }\n"
           "  ret = ConfigThreadPool(THREAD_POOL_DEFAULT, thread_num, bind_mode);\n"
           "  if (ret != 0) {\n"
           "    printf(\"create thread pool failed\");\n"
           "  }\n";
  }
  ofs << "  if (argc >= 4) {\n"
      << "    " << config_->module_name() << "_WarmUp();\n"
      << "    uint64_t timeAvg = 0;\n"
      << "    int loop_count = atoi(argv[3]);\n"
      << "    printf(\"\\n### begin to run %d\", loop_count);\n"
      << "    for (int i = 0; i < loop_count; i++) {\n"
      << "      uint64_t runBegin = GetTimeUs();\n"
      << "      " << config_->module_name() << "_Inference();\n"
      << "      uint64_t runEnd = GetTimeUs();\n"
      << "      uint64_t time = runEnd - runBegin;\n"
      << "      timeAvg += time;\n"
      << "    }\n"
      << "    float cunCost = (float)timeAvg / 1000.0f;\n"
      << "    printf(\"\\n###Run over, total time:\\t %5.5f ms.\\n\", cunCost);\n"
      << "    printf(\"\\n###Run over, predict per time:\\t %5.5f ms.\\n\", cunCost / loop_count);\n"
      << "  }\n";
  ofs << "  " << config_->module_name() << "_Inference();\n";
  CodeTestRelevantTile(ofs);
  ofs.close();
  return RET_OK;
}
}  // namespace mindspore::lite::micro
