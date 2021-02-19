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

#include "coder/generator/component/benchmark_component.h"
#include <memory>
#include "coder/generator/component/const_blocks/license.h"
#include "coder/log.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"

namespace mindspore::lite::micro {
constexpr int kWarmUp = 3;
void CodeBenchmarkHeader(std::ofstream &ofs, const std::string &header) {
  ofs << g_hwLicense;
  ofs << "#include <stdio.h>\n"
         "#include <string.h>\n"
         "#include <stdlib.h>\n"
         "#include <stdint.h>\n"
         "#include \"microtensor.h\"\n"
         "#include \"load_input.h\"\n"
         "#include \"debug_utils.h\"\n";
  ofs << "#include \"" << header << "\"\n";
}

void CodeBenchmarkUsage(std::ofstream &ofs) {
  ofs << "void usage() {\n"
         "  printf(\n"
         "    \"-- mindspore micro params usage:\\n\"\n"
         "    \"args[0]: executable file\\n\"\n"
         "    \"args[1]: inputs binary file\\n\"\n"
         "    \"args[2]: model weight binary file\\n\"\n"
         "    \"args[3]: loop count for performance test\\n\"\n"
         "    \"args[4]: runtime thread num\\n\"\n"
         "    \"args[5]: runtime thread bind mode\\n\\n\");\n"
         "}\n\n";
}

void CodeBenchmarkWarmup(std::ofstream &ofs, const std::string &module_name) {
  ofs << "// the default number of warm-ups is 3\n"
      << "void " << module_name << "_WarmUp() {\n"
      << "  for (int i = 0; i < " << kWarmUp << "; ++i) {\n"
      << "    " << module_name << "_Inference();\n"
      << "  }\n"
      << "}\n";
}

void CodeBenchmarkSetInputs(std::ofstream &ofs, const std::string &module_name,
                            const std::unique_ptr<CoderContext> &ctx) {
  ofs << "int main(int argc, char **argv) {\n"
         "  if (argc < 2) {\n"
         "    MICRO_ERROR(\"input command is invalid\\n\");\n"
         "    usage();\n"
         "    return RET_ERROR;\n"
         "  }\n";
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  size_t inputs_num = inputs.size();
  ofs << "  // input shape: ";
  std::for_each(inputs.begin(), inputs.end(), [&](Tensor *t) {
    ofs << "[ ";
    for (int i : t->shape()) {
      ofs << i << ", ";
    }
    ofs << "], ";
  });
  ofs << "\n";
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
  ofs << "  ret = " << module_name << "_SetInputs((const void **)inputs_binbuf, " << inputs_num
      << ");\n"
         "  if (ret != RET_OK) {\n"
         "    MICRO_ERROR(\"set inputs failed\");\n"
         "    return RET_ERROR;\n"
         "  }\n";
}

void CodeBenchmarkSetBuffer(std::ofstream &ofs, const std::string &module_name) {
  ofs << "  int total_buffer_size = " << module_name << "_GetBufferSize();\n";
  ofs << "  void *buffer = malloc(total_buffer_size);\n";
  ofs << "  if (buffer == NULL ){\n"
         "     MICRO_ERROR(\"malloc memory buffer failed\");\n"
         "     return RET_ERROR;\n"
         "  }\n";
  ofs << "  ret = " << module_name
      << "_SetBuffer(buffer);\n"
         "  if (ret != RET_OK) {\n"
         "    MICRO_ERROR(\"set inputs failed\");\n"
         "    return RET_ERROR;"
         "  }\n";
}

void CodeBenchmarkInitWeight(std::ofstream &ofs, const std::string &module_name) {
  ofs << "  int weight_size = 0;\n"
         "  void *weight_buffer = ReadInputData(argv[2], &weight_size); \n"
         "  if("
      << module_name
      << "_Init(weight_buffer, weight_size) != RET_OK) {\n"
         "    MICRO_ERROR(\"model init failed\");\n"
         "    "
      << module_name
      << "_FreeResource();\n"
         "    return RET_ERROR;\n"
         "  }\n"
         "  free(weight_buffer);\n"
         "  weight_buffer = NULL;\n";
}

void CodeBenchmarkConfigThread(std::ofstream &ofs) {
  ofs << "  int thread_num = 4;\n"
         "  BindMode bind_mode = NO_BIND_MODE;\n"
         "  if (argc >= 6) {\n"
         "    thread_num = atoi(argv[4]);\n"
         "    bind_mode = atoi(argv[5]);\n"
         "  }\n"
         "  ret = ConfigThreadPool(THREAD_POOL_DEFAULT, thread_num, bind_mode);\n"
         "  if (ret != 0) {\n"
         "    MICRO_ERROR(\"create thread pool failed\");\n"
         "  }\n";
}

void CodeBenchmarkInference(std::ofstream &ofs, const std::string &module_name) {
  ofs << "  if (argc >= 4) {\n"
      << "    " << module_name << "_WarmUp();\n"
      << "    uint64_t timeAvg = 0;\n"
      << "    int loop_count = atoi(argv[3]);\n"
      << "    printf(\"======Inference Start======\\n\");\n"
      << "    printf(\"cycles: %d\", loop_count);\n"
      << "    for (int i = 0; i < loop_count; i++) {\n"
      << "      uint64_t runBegin = GetTimeUs();\n"
      << "      " << module_name << "_Inference();\n"
      << "      uint64_t runEnd = GetTimeUs();\n"
      << "      uint64_t time = runEnd - runBegin;\n"
      << "      timeAvg += time;\n"
      << "    }\n"
      << "    float cunCost = (float)timeAvg / 1000.0f;\n"
      << "    printf(\"=======Inference End=======\\n\");\n"
         "    printf(\"total time:\\t %.5fms, per time: \\t %.5fms\\n\", cunCost, cunCost/loop_count);\n"
      << "  }\n";
  ofs << "  " << module_name << "_Inference();\n";
}

void CodeBenchmarkPrintOutputs(std::ofstream &ofs, const std::string &module_name) {
  ofs << "  // print model outputs \n";
  ofs << "  const MicroTensorList *outs = " << module_name << "_GetOutputs();\n";
  ofs << "  for (int i = 0; i < outs->num; ++i) {\n"
         "      MicroTensor *tensor = outs->tensor + i;\n"
         "      PrintTensorData(tensor);\n"
         "  }\n";
  ofs << "  printf(\"" << module_name << " inference success.\\n\");\n";
  ofs << "  free(buffer);\n";
}

/**
 * 1. free malloc memory buffer
 * 2. set input and buffer to NULL, and free packed weight memory
 * 3. free input binary memory
 */
void CodeBenchmarkFreeResourse(std::ofstream &ofs, const std::string &module_name, size_t inputs_num) {
  ofs << "  free(buffer);\n";
  ofs << "  " << module_name << "_FreeResource();\n";
  ofs << "  for (int i = 0; i < " << inputs_num << "; ++i) {\n";
  ofs << "    free(inputs_binbuf[i]);\n"
         "  }\n"
         "  return RET_OK;\n"
         "}\n\n";
}

}  // namespace mindspore::lite::micro
