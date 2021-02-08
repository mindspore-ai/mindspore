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

#include "coder/generator/component/common_component.h"
#include <memory>
#include "coder/generator/component/const_blocks/license.h"
#include "coder/utils/type_cast.h"
#include "coder/log.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"

namespace mindspore::lite::micro {
void CodeSourceFileInclude(std::ofstream &ofs, const std::string &weight_file, const std::string &header) {
  ofs << g_hwLicense << "#include \"microtensor.h\"\n"
      << "#include \"" << weight_file << "\"\n"
      << "#include \"" << header << "\"\n";
}

void CodeInputAndOutputState(std::ofstream &ofs, const std::string &module_name) {
  ofs << "/**\n"
      << "  * set input tensors\n"
      << "  * @param inputs, the input data ptr's array of the model, the tensors' count of input may be greater than "
         "one.\n"
      << "  * @param num, the input data's number of the model.\n"
      << "  **/\n"
      << "int " << module_name << "_SetInputs(const void **inputs, int num);\n\n";

  ofs << "/**\n"
      << "  * get output tensor of the model \n"
      << "  **/\n"
      << "const MicroTensorList *" << module_name << "_GetOutputs();\n\n";
}

void PrintMicroTensors(std::ofstream &ofs, std::vector<Tensor *> tensors, const std::string &name,
                       const std::map<Tensor *, std::string> &tensors_map) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    Tensor *tensor = tensors[i];
    auto item = tensors_map.find(tensor);
    if (item == tensors_map.end()) {
      MS_LOG(ERROR) << "nonexistent tensor";
      break;
    }
    ofs << "  static int dim[] = {";
    for (size_t j = 0; j < tensor->shape().size(); ++j) {
      ofs << tensor->shape()[j] << ", ";
    }
    ofs << "};\n"
        << "  " << name << "[" << i << "].ndim = " << tensor->shape().size() << ";\n"
        << "  " << name << "[" << i << "].dim = dim;\n"
        << "  " << name << "[" << i << "].type = " << EnumMicroTensorDataType(tensor->data_type()) << ";\n"
        << "  " << name << "[" << i << "].format = " << std::to_string(tensor->format()) << ";\n"
        << "  " << name << "[" << i << "].data =" << item->second << ";\n";
  }
}

void CodeInputAndOutputImplement(std::ofstream &ofs, const std::string &module_name,
                                 const std::unique_ptr<CoderContext> &ctx) {
  // input tensors
  ofs << "\n// input tensors\n";
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    ofs << "static const unsigned char *" << ctx->input_name() + std::to_string(i) << " = 0;\n";
  }
  size_t size = inputs.size();
  ofs << "int " << module_name << "_SetInputs(const void **inputs, int num) {\n"
      << "  if (inputs == NULL) {\n"
         "    return RET_ERROR;\n"
         "  }\n"
      << "  if (num !=" << size << ") {\n"
      << "    return RET_ERROR;\n"
         "  }\n";
  for (size_t i = 0; i < size; ++i) {
    ofs << "\t" << ctx->input_name() + std::to_string(i) << " = inputs[" << i << "];\n";
  }
  ofs << "  return RET_OK;\n}\n";

  // output tensors
  ofs << "\n// output tensors\n";
  std::vector<Tensor *> outputs = ctx->graph_outputs();
  size_t output_num = outputs.size();
  std::string output_name = ctx->output_name();

  ofs << "const MicroTensorList* " << module_name << "_GetOutputs() {\n"
      << "  static MicroTensor " << output_name << "[" << output_num << "] ;\n";

  PrintMicroTensors(ofs, outputs, output_name, ctx->tensors_map());
  ofs << "  static MicroTensorList  " << module_name << "_TensorArray;\n"
      << "  " << module_name << "_TensorArray.num = " << output_num << ";\n"
      << "  " << module_name << "_TensorArray.tensor = &" << output_name << "[0];\n"
      << "  return  &" << module_name << "_TensorArray; \n}\n";
}

void CodeGraphQuantArgsState(std::ofstream &ofs, const std::string &module_name) {
  ofs << "/**\n"
      << "  * get input and output QuantArgs of the model \n"
      << "  **/\n"
      << "GraphQuantArgs " << module_name << "_GetInOutQuantArgs();\n\n";
}

void CodeGraphQuantArgsImplement(std::ofstream &ofs, const std::string &module_name,
                                 const std::unique_ptr<CoderContext> &ctx) {
  std::vector<Tensor *> graph_inputs = ctx->graph_inputs();
  Tensor *in_tensor = graph_inputs.at(kInputIndex);
  MS_CHECK_PTR_IF_NULL(in_tensor);
  std::vector<Tensor *> graph_outputs = ctx->graph_outputs();
  Tensor *out_tensor = graph_outputs.at(kOutputIndex);
  MS_CHECK_PTR_IF_NULL(out_tensor);
  std::vector<QuantArg> in_quant_args = in_tensor->quant_params();
  std::vector<QuantArg> out_quant_args = out_tensor->quant_params();
  if (graph_inputs.empty() || graph_outputs.empty() || in_quant_args.empty() || out_quant_args.empty()) {
    MS_LOG(ERROR) << "code model quant args failed";
    return;
  }
  ofs << "GraphQuantArgs " << module_name << "_GetInOutQuantArgs() {\n"
      << "\t\tGraphQuantArgs quan_args = { " << in_quant_args.at(0).scale << ", " << out_quant_args.at(0).scale << ", "
      << in_quant_args.at(0).zeroPoint << ", " << out_quant_args.at(0).zeroPoint << "};\n"
      << "\t\treturn quan_args;\n"
      << "}\n";
}

void CodeInitWeightState(std::ofstream &ofs, const std::string &module_name) {
  ofs << "/**\n"
      << "  * @param weight_buffer, the address of the weight binary file\n"
      << "  * @param weight_size, the size of the model file in bytes\n"
      << "  **/\n"
      << "int " << module_name << "_Init(void *weight_buffer, int weight_size);\n\n";
}

void CodeManageResourceState(std::ofstream &ofs, const std::string &module_name) {
  ofs << "/**\n"
      << "  * get the memory space size of the inference.\n"
      << "  **/\n"
      << "int " << module_name << "_GetBufferSize();\n";

  ofs << "/**\n"
      << "  * set the memory space for the inference\n"
      << "  **/\n"
      << "int " << module_name << "_SetBuffer(void *buffer);\n\n";

  ofs << "/**\n"
      << "  * free the memory of packed weights, and set the membuf buffer and input address to NULL\n"
      << "  **/\n"
      << "void " << module_name << "_FreeResource();\n";
}

void CodeInitResourceImplement(std::ofstream &ofs, const std::string &module_name,
                               const std::unique_ptr<CoderContext> &ctx) {
  ofs << "int " << module_name << "deconv_GetBufferSize() {\n"
      << "  return " << ctx->total_buffer_size() << ";\n"
      << "}\n";
  ofs << "int " << module_name << "_SetBuffer( void *buffer) {\n";
  ofs << "  if (buffer == NULL) {\n"
         "    MICRO_ERROR(\"memory buffer is NULL\");\n"
         "    return RET_ERROR;\n"
         "  }\n";
  ofs << "  " << ctx->buffer_name()
      << " = buffer;\n"
         "  return RET_OK;\n"
         "}\n";
}

void CodeFreeResourceImplement(std::ofstream &ofs, const std::string &module_name,
                               const std::unique_ptr<CoderContext> &ctx) {
  ofs << "void " << module_name << "_FreeResource() {\n";
  ofs << "  " << ctx->buffer_name() << "= NULL;\n";
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  size_t size = inputs.size();
  for (size_t i = 0; i < size; ++i) {
    ofs << "  " << ctx->input_name() + std::to_string(i) << " = NULL;\n";
  }
  ofs << "  void *allocated[] = {";
  size_t num = 0;
  for (const auto &item : ctx->tensors_map()) {
    Tensor *tensor = item.first;
    std::string name = item.second;
    if (tensor->data_c() != nullptr && tensor->category() != Tensor::Category::CONST_TENSOR) {
      ofs << name << ", ";
      num++;
    }
  }
  ofs << "  };\n";
  ofs << "  for (int i = 0; i < " << num << "; ++i) {\n"
      << "    free(allocated[i]);\n"
      << "    allocated[i] = NULL;\n"
      << "  }\n";
  ofs << "}\n";
}

void CodeInferenceState(std::ofstream &ofs, const std::string &module_name) {
  ofs << "/**\n"
      << "  * net inference function\n"
      << "  **/\n"
      << "void " << module_name << "_Inference();\n\n";
}

}  // namespace mindspore::lite::micro
