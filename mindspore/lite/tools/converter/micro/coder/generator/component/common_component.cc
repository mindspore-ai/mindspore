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
#include "coder/generator/component/component.h"
#include "coder/utils/type_cast.h"
#include "coder/utils/coder_utils.h"
#include "coder/log.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"

namespace mindspore::lite::micro {
void CodeMSModelCreate(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  auto array_tostring = [&ofs](Tensor *tensor, const std::string &prefix, size_t index) {
    ofs << kAlignedString << prefix << "_tensors[" << index << "] = malloc(sizeof(MicroTensor));\n";
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->type = " << EnumNameMSDataType(tensor->data_type())
        << ";\n";
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->format = kMSFormatNHWC;\n";
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->ndim = " << tensor->shape().size() << ";\n";
    size_t shape_size = tensor->shape().size();
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->shape = "
        << "malloc(" << shape_size << " * sizeof(int64_t));\n";
    for (size_t i = 0; i < shape_size; i++) {
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->shape[" << i << "]= " << tensor->shape()[i]
          << ";\n";
    }
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->name = \"" << tensor->tensor_name() << "\";\n";
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->data = NULL;\n";
  };
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  std::vector<Tensor *> outputs = ctx->graph_outputs();

  size_t inputs_size = inputs.size();
  ofs << "  MSTensorHandleArray model_inputs;\n";
  ofs << "  model_inputs.handle_num = " << inputs_size << ";\n";
  ofs << "  MicroTensor **input_tensors = malloc(" << inputs_size << " * sizeof(MicroTensor *));\n";
  ofs << "  model_inputs.handle_list = (MSTensorHandle *)(input_tensors);\n";
  ofs << "  micro_model->inputs = model_inputs;\n";
  for (size_t i = 0; i < inputs_size; ++i) {
    Tensor *input = inputs[i];
    array_tostring(input, "input", i);
  }
  size_t outputs_size = outputs.size();
  ofs << "  MSTensorHandleArray model_outputs;\n";
  ofs << "  model_outputs.handle_num = " << outputs_size << ";\n";
  ofs << "  MicroTensor **output_tensors = malloc(" << outputs_size << " * sizeof(MicroTensor *));\n";
  ofs << "  model_outputs.handle_list = (MSTensorHandle *)(output_tensors);\n";
  ofs << "  micro_model->outputs = model_outputs;\n";
  for (size_t i = 0; i < outputs_size; ++i) {
    Tensor *output = outputs[i];
    array_tostring(output, "output", i);
  }
  ofs << "  return (MSModelHandle)micro_model;\n";
  ofs << "}\n\n";
}

void CodeMSModelBuild(std::ofstream &ofs, const Configurator *config) {
  ofs
    << "MSStatus MSModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type,\n"
       "                      const MSContextHandle model_context) {\n"
       "  if (model_type != kMSModelTypeMindIR) {\n"
       "    return kMSStatusLiteNotSupport;\n"
       "  }\n";
  ofs << "  int ret = RET_OK;\n";
  if (config->target() != kARM32M) {
    ofs << "  ret = Init((void*)model_data, data_size);\n";
  }
  if (config->support_parallel()) {
    ofs << "  MicroContext *micro_context = (MicroContext *)model_context;\n"
           "  if (micro_context == NULL) {\n"
           "      return RET_ERROR;"
           "  }\n"
           "  ret = CreateThreadPool(micro_context->thread_num_);\n"
           "  if(ret != RET_OK) {\n"
           "     return ret;\n"
           "  }\n"
           "  ret = SetCoreAffinity(micro_context->affinity_mode);\n";
  }
  ofs << "  return ret;\n";
  ofs << "}\n";
}

void CodeMSModelDestory(std::ofstream &ofs, const Configurator *config) {
  ofs << "void MSModelDestroy(MSModelHandle *model) {\n"
         "  if (model) {\n"
         "    MicroModel *micro_model = (MicroModel *)*model;\n"
         "    if (micro_model->runtime_buffer) {\n"
         "      free(micro_model->runtime_buffer);\n"
         "      micro_model->runtime_buffer = NULL;\n"
         "    }\n"
         "    MSTensorHandleArrayDestroy(micro_model->inputs);\n"
         "    MSTensorHandleArrayDestroy(micro_model->outputs);\n"
         "    free(*model);\n"
         "    *model = NULL;\n"
         "  }\n";
  if (config->support_parallel()) {
    ofs << "  ClearThreadPool();\n";
  }
  ofs << "}\n";
}

void CodeCopyOutputsState(std::ofstream &ofs) { ofs << "int CopyOutputsData(void **outputs, int num);\n\n"; }

void CodeCopyOutputsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  auto tensor_map = ctx->tensors_map();
  std::vector<Tensor *> outputs = ctx->graph_outputs();
  size_t outputs_size = outputs.size();

  ofs << "int CopyOutputsData(void **outputs, int num) {\n"
         "  if (outputs == NULL) {\n"
         "    return RET_ERROR;\n"
         "  }\n"
      << "  if (num != " << outputs_size << ") {\n"
      << "    return RET_ERROR;\n"
         "  }\n";
  for (size_t i = 0; i < outputs_size; ++i) {
    Tensor *output = outputs[i];
    MS_CHECK_PTR_IF_NULL(output);
    ofs << "  memcpy(outputs[" << i << "], " << tensor_map[output] << ", " << output->Size() << ");\n";
  }
  ofs << "  return RET_OK;\n"
         "}\n\n";
}

void CodeGlobalCodeBlocks(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  for (const auto &block : ctx->global_code_blocks()) {
    ofs << block << "\n";
  }
}

void CodeInputState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * set input tensors\n"
      << "  * @param inputs, the input data ptr's array of the model, the tensors' count of input may be greater than "
         "one.\n"
      << "  * @param num, the input data's number of the model.\n"
      << "  **/\n"
      << "int "
      << "SetInputs(const void **inputs, int num);\n\n";
}

void CodeInputImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  // input tensors
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    ofs << "static const unsigned char *" << ctx->input_name() + std::to_string(i) << " = 0;\n";
  }
  size_t size = inputs.size();
  ofs << "int "
      << "SetInputs(const void **inputs, int num) {\n"
      << "  if (inputs == NULL) {\n"
         "    return RET_ERROR;\n"
         "  }\n"
      << "  if (num !=" << size << ") {\n"
      << "    return RET_ERROR;\n"
         "  }\n";
  for (size_t i = 0; i < size; ++i) {
    ofs << "\t" << ctx->input_name() << i << " = (unsigned char *)inputs[" << i << "];\n";
  }
  ofs << "  return RET_OK;\n}\n";
}

void CodeGraphQuantArgsState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * get input and output QuantArgs of the model \n"
      << "  **/\n"
      << "GraphQuantArgs "
      << "GetInOutQuantArgs();\n\n";
}

void CodeGraphQuantArgsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  std::vector<Tensor *> graph_inputs = ctx->graph_inputs();
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "graph input tensors' number is 0";
    return;
  }
  Tensor *in_tensor = graph_inputs.at(kInputIndex);
  MS_CHECK_PTR_IF_NULL(in_tensor);
  std::vector<Tensor *> graph_outputs = ctx->graph_outputs();
  if (graph_outputs.empty()) {
    MS_LOG(ERROR) << "graph output tensors' number is 0";
    return;
  }
  Tensor *out_tensor = graph_outputs.at(kOutputIndex);
  MS_CHECK_PTR_IF_NULL(out_tensor);
  std::vector<LiteQuantParam> in_quant_args = in_tensor->quant_params();
  std::vector<LiteQuantParam> out_quant_args = out_tensor->quant_params();
  if (in_quant_args.empty() || out_quant_args.empty()) {
    MS_LOG(ERROR) << "code model quant args failed";
    return;
  }
  ofs << "GraphQuantArgs "
      << "GetInOutQuantArgs() {\n"
      << "\t\tGraphQuantArgs quan_args = { " << in_quant_args.at(0).scale << ", " << out_quant_args.at(0).scale << ", "
      << in_quant_args.at(0).zeroPoint << ", " << out_quant_args.at(0).zeroPoint << "};\n"
      << "\t\treturn quan_args;\n"
      << "}\n";
}

void CodeManageResourceState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * get the memory space size of the inference.\n"
      << "  **/\n"
      << "int "
      << "GetBufferSize();\n";

  ofs << "/**\n"
      << "  * set the memory space for the inference\n"
      << "  **/\n"
      << "int "
      << "SetBuffer(void *buffer);\n\n";

  ofs << "/**\n"
      << "  * free the memory of packed weights, and set the membuf buffer and input address to NULL\n"
      << "  **/\n"
      << "void "
      << "FreeResource();\n";
}

void CodeInitResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  ofs << "int "
      << "GetBufferSize() {\n"
      << "  return " << ctx->total_buffer_size() << ";\n"
      << "}\n";
  ofs << "int "
      << "SetBuffer( void *buffer) {\n";
  ofs << "  if (buffer == NULL) {\n"
         "    return RET_ERROR;\n"
         "  }\n";
  ofs << "  " << ctx->buffer_name() << " = (unsigned char *)buffer;\n"
      << "  return RET_OK;\n"
         "}\n";
}

void CodeFreeResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  ofs << "void "
      << "FreeResource() {\n";
  ofs << "  " << ctx->buffer_name() << "= NULL;\n";
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  size_t size = inputs.size();
  for (size_t i = 0; i < size; ++i) {
    ofs << "  " << ctx->input_name() + std::to_string(i) << " = NULL;\n";
  }
  ofs << "  void **allocated[] = {\n";
  size_t num = 0;
  for (const auto &item : ctx->tensors_map()) {
    Tensor *tensor = item.first;
    std::string name = item.second;
    if (tensor->data() != nullptr && !(CheckConstantTensor(tensor))) {
      ofs << "    (void**)&" << name << ",\n";
      num++;
    }
  }
  ofs << "\n  };\n";
  ofs << "  for (int i = 0; i < " << num << "; ++i) {\n"
      << "    *(allocated[i]) = NULL;\n"
      << "  }\n";
  ofs << "  if (" << ctx->weight_name() << " != NULL) {\n";
  ofs << "    free(" << ctx->weight_name() << ");\n";
  ofs << "    " << ctx->weight_name() << " = NULL;\n  }\n";
  ofs << "}\n";
}

void CodeInferenceState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * net inference function\n"
      << "  **/\n"
      << "void "
      << "Inference();\n\n";
}
}  // namespace mindspore::lite::micro
