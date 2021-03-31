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
void CodeSessionCompileGraph(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator *config) {
  auto array_tostring = [&ofs](const std::vector<int> &array, const std::string &name) {
    size_t num = array.size();
    ofs << "  Vector<int> " << name << ";\n";
    ofs << "  " << name << ".resize(" << num << ");\n";
    for (size_t i = 0; i < num; ++i) {
      ofs << "  " << name << "[" << i << "] = " << array[i] << ";\n";
    }
  };
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  std::vector<Tensor *> outputs = ctx->graph_outputs();
  size_t inputs_size = inputs.size();
  size_t outputs_size = outputs.size();
  ofs << kNameSpaceMindSpore << " {\n";
  ofs << kNameSpaceLite << " {\n";
  ofs << "int LiteSession::CompileGraph(lite::Model *model) {\n";
  ofs << "  inputs_.resize(" << inputs_size << ");\n";
  for (size_t i = 0; i < inputs_size; ++i) {
    Tensor *input = inputs[i];
    std::string shape_i = "in_shape_" + std::to_string(i);
    array_tostring(input->shape(), shape_i);
    ofs << "  inputs_[" << i << "] = new (std::nothrow) MTensor(String(\"" << input->tensor_name() << "\"), "
        << EnumNameDataType(input->data_type()) << ", " << shape_i << ");\n";
    ofs << "  MS_ERROR_IF_NULL(inputs_[" << i << "]);\n";
  }
  ofs << "  outputs_.resize(" << outputs_size << ");\n";
  for (size_t i = 0; i < outputs_size; ++i) {
    Tensor *output = outputs[i];
    std::string shape_i = "out_shape_" + std::to_string(i);
    array_tostring(output->shape(), shape_i);
    ofs << "  outputs_[" << i << "] = new (std::nothrow) MTensor(String(\"" << output->tensor_name() << "\"), "
        << EnumNameDataType(output->data_type()) << ", " << shape_i << ");\n";
    ofs << "  MS_ERROR_IF_NULL(outputs_[" << i << "]);\n";
  }
  if (config->target() != kARM32M) {
    ofs << "  int ret = Init(model->buf, static_cast<MModel *>(model)->buf_size());\n"
           "  return ret;\n"
           "}\n\n";
    return;
  }
  ofs << "  return RET_OK;\n";
  ofs << "}\n\n";
}

void CodeCreateSessionImplement(std::ofstream &ofs, const Configurator *config) {
  ofs << "session::LiteSession *session::LiteSession::CreateSession(const lite::Context *context) {\n"
         "  auto *session = new (std::nothrow) lite::LiteSession();\n"
         "  MS_NULLPTR_IF_NULL(session);\n"
         "  int ret = session->InitRuntimeBuffer();\n"
         "  MS_NULLPTR_IF_ERROR(ret);\n";
  if (config->support_parallel()) {
    ofs << "  MS_NULLPTR_IF_NULL(context);\n"
           "  struct ThreadPool *thread_pool =\n"
           "    CreateThreadPool(context->thread_num_, "
           "context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_);\n"
           "  MS_NULLPTR_IF_NULL(thread_pool);\n"
           "  ret = SetThreadPool(thread_pool);\n"
           "  MS_NULLPTR_IF_ERROR(ret);\n";
  }
  ofs << "  return session;\n"
         "}\n\n";
  ofs << "session::LiteSession *session::LiteSession::CreateSession(const char *model_buf, size_t size,\n"
         "                                                          const lite::Context *context) {\n"
         "  session::LiteSession *session = CreateSession(context);\n"
         "  MS_NULLPTR_IF_NULL(session);\n"
         "  lite::Model *model = lite::Model::Import(model_buf, size);\n"
         "  int ret = session->CompileGraph(model);\n"
         "  MS_NULLPTR_IF_ERROR(ret);\n"
         "  delete model;\n"
         "  return session;\n"
         "}\n"
         "}  // namespace mindspore\n\n";
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
    ofs << "\t" << ctx->input_name() << i << " = inputs[" << i << "];\n";
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
  ofs << "  " << ctx->buffer_name() << " = buffer;\n"
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

void CodeInferenceState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * net inference function\n"
      << "  **/\n"
      << "void "
      << "Inference();\n\n";
}

}  // namespace mindspore::lite::micro
