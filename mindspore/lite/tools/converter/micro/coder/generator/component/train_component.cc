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

#include "coder/generator/component/train_component.h"
#include <string>
#include "nnacl/op_base.h"
#include "coder/utils/type_cast.h"

namespace mindspore::lite::micro {
void CodeMSModelSetTrainMode(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  std::vector<Tensor *> train_outputs = ctx->graph_train_outputs();
  std::vector<Tensor *> eval_outputs = ctx->graph_eval_outputs();
  auto train_outputs_size = train_outputs.size();
  auto eval_outputs_size = eval_outputs.size();

  auto array_tostring = [&ofs](Tensor *tensor, size_t index) {
    ofs << "    output_tensors[" << index << "] = malloc(sizeof(MicroTensor));\n";
    ofs << "    output_tensors[" << index << "]->type = " << EnumNameMSDataType(tensor->data_type()) << ";\n";
    ofs << "    output_tensors[" << index << "]->format = kMSFormatNHWC;\n";
    ofs << "    output_tensors[" << index << "]->ndim = " << tensor->shape().size() << ";\n";
    size_t shape_size = tensor->shape().size();
    ofs << "    output_tensors[" << index << "]->shape = "
        << "malloc(" << shape_size << " * sizeof(int64_t));\n";
    for (size_t i = 0; i < shape_size; i++) {
      ofs << "    output_tensors[" << index << "]->shape[" << i << "]= " << tensor->shape()[i] << ";\n";
    }
    ofs << "    output_tensors[" << index << "]->name = \"" << tensor->tensor_name() << "\";\n";
    ofs << "    output_tensors[" << index << "]->data = NULL;\n";
  };

  ofs << R"RAW(
MSStatus MSModelSetTrainMode(MSModelHandle model, bool train) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  micro_model->train_mode = train;
  MSTensorHandleArrayDestroy(micro_model->outputs);
)RAW";
  ofs << "  if (train) {\n"
      << "    MSTensorHandleArray model_outputs;\n"
      << "    model_outputs.handle_num = " << train_outputs_size << ";\n"
      << "    MicroTensor **output_tensors = malloc(" << train_outputs_size << " * sizeof(MicroTensor *));\n"
      << "    model_outputs.handle_list = (MSTensorHandle *)(output_tensors);\n"
      << "    micro_model->outputs = model_outputs;\n";
  for (size_t i = 0; i < train_outputs_size; ++i) {
    Tensor *output = train_outputs[i];
    array_tostring(output, i);
  }
  ofs << "  } else {\n"
      << "    MSTensorHandleArray model_outputs;\n"
      << "    model_outputs.handle_num = " << eval_outputs_size << ";\n"
      << "    MicroTensor **output_tensors = malloc(" << eval_outputs_size << " * sizeof(MicroTensor *));\n"
      << "    model_outputs.handle_list = (MSTensorHandle *)(output_tensors);\n"
      << "    micro_model->outputs = model_outputs;\n";
  for (size_t i = 0; i < eval_outputs_size; ++i) {
    Tensor *output = eval_outputs[i];
    array_tostring(output, i);
  }
  ofs << "  }\n"
      << "  return kMSStatusSuccess;\n"
         "}\n\n";
}

void CodeMSModelRunStep(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  auto inputs_size = ctx->graph_inputs().size();
  size_t train_outputs_size = ctx->graph_train_outputs().size();
  size_t eval_outputs_size = ctx->graph_eval_outputs().size();
  ofs << R"RAW(
MSStatus MSModelRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  bool train_mode = micro_model->train_mode;
)RAW";
  ofs << "  if (micro_model->inputs.handle_num != " << inputs_size << ") {\n"
      << "    return kMSStatusLiteParamInvalid;\n"
      << "  }\n"
      << "  const void *inputs_data_array[" << inputs_size << "];\n"
      << "  for (int i = 0; i < " << inputs_size << "; i++) {\n"
      << "    inputs_data_array[i] = ((MicroTensor *)(micro_model->inputs.handle_list[i]))->data;\n"
      << "  }\n"
      << "  SetInputs(inputs_data_array, " << inputs_size << ");\n"
      << "\n"
      << "  Execute(train_mode);\n\n"
      << "  // copy data to outputs handle\n"
      << "  if (train_mode) {\n"
      << "    if (micro_model->outputs.handle_num != " << train_outputs_size << ") {\n"
      << "      return kMSStatusLiteParamInvalid;\n"
      << "    }\n"
      << "    void *outputs_data_array[" << train_outputs_size << "];\n"
      << "    for (int i = 0; i < " << train_outputs_size << "; i++) {\n"
      << "      outputs_data_array[i] = MSTensorGetMutableData(micro_model->outputs.handle_list[i]);\n"
      << "    }\n"
      << "    if (CopyOutputsDataWithFlag(outputs_data_array, " << train_outputs_size << ", true) != RET_OK) {\n"
      << "      return kMSStatusLiteError;\n"
      << "    }\n"
      << "  } else {\n"
      << "    if (micro_model->outputs.handle_num != " << eval_outputs_size << ") {\n"
      << "      return kMSStatusLiteParamInvalid;\n"
      << "    }\n"
      << "    void *outputs_data_array[" << eval_outputs_size << "];\n"
      << "    for (int i = 0; i < " << eval_outputs_size << "; i++) {\n"
      << "      outputs_data_array[i] = MSTensorGetMutableData(micro_model->outputs.handle_list[i]);\n"
      << "    }\n"
      << "    if (CopyOutputsDataWithFlag(outputs_data_array, " << eval_outputs_size << ", false) != RET_OK) {\n"
      << "      return kMSStatusLiteError;\n"
      << "    }\n"
      << "  }\n"
      << "  return kMSStatusSuccess;\n"
      << "}\n\n";
}

void CodeMSModelExportWeight(std::ofstream &ofs) {
  ofs << R"RAW(
MSStatus MSModelExportWeight(MSModelHandle model, const char *export_path) {
  int ret = Export(export_path);
  return ret == RET_OK ? kMSStatusSuccess : kMSStatusLiteError;
})RAW";
  ofs << "\n\n";
}

void CodeCopyTrainOutputsState(std::ofstream &ofs) {
  ofs << "int CopyOutputsDataWithFlag(void **outputs, int num, bool train_mode);\n\n";
}

void CodeCopyTrainOutputsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  auto tensor_map = ctx->tensors_map();
  std::vector<Tensor *> train_outputs = ctx->graph_train_outputs();
  std::vector<Tensor *> eval_outputs = ctx->graph_eval_outputs();
  size_t train_outputs_size = train_outputs.size();
  size_t eval_outputs_size = eval_outputs.size();

  ofs << "int CopyOutputsDataWithFlag(void **outputs, int num, bool train_mode) {\n"
      << "  if (outputs == NULL) {\n"
      << "    return RET_ERROR;\n"
      << "  }\n"
      << "  if (train_mode) {\n"
      << "    if (num != " << train_outputs_size << ") {\n"
      << "      return RET_ERROR;\n"
      << "    }\n";
  for (size_t i = 0; i < train_outputs_size; ++i) {
    Tensor *output = train_outputs[i];
    MS_CHECK_PTR_IF_NULL(output);
    MS_CHECK_TRUE_RET_VOID(tensor_map.find(output) != tensor_map.end());
    ofs << "    memcpy(outputs[" << i << "], " << tensor_map[output] << ", " << output->Size() << ");\n";
  }
  ofs << "  } else {\n"
      << "    if (num != " << eval_outputs_size << ") {\n"
      << "      return RET_ERROR;\n"
      << "    }\n";
  for (size_t i = 0; i < eval_outputs_size; ++i) {
    Tensor *output = eval_outputs[i];
    MS_CHECK_PTR_IF_NULL(output);
    MS_CHECK_TRUE_RET_VOID(tensor_map.find(output) != tensor_map.end());
    ofs << "    memcpy(outputs[" << i << "], " << tensor_map[output] << ", " << output->Size() << ");\n";
  }
  ofs << "  }\n"
      << "  return RET_OK;\n"
         "}\n\n";
}
}  // namespace mindspore::lite::micro
