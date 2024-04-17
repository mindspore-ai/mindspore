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

#include "coder/generator/component/common_component.h"
#include <memory>
#include "coder/generator/component/component.h"
#include "coder/utils/type_cast.h"
#include "coder/utils/coder_utils.h"
#include "coder/log.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "include/c_api/model_c.h"
#include "coder/generator/component/const_blocks/license.h"
#include "tools/common/string_util.h"

namespace mindspore::lite::micro {
const char handle_array_destroy_state[] = R"RAW(
void MSTensorHandleArrayDestroy(MSTensorHandleArray inputs);
)RAW";

const char handle_array_destroy[] = R"RAW(
void MSTensorHandleArrayDestroy(MSTensorHandleArray inputs) {
 if (inputs.handle_list == NULL) {
   return;
 }
 for (size_t i = 0; i < inputs.handle_num; i++) {
   MicroTensor *micro_tensor = inputs.handle_list[i];
   if (micro_tensor == NULL) {
     continue;
   }
   if (micro_tensor->data != NULL && micro_tensor->owned) {
     free(micro_tensor->data);
     micro_tensor->data = NULL;
     micro_tensor->owned = false;
   }
   if (micro_tensor->shape) {
     free(micro_tensor->shape);
     micro_tensor->shape = NULL;
   }
   free(micro_tensor);
   micro_tensor = NULL;
 }
 free(inputs.handle_list);
 inputs.handle_list = NULL;
}

)RAW";

const char cortex_set_workspace[] = R"RAW(
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return;
  }
  if (workspace_size < MSModelCalcWorkspaceSize(model)) {
    return;
  }
  if (micro_model->inputs.handle_num != GRAPH_INPUTS_SIZE) {
    return;
  }
  if (micro_model->outputs.handle_num != GRAPH_OUTPUTS_SIZE) {
    return;
  }

)RAW";

const char micro_model_build_state[] = R"RAW(
typedef MSStatus (*ModelBuild)(MSModelHandle model, const void *model_data,
                               size_t data_size,
                               const MSContextHandle model_context);
)RAW";

const char micro_model_build_implement[] = R"RAW(
MSStatus MSModelBuild(MSModelHandle model, const void *model_data,
                      size_t data_size, MSModelType model_type,
                      const MSContextHandle model_context) {
  if (model_type != kMSModelTypeMindIR) {
    return kMSStatusLiteNotSupport;
  }
  if (model == NULL) {
    return kMSStatusLiteParamInvalid;
  }
)RAW";

const char micro_model_predict_state[] = R"RAW(
typedef MSStatus (*ModelPredict)(MSModelHandle model,
                                 const MSTensorHandleArray inputs,
                                 MSTensorHandleArray *outputs,
                                 const MSKernelCallBackC before,
                                 const MSKernelCallBackC after);
)RAW";

const char free_resource_state[] = R"RAW(
typedef void (*FreeResource)();
)RAW";

void CodeMSModelCalcWorkspaceSize(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx,
                                  const Configurator &config) {
  if (config.target() == kCortex_M) {
    ofs << "size_t MSModelCalcWorkspaceSize(MSModelHandle model) {\n"
        << "  MicroModel *micro_model = (MicroModel *)model;\n"
        << "  if (micro_model == NULL) {\n"
        << "    return 0;\n"
        << "  }\n"
        << "  if (micro_model->calc_work_space == NULL) {\n"
        << "    return 0;\n"
        << "  }\n"
        << "  return micro_model->calc_work_space(model);\n"
        << "}\n";
  } else {
    ofs << "size_t MSModelCalcWorkspaceSize(MSModelHandle model) {\n  return 0;\n}\n";
  }
  ofs << "\n";
}

void CodeCortexCalcWorkspaceSize(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  ofs << "size_t MSModelCalcWorkspaceSize" << ctx->GetCurModelIndex() << "(MSModelHandle model) {\n"
      << "size_t shape_size = 0;\n";
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    ofs << "  shape_size += " << inputs[i]->shape().size() << " * sizeof(int64_t);\n";
  }
  std::vector<Tensor *> outputs = ctx->graph_outputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    ofs << "  shape_size += " << outputs[i]->shape().size() << " * sizeof(int64_t);\n";
  }
  ofs << "  return UP_ROUND(GetBufferSize" << ctx->GetCurModelIndex()
      << "(),4) + UP_ROUND(WEIGHT_BUF_SIZE,4) + shape_size + "
      << "(UP_ROUND(sizeof(MicroTensor),4) + UP_ROUND(sizeof(MicroTensor *),4)) * "
      << (ctx->graph_inputs().size() + ctx->graph_outputs().size()) << ";\n}\n";
}

void CodeMSModelSetWorkspace(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  ofs << "void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size) {";
  if (config.target() == kCortex_M) {
    ofs << "  MicroModel *micro_model = (MicroModel *)model;\n"
        << "  if (micro_model == NULL) {\n"
        << "    return;\n"
        << "  }\n"
        << "  if (micro_model->set_work_space == NULL) {\n"
        << "    return;\n"
        << "  }\n"
        << "  micro_model->set_work_space(model, workspace, workspace_size);\n";
  }
  ofs << "}\n\n";
}

void CodeCortexSetWorkspace(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  ofs << "void MSModelSetWorkspace" << ctx->GetCurModelIndex()
      << "(MSModelHandle model, void *workspace, size_t workspace_size) {\n";
  ofs << cortex_set_workspace;
  ofs << "  micro_model->runtime_buffer = workspace;\n"
         "  int buffer_size = GetBufferSize"
      << ctx->GetCurModelIndex()
      << "();\n"
         "  char* buf = workspace;\n"
         "  SetBuffer"
      << ctx->GetCurModelIndex()
      << "(buf);\n"
         "  buffer_size = UP_ROUND(buffer_size, 4);\n";
  ofs << "  " << ctx->weight_name() << " = (uint8_t *)&buf[buffer_size];\n";
  ofs << R"RAW(
  buffer_size += WEIGHT_BUF_SIZE;
  buffer_size = UP_ROUND(buffer_size,4);

  micro_model->inputs.handle_list = (MSTensorHandle *)&buf[buffer_size];
  buffer_size +=  GRAPH_INPUTS_SIZE * sizeof(MicroTensor *);
  buffer_size = UP_ROUND(buffer_size,4);
  MicroTensor **input_tensors = (MicroTensor **)micro_model->inputs.handle_list;

  micro_model->outputs.handle_list = (MSTensorHandle *)&buf[buffer_size];
  buffer_size +=  GRAPH_OUTPUTS_SIZE * sizeof(MicroTensor *);
  buffer_size = UP_ROUND(buffer_size,4);
  MicroTensor **output_tensors = (MicroTensor **)micro_model->outputs.handle_list;
)RAW";
  ofs << "  int i;\n"
      << "  for (i = 0; i < GRAPH_INPUTS_SIZE; i++) {\n";
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    ofs << "    input_tensors[i] = (MicroTensor *)&buf[buffer_size];\n"
        << "    buffer_size += sizeof(MicroTensor);\n"
        << "    buffer_size = UP_ROUND(buffer_size,4);\n";
    ofs << "    input_tensors[i]->shape = (int64_t *)&buf[buffer_size];\n"
        << "    buffer_size += " << inputs[i]->shape().size() * sizeof(int64_t) << ";\n"
        << "    buffer_size = UP_ROUND(buffer_size,4);\n";
  }
  ofs << "  }\n";

  ofs << "  for (i = 0; i < GRAPH_OUTPUTS_SIZE; i++) {\n";
  std::vector<Tensor *> outputs = ctx->graph_outputs();
  for (size_t i = 0; i < outputs.size(); ++i) {
    ofs << "    output_tensors[i] = (MicroTensor *)&buf[buffer_size];\n"
        << "    buffer_size += sizeof(MicroTensor);\n"
        << "    buffer_size = UP_ROUND(buffer_size,4);\n";
    ofs << "    output_tensors[i]->shape = (int64_t *)&buf[buffer_size];\n"
        << "    buffer_size += " << outputs[i]->shape().size() * sizeof(int64_t) << ";\n"
        << "    buffer_size = UP_ROUND(buffer_size,4);\n";
  }
  ofs << "  }\n";
  ofs << "  if (buffer_size > workspace_size) {\n"
      << "    micro_model->runtime_buffer = NULL;\n"
      << "    SetBuffer" << ctx->GetCurModelIndex() << "(NULL);\n"
      << "    return;\n"
      << "  }\n";
  auto array_tostring = [&ofs](Tensor *tensor, const std::string &prefix, size_t index) {
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->type = " << EnumNameMSDataType(tensor->data_type())
        << ";\n";
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->format = kMSFormatNHWC;\n";
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->ndim = " << tensor->shape().size() << ";\n";
    size_t shape_size = tensor->shape().size();
    for (size_t i = 0; i < shape_size; i++) {
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->shape[" << i << "]= " << tensor->shape()[i]
          << ";\n";
    }
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->name = \"" << tensor->tensor_name() << "\";\n";
    ofs << kAlignedString << prefix << "_tensors[" << index << "]->data = NULL;\n";
  };
  for (size_t i = 0; i < inputs.size(); ++i) {
    array_tostring(inputs[i], "input", i);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    array_tostring(outputs[i], "output", i);
  }
  ofs << "}\n";
}

void CodeMSModelCreateDefault(std::ofstream &ofs) { ofs << "MSModelHandle MSModelCreate() { return model0; }\n"; }

void CodeMSModelCreate(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  if (config.target() != kCortex_M) {
    ofs << "MSStatus MSModelCreate" << ctx->GetCurModelIndex() << "(MicroModel *micro_model) {";
    ofs << R"RAW(
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
)RAW";
    if (!config.dynamic_shape()) {
      ofs << "void *runtime_buffer = GlobalMemory();\n"
          << "if (runtime_buffer == NULL) {\n"
          << "    return kMSStatusLiteNullptr;\n"
          << "  }\n"
          << "  micro_model->runtime_buffer = runtime_buffer;\n";
      ofs << "  int ret = SetBuffer" << ctx->GetCurModelIndex() << "(((MemBlock *)runtime_buffer)->addr);\n"
          << "  if (ret != kMSStatusSuccess) {\n"
          << "    return kMSStatusLiteMemoryFailed;\n"
          << "  }\n\n";
    } else {
      ofs << "  micro_model->runtime_buffer = NULL;\n";
    }
    if (config.code_mode() == CodeMode::Inference) {
      ofs << "  micro_model->train_mode = false;\n";
    } else if (config.code_mode() == CodeMode::Train) {
      ofs << "  micro_model->train_mode = true;\n";
    }
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
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->owned = false;\n";
    };
    std::vector<Tensor *> inputs = ctx->graph_inputs();
    std::vector<Tensor *> outputs = ctx->graph_outputs();
    if (config.code_mode() == CodeMode::Inference) {
      outputs = ctx->graph_outputs();
    } else if (config.code_mode() == CodeMode::Train) {
      outputs = ctx->graph_train_outputs();
    }
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
    ofs << "  return kMSStatusSuccess;\n";
  } else {
    ofs << "MSStatus MSModelCreate" << ctx->GetCurModelIndex() << "(MicroModel *micro_model) {\n";
    ofs << "  micro_model->train_mode = false;\n";
    ofs << "  return kMSStatusSuccess;\n";
  }
  ofs << "}\n\n";
}

void CodeMSModelBuildState(std::ofstream &ofs) { ofs << micro_model_build_state; }

void CodeMSModelBuildCommon(std::ofstream &ofs, const Configurator &config) {
  ofs << micro_model_build_implement;
  ofs << R"RAW(
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (micro_model->build == NULL) {
    return kMSStatusLiteNullptr;
  }
)RAW";
  if (config.target() != kCortex_M && !config.dynamic_shape()) {
    ofs << "  IncRefCount();\n";
  }
  ofs << R"RAW(
  MSStatus ret =
    micro_model->build(model, model_data, data_size, model_context);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(model);
  }
  return ret;
}
)RAW";
}

void CodeMSModelBuild(std::ofstream &ofs, const int model_index, const size_t weight_size, const Configurator &config) {
  ofs << "MSStatus MSModelBuild" << model_index
      << "(MSModelHandle model, const void *model_data, size_t data_size,\n"
         "                      const MSContextHandle model_context) {\n"
         "  if (model == NULL) {\n"
         "    return kMSStatusLiteParamInvalid;\n"
         "  }\n";
  if (config.changeable_weights_name().empty()) {
    ofs << "  if (data_size != " << weight_size
        << ") {\n"
           "    return kMSStatusLiteInputParamInvalid;\n"
           "  }\n";
  }
  ofs << "  MicroModel *micro_model = (MicroModel *)model;\n"
         "  int ret = MSModelCreate"
      << model_index
      << "(micro_model);\n"
         "  if (ret != kMSStatusSuccess) {\n"
         "    return ret;\n"
         "  }\n";
  if (config.target() != kCortex_M) {
    ofs << "  ret = Init" << model_index << "((void*)model_data, data_size);\n";
  } else {
    ofs << "  ret = Init" << model_index << "(NULL, 0);\n";
  }
  if (config.support_parallel()) {
    ofs << "  MicroContext *micro_context = (MicroContext *)model_context;\n"
           "  if (micro_context == NULL) {\n"
           "      return kMSStatusLiteNullptr;"
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

void CodeMSModelResizeInit(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  const auto &dynamic_symbols_num = config.dynamic_symbols_num();
  std::string array_index;
  for (const auto num : dynamic_symbols_num) {
    array_index += "[" + std::to_string(num) + "]";
  }
  auto shapes = ctx->shape_all_scenes();
  if (!shapes.empty()) {
    auto num_of_each_scene = shapes.begin()->second.size();
    ofs << "  static int shapes" << array_index << "[" + std::to_string(num_of_each_scene) + "] = {";
    for (auto &item : shapes) {
      auto &shape_val = item.second;
      for (size_t j = 0; j < shape_val.size(); ++j) {
        ofs << shape_val[j] << ", ";
      }
    }
    ofs << "};\n";
  }
  auto offsets = ctx->offset_all_scenes();
  if (!offsets.empty()) {
    auto num_of_each_scene = offsets.begin()->second.size();
    ofs << "  static int offsets" << array_index << "[" + std::to_string(num_of_each_scene) + "] = {";
    for (auto &item : offsets) {
      auto &offset_val = item.second;
      for (size_t j = 0; j < offset_val.size(); ++j) {
        ofs << offset_val[j] << ", ";
      }
    }
    ofs << "};\n";
  }
  ofs << "  size_t buffer_sizes" << array_index << " = {";
  auto buffer_size = ctx->buffer_sizes();
  auto workspace = ctx->workspaces();
  if (buffer_size.size() != workspace.size()) {
    return;
  }
  for (size_t i = 0; i < buffer_size.size(); i++) {
    ofs << buffer_size[i] + workspace[i] << ", ";
  }
  ofs << "};\n";
}

void CodeRealDimImplement(std::ofstream &ofs, const Configurator &config,
                          const std::map<std::string, std::vector<int>> &symbol_to_indexes,
                          const std::map<std::string, std::string> &user_to_inner,
                          std::map<std::string, std::string> *inner_to_outer) {
  int index = 0;
  for (auto &item : symbol_to_indexes) {
    ofs << "  int dim" << index << " = shape_infos[" << item.second[0] << "].shape[" << item.second[1] << "];\n";
    (*inner_to_outer)[item.first] = "dim" + std::to_string(index);
    std::string cur_dim_symbol;
    for (std::map<std::string, std::string>::const_iterator it = user_to_inner.begin(); it != user_to_inner.end();
         ++it) {
      if (it->second == item.first) {
        cur_dim_symbol = it->first;
        break;
      }
    }
    auto dynamic_dim_range = config.dynamic_symbols_map().at(cur_dim_symbol);
    ofs << "  int dim" << index << "_range[" << dynamic_dim_range.size() << "] = {";
    for (const auto dim : dynamic_dim_range) {
      ofs << dim << ", ";
    }
    ofs << "};\n";
    ++index;
  }
}

void CodeMSModelResize(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  auto &shape_templates = ctx->shape_templates();
  ofs << "MSStatus MSModelResize" << ctx->GetCurModelIndex()
      << "(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos, size_t shape_info_num) {\n"
         "  if (model == NULL) {\n"
         "    return kMSStatusLiteParamInvalid;\n"
         "  }\n";
  if (!config.dynamic_shape()) {
    ofs << "  return kMSStatusLiteNotSupport;\n";
  } else {
    ofs << "  MicroModel *micro_model = (MicroModel *)model;\n"
        << "  if (micro_model == NULL) {\n"
           "    return kMSStatusLiteNullptr;\n"
           "  }\n";
    CodeMSModelResizeInit(ofs, ctx, config);
    std::map<std::string, std::vector<int>> symbol_to_indexes;
    std::map<std::string, std::string> user_to_inner;
    auto &user_graph_inputs_template = config.user_graph_inputs_template();
    for (size_t i = 0; i < ctx->graph_inputs().size(); ++i) {
      auto cur_tensor = ctx->graph_inputs()[i];
      auto cur_shapes = shape_templates.at(cur_tensor);
      for (size_t j = 0; j < cur_shapes.size(); ++j) {
        if (IsNumber(cur_shapes.at(j))) {
          continue;
        }
        ofs << "  if (shape_infos[" << i << "].shape[" << j << "] <= 0) {\n"
            << "    return kMSStatusLiteParamInvalid;\n"
            << "  }\n";
        ofs << "  ((MicroTensor *)(inputs.handle_list[" << i << "]))->shape[" << j << "] = shape_infos[" << i
            << "].shape[" << j << "];\n";
        if (symbol_to_indexes.find(cur_shapes.at(j)) != symbol_to_indexes.end()) {
          continue;
        }
        symbol_to_indexes[cur_shapes.at(j)] = {static_cast<int>(i), static_cast<int>(j)};
        user_to_inner[user_graph_inputs_template[i][j]] = cur_shapes.at(j);
      }
    }
    std::map<std::string, std::string> inner_to_outer;
    CodeRealDimImplement(ofs, config, symbol_to_indexes, user_to_inner, &inner_to_outer);
    std::string condition;
    int index = 0;
    for (; index < static_cast<int>(symbol_to_indexes.size()) - 1; ++index) {
      condition += "store" + std::to_string(ctx->GetCurModelIndex()) + "_" + std::to_string(index) + " == dim" +
                   std::to_string(index) + " && ";
    }
    condition += "store" + std::to_string(ctx->GetCurModelIndex()) + "_" + std::to_string(index) + " == dim" +
                 std::to_string(index);
    ofs << "  if (" << condition << ") {\n"
        << "    return kMSStatusSuccess;\n"
        << "  }\n";
    for (size_t i = 0; i < symbol_to_indexes.size(); ++i) {
      ofs << "  store" + std::to_string(ctx->GetCurModelIndex()) + "_" << i << " = dim" << i << ";\n";
    }
    auto &dynamic_symbols = config.dynamic_symbols();
    int id = 0;
    for (auto &symbol : dynamic_symbols) {
      auto cur_dim = inner_to_outer[user_to_inner[symbol]];
      auto dim_list = cur_dim + "_range";
      ofs << "  int index" << id << " = 0;\n";
      ofs << "  for (int i = 0; i < sizeof(" << dim_list << ") / sizeof(" << dim_list << "[0]); i++) {\n"
          << "    if (" << dim_list << "[i] == " << cur_dim << ") {\n"
          << "      index" << id << " = i;\n"
          << "      break;\n"
          << "    }\n"
          << "  }\n";
      id++;
    }
    ofs << "  if (" << kBufferPrefixName << " != NULL) {\n";
    ofs << "    free(" << kBufferPrefixName << ");\n";
    ofs << "    " << kBufferPrefixName << " = NULL;\n";
    ofs << "  }\n";
    std::string array_index_str;
    for (size_t i = 0; i < dynamic_symbols.size(); i++) {
      array_index_str += "[index" + std::to_string(i) + "]";
    }
    ofs << "  " << kBufferPrefixName << " = malloc(buffer_sizes" << array_index_str << ");\n";
    ofs << "  micro_model->runtime_buffer = " << kBufferPrefixName << ";\n";
    ofs << "  " << kShapePrefixName << " = &shapes" << array_index_str << "[0];\n";
    ofs << "  " << kOffsetPrefixName << " = &offsets" << array_index_str << "[0];\n";
    ofs << "  MSTensorHandleArray outputs = MSModelGetOutputs(model);\n";
    for (size_t i = 0; i < ctx->graph_outputs().size(); ++i) {
      ofs << "  MSTensorSetData(outputs.handle_list[" << i << "], NULL);\n";
      auto cur_tensor = ctx->graph_outputs()[i];
      auto cur_shapes = shape_templates.at(cur_tensor);
      for (size_t j = 0; j < cur_shapes.size(); ++j) {
        if (IsNumber(cur_shapes.at(j))) {
          continue;
        }
        ofs << "  ((MicroTensor *)(outputs.handle_list[" << i << "]))->shape[" << j << "] = " << cur_shapes.at(j)
            << ";\n";
      }
    }
    ofs << "  return kMSStatusSuccess;\n";
  }
  ofs << "}\n";
}

void CodeMSModelDestory(std::ofstream &ofs, const Configurator *config) {
  if (config->target() != kCortex_M) {
    ofs << handle_array_destroy;
  }
  ofs << "void MSModelDestroy(MSModelHandle *model) {\n";
  ofs << "  if (*model) {\n"
         "    MicroModel *micro_model = (MicroModel *)*model;\n";
  if (config->target() != kCortex_M) {
    ofs << "    if (micro_model->runtime_buffer) {\n";
    if (config->dynamic_shape()) {
      ofs << "      free(micro_model->runtime_buffer);\n";
    } else {
      ofs << "      micro_model->runtime_buffer = NULL;\n";
    }
    ofs << "    }\n";
    ofs << "    MSTensorHandleArrayDestroy(micro_model->inputs);\n"
           "    MSTensorHandleArrayDestroy(micro_model->outputs);\n"
           "    micro_model->inputs.handle_list = NULL;\n"
           "    micro_model->outputs.handle_list = NULL;\n"
           "    micro_model->free_resource();\n";
    if (!config->dynamic_shape()) {
      ofs << "    DecRefCount();\n";
    }
    ofs << "  }\n";

    if (config->support_parallel()) {
      ofs << "  ClearThreadPool();\n";
    }
  } else {
    ofs << "    micro_model->runtime_buffer = NULL;\n"
           "    *model = NULL;\n"
           "  }\n";
  }
  ofs << "}\n";
}

void CodeMSModelPredictState(std::ofstream &ofs) { ofs << micro_model_predict_state; }

void CodeMSModelPredictCommon(std::ofstream &ofs) {
  ofs << R"RAW(
MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
                        const MSKernelCallBackC before, const MSKernelCallBackC after) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (micro_model->predict == NULL) {
    return kMSStatusLiteNullptr;
  }
  return micro_model->predict(model, inputs, outputs, before, after);
}

)RAW";
}

void CodeMSModelPredict(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  auto inputs_num = ctx->graph_inputs().size();
  auto outputs_num = ctx->graph_outputs().size();
  ofs << "MSStatus MSModelPredict" << ctx->GetCurModelIndex()
      << "(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,\n"
      << "                         const MSKernelCallBackC before, const MSKernelCallBackC after) {\n";
  ofs << R"RAW(
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (micro_model->runtime_buffer == NULL) {
    return kMSStatusLiteMemoryFailed;
  }
)RAW";
  ofs << "  if (inputs.handle_num != " << inputs_num << ") {\n";
  ofs << "    return kMSStatusLiteParamInvalid;\n";
  ofs << "  }\n";
  ofs << "  if (outputs->handle_num != " << outputs_num << ") {\n";
  ofs << "    return kMSStatusLiteParamInvalid;\n";
  ofs << "  }\n";
  if (config.target() != kCortex_M && !config.dynamic_shape()) {
    ofs << "  if (!LockBuffer(micro_model->runtime_buffer)) {\n"
        << "    void *buffer = Malloc(GetBufferSize" << ctx->GetCurModelIndex() << "());\n"
        << "    if (buffer == NULL) {\n"
        << "      return kMSStatusLiteNullptr;\n"
        << "    }\n"
        << "    if (micro_model->runtime_buffer != buffer) {\n"
        << "      micro_model->runtime_buffer = buffer;\n"
        << "      int ret = SetBuffer" << ctx->GetCurModelIndex() << "(((MemBlock *)buffer)->addr);\n"
        << "      if (ret != kMSStatusSuccess) {\n"
        << "        return kMSStatusLiteMemoryFailed;\n"
        << "      }\n"
        << "    }\n"
        << "  }\n";
  }
  ofs << "  const void *inputs_data_array[" << inputs_num << "];\n";
  ofs << "  int expect_types[" << inputs_num << "] = {";
  for (size_t i = 0; i < inputs_num; ++i) {
    ofs << ctx->graph_inputs().at(i)->data_type() << ", ";
  }
  ofs << "};\n";
  ofs << "  bool type_changed[" << inputs_num << "] = {";
  for (size_t i = 0; i < inputs_num; ++i) {
    ofs << "false, ";
  }
  ofs << "};\n";
  ofs << "  for (int i = 0; i < " << inputs_num << "; i++) {\n";
  ofs << "    inputs_data_array[i] = TransformInput((MicroTensor *)inputs.handle_list[i], expect_types[i], "
         "&type_changed[i]);\n";
  ofs << "  }\n";
  ofs << "  SetInputs" << ctx->GetCurModelIndex() << "(inputs_data_array, " << inputs_num << ");\n";
  ofs << "  Execute" << ctx->GetCurModelIndex() << "(micro_model->train_mode);\n";
  ofs << "\n";
  ofs << "  for (int i = 0; i < " << inputs_num << "; i++) {\n";
  ofs << "    if (type_changed[i]) {\n";
  ofs << "      free((void *)inputs_data_array[i]);\n";
  ofs << "    }\n";
  ofs << "  }\n";
  ofs << "\n";
  ofs << "  int cur_out_types[" << outputs_num << "] = {";
  for (size_t i = 0; i < outputs_num; ++i) {
    ofs << ctx->graph_outputs().at(i)->data_type() << ", ";
  }
  ofs << "};\n";
  ofs << "  bool out_type_changed[" << outputs_num << "] = {";
  for (size_t i = 0; i < outputs_num; ++i) {
    ofs << "false, ";
  }
  ofs << "};\n";
  ofs << "  MSStatus ret = CopyOutputsData" << ctx->GetCurModelIndex()
      << "(outputs, cur_out_types, out_type_changed);\n";
  if (config.target() != kCortex_M && !config.dynamic_shape()) {
    ofs << "  UnLockBuffer(micro_model->runtime_buffer);\n";
  }
  ofs << "  return ret;\n";
  ofs << "}\n";
}

void CodeCopyOutputsState(std::ofstream &ofs, const int model_index) {
  ofs << "int CopyOutputsData" << model_index
      << "(MSTensorHandleArray *outputs_ori, void **outputs, int *cur_types, bool *type_changed);\n\n";
}

void CodeCopyOutputsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  auto tensor_map = ctx->tensors_map();
  std::vector<Tensor *> outputs = ctx->graph_outputs();
  size_t outputs_size = outputs.size();

  ofs << "int CopyOutputsData" << ctx->GetCurModelIndex()
      << "(MSTensorHandleArray *outputs_ori, int *cur_out_types, bool *type_changed) {\n"
         "  if (outputs_ori == NULL || cur_out_types == NULL || type_changed == NULL) {\n"
         "    return kMSStatusLiteNullptr;\n"
         "  }\n";
  ofs << "  unsigned char *buffer[" << outputs_size << "] = {";
  for (size_t i = 0; i < outputs_size; ++i) {
    auto out_str = ctx->tensor_addr(outputs[i]);
    if (out_str.empty()) {
      ofs << tensor_map[outputs[i]] << ", ";
    } else {
      ofs << out_str << ", ";
    }
  }
  ofs << "};\n";
  ofs << "  for (int i = 0; i < " << outputs_size << "; i++) {\n"
      << "    MicroTensor *micro_tensor = (MicroTensor *)outputs_ori->handle_list[i];\n"
      << "    int expect_type = micro_tensor->type;\n"
      << "    int cur_type = cur_out_types[i];\n";
  ofs << "    if (expect_type == cur_type) {\n"
      << "      micro_tensor->data = buffer[i];\n"
      << "      micro_tensor->owned = false;\n"
      << "      continue;\n"
      << "    }\n";
  ofs << "#ifdef ENABLE_FP16\n"
      << "    int type_trans_mode = TypeTransMode_MAX;\n"
         "    if (expect_type == kMSDataTypeNumberTypeFloat16 && cur_type == kMSDataTypeNumberTypeFloat32) {\n"
         "      type_trans_mode = TypeTransMode_FP32_TO_FP16;\n"
         "    } else if (expect_type == kMSDataTypeNumberTypeFloat32 && cur_type == kMSDataTypeNumberTypeFloat16) {\n"
         "      type_trans_mode = TypeTransMode_FP16_TO_FP32;\n"
         "    }\n";
  ofs << "    if (type_trans_mode == TypeTransMode_UNSUPPORT) {\n"
      << "      return kMSStatusLiteNotSupport;\n"
      << "    }\n";
  ofs << "    int shape_size = micro_tensor->ndim;\n"
      << "    int num = 1;\n"
      << "    for (int i = 0; i < shape_size; ++i) {\n"
      << "      num *= micro_tensor->shape[i];\n"
      << "    }\n";
  ofs << "    void *out_data = MSTensorGetMutableData(micro_tensor);\n"
      << "    if (type_trans_mode == TypeTransMode_FP32_TO_FP16) {\n"
      << "      Fp32CastToFp16((float *)(buffer[i]), (float16_t *)out_data, num);\n"
      << "      type_changed[i] = true;\n"
      << "    } else if (type_trans_mode == TypeTransMode_FP16_TO_FP32) {\n"
      << "      Fp16CastToFp32((float16_t *)(buffer[i]), (float *)out_data, num);\n"
      << "      type_changed[i] = true;\n"
      << "    }\n"
      << "#endif\n"
      << "  }\n";
  ofs << "  return RET_OK;\n"
         "}\n\n";
}

void CodeGlobalCodeBlocks(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  for (const auto &block : ctx->global_code_blocks()) {
    ofs << block << "\n";
  }
}

void CodeInputState(std::ofstream &ofs, const int model_index) {
  ofs << "/**\n"
      << "  * set input tensors\n"
      << "  * @param inputs, the input data ptr's array of the model, the tensors' count of input may be greater than "
         "one.\n"
      << "  * @param num, the input data's number of the model.\n"
      << "  **/\n"
      << "int "
      << "SetInputs" << model_index << "(const void **inputs, int num);\n\n";
}

void CodeInputImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  // input tensors
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    ofs << "const unsigned char *" << ctx->input_name() + std::to_string(i) << " = 0;\n";
  }
  size_t size = inputs.size();
  ofs << "int "
      << "SetInputs" << ctx->GetCurModelIndex() << "(const void **inputs, int num) {\n"
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

void CodeGraphQuantArgsState(std::ofstream &ofs, const int model_index) {
  ofs << "/**\n"
      << "  * get input and output QuantArgs of the model \n"
      << "  **/\n"
      << "GraphQuantArgs "
      << "GetInOutQuantArgs" << model_index << "();\n\n";
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
      << "GetInOutQuantArgs" << ctx->GetCurModelIndex() << "() {\n"
      << "\t\tGraphQuantArgs quan_args = { " << in_quant_args.at(0).scale << ", " << out_quant_args.at(0).scale << ", "
      << in_quant_args.at(0).zeroPoint << ", " << out_quant_args.at(0).zeroPoint << "};\n"
      << "\t\treturn quan_args;\n"
      << "}\n";
}

void CodeManageResourceState(std::ofstream &ofs, const int model_index) {
  ofs << "/**\n"
      << "  * get the memory space size of the inference.\n"
      << "  **/\n"
      << "int "
      << "GetBufferSize" << model_index << "();\n";

  ofs << "/**\n"
      << "  * set the memory space for the inference\n"
      << "  **/\n"
      << "int "
      << "SetBuffer" << model_index << "(void *buffer);\n\n";

  ofs << "/**\n"
      << "  * free the memory of packed weights, and set the membuf buffer and input address to NULL\n"
      << "  **/\n"
      << "void "
      << "FreeResource" << model_index << "();\n";
}

void CodeInitResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  ofs << "int "
      << "GetBufferSize" << ctx->GetCurModelIndex() << "() {\n"
      << "  return " << ctx->total_buffer_size() << ";\n"
      << "}\n";
  ofs << "int "
      << "SetBuffer" << ctx->GetCurModelIndex() << "( void *buffer) {\n";
  ofs << "  " << ctx->buffer_name() << " = (unsigned char *)buffer;\n"
      << "  return RET_OK;\n"
         "}\n";
}

void CodeResetImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  ofs << "void Reset" << ctx->GetCurModelIndex() << "() {\n";
  auto &dynamic_symbols = config.dynamic_symbols();
  for (size_t i = 0; i < dynamic_symbols.size(); ++i) {
    ofs << "  store" << ctx->GetCurModelIndex() << "_" << i << " = -1;\n";
  }
  ofs << "  FreeResource" << ctx->GetCurModelIndex() << "();\n";
  ofs << "}\n";
}

void CodeFreeResourceState(std::ofstream &ofs) { ofs << free_resource_state; }

void CodeFreeResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx,
                               const Configurator &config) {
  ofs << "void "
      << "FreeResource" << ctx->GetCurModelIndex() << "() {\n";
  if (config.target() != kCortex_M) {
    ofs << "  " << ctx->buffer_name() << "= NULL;\n";
    std::vector<Tensor *> inputs = ctx->graph_inputs();
    size_t size = inputs.size();
    for (size_t i = 0; i < size; ++i) {
      ofs << "  " << ctx->input_name() + std::to_string(i) << " = NULL;\n";
    }
    ofs << "  void **allocated[] = {\n";
    size_t num = 0;
    auto &w_auxiliary = ctx->auxiliary_weights();
    for (const auto &item : ctx->tensors_map()) {
      Tensor *tensor = item.first;
      std::string name = item.second;
      if (tensor->data() != nullptr &&
          (!(CheckConstantTensor(tensor)) || w_auxiliary.find(tensor) != w_auxiliary.end())) {
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
  }
  ofs << "}\n";
}

void CodeExecuteState(std::ofstream &ofs, const int model_index) {
  ofs << "/**\n"
      << "  * net execute function\n"
      << "  **/\n"
      << "void "
      << "Execute" << model_index << "(bool train_mode);\n\n";
}
}  // namespace mindspore::lite::micro
