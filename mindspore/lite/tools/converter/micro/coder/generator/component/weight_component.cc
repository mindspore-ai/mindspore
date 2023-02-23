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

#include "coder/generator/component/weight_component.h"
#include <memory>
#include <utility>
#include "coder/generator/component/const_blocks/license.h"
#include "coder/utils/coder_utils.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
void CodeWeightFileHeader(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  ofs << g_hwLicense;
  // include all operator header
  for (const auto &h_file : ctx->h_files()) {
    ofs << "#include \"" << h_file << "\"\n";
  }
  ofs << "#include <stdlib.h>\n"
      << "#include <stdint.h>\n"
      << "#include <string.h>\n"
      << "extern unsigned char *" << ctx->buffer_name() << ";\n"
      << "extern uint8_t *" << ctx->weight_name() << ";\n"
      << "enum STATUS {\n"
         "  RET_OK = 0,\n"
         "  RET_ERROR = 1,\n"
         "};\n\n";
  // set a global var for thread_pool
  ofs << "extern int " << gThreadNum << ";\n";
}

void CodeModelParamsState(std::ofstream &ofs, const std::map<std::string, Tensor *> &weights) {
  for (auto &item : weights) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (CheckConstantTensor(tensor)) {
      if (tensor->data() == nullptr) {
        continue;
      }
      ofs << "extern const " << GetTensorDataType(tensor->data_type()) << name << "[];\n";
    }
  }
}

void CodeModelParamsData(std::ofstream &ofs, const std::map<std::string, Tensor *> &weights) {
  for (auto &item : weights) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (CheckConstantTensor(tensor)) {
      if (tensor->data() == nullptr) {
        continue;
      }
      ofs << "const " << GetTensorDataType(tensor->data_type()) << name << "[] = ";
      PrintTensorData(tensor, ofs);
    }
  }
}

void CodeModelParamsForNet(std::ofstream &hofs, std::ofstream &cofs, const std::unique_ptr<CoderContext> &ctx,
                           const Configurator &config) {
  // reverse key and value of tensors_map
  std::map<std::string, Tensor *> address_map;
  for (const auto &item : ctx->tensors_map()) {
    address_map.insert(std::make_pair(item.second, item.first));
  }
  for (auto &item : address_map) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (tensor->data() == nullptr) {
      continue;
    }
    if (CheckConstantTensor(tensor)) {
      if (config.target() != kCortex_M) {
        hofs << "extern " << GetTensorDataType(tensor->data_type()) << name << "[];  // " << tensor->tensor_name()
             << std::endl;
        cofs << GetTensorDataType(tensor->data_type()) << name << "[" << tensor->ElementsNum() << "];\n";
      } else {
        hofs << "extern const " << GetTensorDataType(tensor->data_type()) << name << "[];  // " << tensor->tensor_name()
             << std::endl;
      }
    } else if (tensor->category() == lite::Category::VAR) {
      hofs << "extern " << GetTensorDataType(tensor->data_type()) << "*" << name << ";  // " << tensor->tensor_name()
           << std::endl;
      cofs << GetTensorDataType(tensor->data_type()) << "*" << name << " = NULL;\n";
    }
  }
  cofs << "\n";
}

void CodeInitWeightState(std::ofstream &ofs, const int model_index) {
  ofs << "/// \\brief Init model weight from buffer.\n\n"
      << "/// \\param[in] weight_buffer The address of the weight binary file.\n"
      << "/// \\param[in] weight_size The size of the weight file in bytes.\n"
      << "int Init" << model_index << "(void *weight_buffer, int weight_size);\n\n";
}

void CodeExportWeightState(std::ofstream &ofs, const int model_index) {
  ofs << "/// \\brief Export model weight to the specified path.\n\n"
      << "/// \\param[in] output_weight_file The path of the export weight file.\n\n"
      << "/// \\return 0 on success or -1 in case of error.\n"
      << "int Export" << model_index << "(const char* output_weight_file);\n\n";
}

void CodeWeightInitFunc(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  if (config.target() != kCortex_M) {
    ofs << "static size_t PackWeightSize" << ctx->GetCurModelIndex() << "() {\n";
    ofs << "  size_t w_size = 0;\n";
    for (const auto &block : ctx->GetInitWeightSizeCode()) {
      ofs << "  " << block;
    }
    ofs << "  return w_size;\n";
    ofs << "}\n\n";

    ofs << "int Init" << ctx->GetCurModelIndex() << "(void *weight_buffer, int weight_size) {\n"
        << "  if (weight_buffer == NULL) {\n"
        << "    return RET_ERROR;\n"
        << "  }\n";
    ofs << "  struct ModelParameter {\n"
        << "    void *addr;\n"
        << "    size_t size;\n"
        << "    size_t offset;\n"
        << "  };\n";

    ofs << "  size_t " << ctx->weight_size_name() << " = PackWeightSize" << ctx->GetCurModelIndex() << "();\n";
    size_t params_num = 0;
    size_t offset = 0;
    std::string params;
    std::string origins;
    for (const auto &item : ctx->saved_weights()) {
      std::string name = item.first;
      Tensor *tensor = item.second;
      if (!CheckConstantTensor(tensor)) {
        continue;
      }
      std::map<Tensor *, std::string> ctx_tensor_map = ctx->tensors_map();
      auto iter = ctx_tensor_map.find(tensor);
      if (iter != ctx_tensor_map.end()) {
        origins += "    {" + name + ", " + std::to_string(tensor->Size()) + ", " + std::to_string(offset) + "},\n";
        params_num++;
      } else {
        TypeId data_type = tensor->data_type();
        params +=
          "  " + GetTensorDataType(data_type) + "*" + name + " = (weight_buffer + " + std::to_string(offset) + ");\n";
      }
      offset += tensor->Size();
    }
    ofs << params << "\n";
    ofs << "  struct ModelParameter model_params[] = {\n" << origins << "  };\n";
    ofs << "\n";
    ofs << "  for(int i = 0; i < " << params_num << "; ++i) {\n"
        << "    if (model_params[i].offset + model_params[i].size > weight_size) {\n"
           "      return RET_ERROR;\n"
           "    }\n"
        << "    memcpy(model_params[i].addr, (weight_buffer + model_params[i].offset), model_params[i].size);\n"
        << "  }\n";
    ofs << "  if (" << ctx->weight_size_name() << " > 0) {\n";
    ofs << "    " << ctx->weight_name() << " = malloc(" << ctx->weight_size_name() << ");\n";
    ofs << "    if (" << ctx->weight_name() << " == NULL) {\n      return RET_ERROR;\n    }\n";
    ofs << "    memset(" << ctx->weight_name() << ", 0, " << ctx->weight_size_name() << ");\n";
    ofs << "  }\n";
  } else {
    ofs << "int Init" << ctx->GetCurModelIndex() << "(void *weight_buffer, int weight_size) {\n";
    ofs << "  if (" << ctx->weight_name() << "== NULL) {\n";
    ofs << "    return RET_ERROR;\n  }\n";
    ofs << "  const size_t w_size = " << ctx->weight_buffer_size() << ";\n";
  }
  ofs << "  size_t " << ctx->weight_offset_name() << " = 0;\n";
  for (const auto &block : ctx->init_contents()) {
    ofs << "\n{\n" << block << "}\n";
  }
  ofs << "  if (" << ctx->weight_size_name() << " < " << ctx->weight_offset_name()
      << ") {\n    return RET_ERROR;\n  }\n";
  ofs << "  return RET_OK;\n";
  ofs << "}\n\n";
}

void CodeWeightExportFunc(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  if (config.target() == kCortex_M) {
    MS_LOG(DEBUG) << "weight file is unsupported to export when in Cortex M mode.";
    return;
  }
  ofs << "int Export" << ctx->GetCurModelIndex() << "(const char* output_weight_file) {\n"
      << "  if (output_weight_file == NULL) {\n"
      << "    return RET_ERROR;\n"
      << "  }\n\n"
      << "  FILE *fp;\n"
      << "  if((fp = fopen(output_weight_file, \"wb\")) == NULL) {\n"
      << "    printf(\"open file failed.\");\n"
      << "    return RET_ERROR;\n"
      << "  }\n"
      << "  int params_len = sizeof(model_params) / sizeof(model_params[0]);\n"
      << "  for (int i = 0; i < params_len; ++i) {\n"
      << "    fwrite(model_params[i].addr, sizeof(char), model_params[i].size, fp);\n"
      << "  }\n"
      << "  fclose(fp);\n"
      << "  return RET_OK;\n"
      << "}\n";
}

void SaveDataToNet(const std::map<std::string, Tensor *> &saved_weights, const std::string &net_file) {
  std::ofstream net(net_file, std::ios::out | std::ios::trunc | std::ios::binary);
  MS_CHECK_TRUE_WITHOUT_RET(net.is_open(), "net file open failed!");
  for (auto &item : saved_weights) {
    Tensor *tensor = item.second;
    if ((CheckConstantTensor(tensor)) && tensor->data() != nullptr) {
      net.write(reinterpret_cast<const char *>(tensor->data()), tensor->Size());
    }
  }
  net.close();
}
}  // namespace mindspore::lite::micro
