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
#include <algorithm>
#include <memory>
#include <utility>
#include "coder/generator/component/const_blocks/license.h"
#include "coder/utils/coder_utils.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro {
namespace {
constexpr size_t kMaxLineSize = 120;
struct camp {
  bool operator()(const std::string &a, const std::string &b) const { return a.size() < b.size() || a < b; }
};

std::string GenerateArrayContent(const std::vector<size_t> &contents, const std::string &prefix) {
  std::string lines;
  std::string line = prefix;
  for (auto content : contents) {
    std::string append = std::to_string(content) + ", ";
    if (line == prefix) {
      line += append;
      continue;
    }
    if (line.size() + append.size() > kMaxLineSize) {
      lines += line + "\n";
      line = prefix + append;
    } else {
      line += append;
    }
  }
  lines += line + "\n";
  return lines;
}
}  // namespace

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
  std::map<std::string, Tensor *, camp> address_map;
  for (const auto &item : ctx->tensors_map()) {
    address_map.insert(std::make_pair(item.second, item.first));
  }
  auto &w_auxiliary = ctx->auxiliary_weights();
  for (auto &item : address_map) {
    std::string name = item.first;
    Tensor *tensor = item.second;
    if (tensor->data() == nullptr) {
      continue;
    }
    if (CheckConstantTensor(tensor)) {
      if (config.target() != kCortex_M) {
        if (w_auxiliary.find(tensor) == w_auxiliary.end()) {
          hofs << "extern " << GetTensorDataType(tensor->data_type()) << name << "[];  // " << tensor->tensor_name()
               << std::endl;
          cofs << GetTensorDataType(tensor->data_type()) << name << "[" << tensor->ElementsNum() << "];\n";
        } else {
          hofs << "extern " << GetTensorDataType(tensor->data_type()) << "*" << name << ";  // "
               << tensor->tensor_name() << std::endl;
          cofs << GetTensorDataType(tensor->data_type()) << "*" << name << " = NULL;\n";
        }
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

void CodeWeightContentInit(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx,
                           const std::map<Tensor *, int> &tensors_index) {
  auto &w_auxiliary = ctx->auxiliary_weights();
  std::map<std::string, Tensor *, camp> real_need_tensors;
  auto record_saved_tensors = ctx->saved_weights();
  for (auto &item : record_saved_tensors) {
    real_need_tensors.insert(std::make_pair(item.first, item.second));
  }
  std::string non_copy;
  std::string copy_static;
  std::string copy_dynamic;
  int copy_static_num = 0;
  int copy_dynamic_num = 0;
  auto tensors_map = ctx->tensors_map();
  for (const auto &item : real_need_tensors) {
    if (!CheckConstantTensor(item.second) || item.second->data() == nullptr) {
      continue;
    }
    auto iter = tensors_map.find(item.second);
    if (iter == tensors_map.end()) {
      TypeId data_type = item.second->data_type();
      non_copy += "  " + GetTensorDataType(data_type) + "*" + item.first + " = (weight_buffer + offsets[" +
                  std::to_string(tensors_index.at(item.second)) + "]);\n";
      continue;
    }
    if (w_auxiliary.find(item.second) == w_auxiliary.end()) {
      copy_static += "    {" + item.first + ", " + std::to_string(tensors_index.at(item.second)) + "},\n";
      ++copy_static_num;
    } else {
      copy_dynamic += "    {&" + item.first + ", " + std::to_string(tensors_index.at(item.second)) + "},\n";
      ++copy_dynamic_num;
    }
  }
  for (const auto &item : w_auxiliary) {
    copy_static += "    {" + item.second.second + ", " + std::to_string(tensors_index.at(item.second.first)) + "},\n";
    ++copy_static_num;
  }
  ofs << non_copy << "\n";
  if (copy_static_num > 0) {
    ofs << "  {\n  struct ModelParameter static_copy[] = {\n" << copy_static << "    };\n";
    ofs << "    for(int i = 0; i < " << copy_static_num << "; ++i) {\n"
        << "      int index = static_copy[i].index;\n"
        << "      if (offsets[index] + tensors_size[index] > weight_size) {\n"
           "        return RET_ERROR;\n"
           "      }\n"
        << "      memcpy(static_copy[i].addr, (weight_buffer + offsets[index]), tensors_size[index]);\n"
        << "    }\n  }\n\n";
  }
  ofs << "  size_t " << ctx->weight_size_name() << " = PackWeightSize" << ctx->GetCurModelIndex()
      << "() + dynamic_memory;\n";
  ofs << "  if (" << ctx->weight_size_name() << " > 0) {\n";
  ofs << "    " << ctx->weight_name() << " = malloc(" << ctx->weight_size_name() << ");\n";
  ofs << "    if (" << ctx->weight_name() << " == NULL) {\n      return RET_ERROR;\n    }\n";
  ofs << "    memset(" << ctx->weight_name() << ", 0, " << ctx->weight_size_name() << ");\n";
  ofs << "  }\n";
  ofs << "  size_t " << ctx->weight_offset_name() << " = 0;\n";
  if (copy_dynamic_num > 0) {
    ofs << "  {\n  struct ModelParameter dynamic_copy[] = {\n" << copy_dynamic << "  };\n";
    ofs << "    for(int i = 0; i < " << copy_dynamic_num << "; ++i) {\n"
        << "      int index = dynamic_copy[i].index;\n"
        << "      memcpy(" << ctx->weight_name() << " + " << ctx->weight_offset_name()
        << ", (weight_buffer + offsets[index]), tensors_size[index]);\n"
        << "      *((void **)dynamic_copy[i].addr) = " << ctx->weight_name() << " + " << ctx->weight_offset_name()
        << ";\n"
        << "      " << ctx->weight_offset_name() << " += tensors_size[index];\n"
        << "    }\n  }\n\n";
  }
}

void CodeWeightInitIfKeepWeight(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  const auto &w_origin = ctx->origin_weights();
  const auto &w_auxiliary = ctx->auxiliary_weights();
  std::vector<size_t> tensors_size;
  std::vector<size_t> online_compute_index;
  std::map<Tensor *, int> tensors_index;
  for (auto tensor : w_origin) {
    if (!(CheckConstantTensor(tensor)) || tensor->data() == nullptr) {
      continue;
    }
    auto iter = w_auxiliary.find(tensor);
    if (iter == w_auxiliary.end()) {
      tensors_index[tensor] = tensors_size.size();
      tensors_size.push_back(tensor->Size());
    } else {
      tensors_index[iter->second.first] = tensors_size.size();
      tensors_size.push_back(iter->second.first->Size());
      tensors_index[tensor] = tensors_size.size();
      online_compute_index.push_back(tensors_size.size());
      tensors_size.push_back(DataTypeSize(tensor->data_type()));
    }
  }
  std::vector<size_t> offsets{0};
  int last = online_compute_index.empty() ? tensors_size.size() - 1 : online_compute_index.front();
  for (int i = 1; i <= last; ++i) {
    offsets.push_back(offsets[i - 1] + tensors_size[i - 1]);
  }
  ofs << "int Init" << ctx->GetCurModelIndex() << "(void *weight_buffer, int weight_size) {\n"
      << "  if (weight_buffer == NULL) {\n"
      << "    return RET_ERROR;\n"
      << "  }\n";
  ofs << "  struct ModelParameter {\n"
      << "    void *addr;\n"
      << "    int index;\n"
      << "  };\n";
  ofs << "  int offsets[" << std::to_string(tensors_size.size()) << "] = {\n"
      << GenerateArrayContent(offsets, "      ") << "  };\n";
  ofs << "  size_t tensors_size[" << std::to_string(tensors_size.size()) << "] = {\n"
      << GenerateArrayContent(tensors_size, "         ") << "  };\n";
  ofs << "  size_t dynamic_memory = 0;\n";
  offsets.insert(offsets.end(), tensors_size.size() - offsets.size(), 0);
  if (!online_compute_index.empty()) {
    ofs << "  int online_compute_index[] = {\n" << GenerateArrayContent(online_compute_index, "      ") << "  };\n";
    ofs << "  for (size_t i = 0; i < " << std::to_string(online_compute_index.size()) + "; ++i) {\n";
    ofs << "    int *shape = (int *)(weight_buffer + offsets[online_compute_index[i] - 1]);\n";
    ofs << "    int dim_num = tensors_size[online_compute_index[i] - 1] / 4;\n";
    ofs << "    size_t tensor_size = tensors_size[online_compute_index[i]];\n";
    ofs << "    for (int j = 0; j < dim_num; ++j) {\n";
    ofs << "      tensor_size *= shape[j];\n";
    ofs << "    }\n";
    ofs << "    tensors_size[online_compute_index[i]] = tensor_size;\n";
    ofs << "    dynamic_memory += tensor_size;\n";
    ofs << "    int next_index = (i + 1) < " << std::to_string(online_compute_index.size())
        << " ? online_compute_index[i + 1] : " << std::to_string(tensors_size.size()) << " - 1;\n";
    ofs << "    for (int j = online_compute_index[i] + 1; j <= next_index; ++j) {\n";
    ofs << "      offsets[j] = offsets[j - 1] + tensors_size[j - 1];\n";
    ofs << "    }\n  }\n";
  }
  CodeWeightContentInit(ofs, ctx, tensors_index);
}

void CodeWeightInitIfNonKeepWeight(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
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
  ofs << "  size_t " << ctx->weight_offset_name() << " = 0;\n";
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
    if (config.keep_original_weight()) {
      CodeWeightInitIfKeepWeight(ofs, ctx);
    } else {
      CodeWeightInitIfNonKeepWeight(ofs, ctx);
    }
  } else {
    ofs << "int Init" << ctx->GetCurModelIndex() << "(void *weight_buffer, int weight_size) {\n";
    ofs << "  if (" << ctx->weight_name() << "== NULL) {\n";
    ofs << "    return RET_ERROR;\n  }\n";
    ofs << "  const size_t w_size = " << ctx->weight_buffer_size() << ";\n";
    ofs << "  size_t " << ctx->weight_offset_name() << " = 0;\n";
  }
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

void SaveDataToNet(const std::unique_ptr<CoderContext> &ctx, const std::string &net_file, bool keep_weight,
                   size_t *weight_size) {
  std::ofstream net(net_file, std::ios::out | std::ios::trunc | std::ios::binary);
  MS_CHECK_TRUE_WITHOUT_RET(net.is_open(), "net file open failed!");
  std::vector<Tensor *> save_tensors;
  if (keep_weight) {
    const auto &w_origin = ctx->origin_weights();
    const auto &w_auxiliary = ctx->auxiliary_weights();
    (void)std::for_each(w_origin.begin(), w_origin.end(), [&save_tensors, &w_auxiliary](Tensor *tensor) {
      auto iter = w_auxiliary.find(tensor);
      if (iter != w_auxiliary.end()) {
        save_tensors.push_back(iter->second.first);
      }
      save_tensors.push_back(tensor);
    });
  } else {
    auto recorded_saved_tensors = ctx->saved_weights();
    (void)std::transform(recorded_saved_tensors.begin(), recorded_saved_tensors.end(), std::back_inserter(save_tensors),
                         [](const std::pair<std::string, Tensor *> &item) { return item.second; });
  }
  size_t size = 0;
  for (auto tensor : save_tensors) {
    if ((CheckConstantTensor(tensor)) && tensor->data() != nullptr) {
      net.write(reinterpret_cast<const char *>(tensor->data()), tensor->Size());
      size += tensor->Size();
    }
  }
  if (weight_size != nullptr) {
    *weight_size = size;
  }
  net.close();
}
}  // namespace mindspore::lite::micro
