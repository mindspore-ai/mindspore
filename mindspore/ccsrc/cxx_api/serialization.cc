/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "include/api/serialization.h"
#include <fstream>
#include "cxx_api/graph/graph_data.h"
#include "utils/log_adapter.h"
#include "mindspore/core/load_mindir/load_model.h"

namespace mindspore {
static Buffer ReadFile(const std::string &file) {
  Buffer buffer;
  if (file.empty()) {
    MS_LOG(ERROR) << "Pointer file is nullptr";
    return buffer;
  }

  char real_path_mem[PATH_MAX] = {0};
  char *real_path_ret = nullptr;
#if defined(_WIN32) || defined(_WIN64)
  real_path_ret = _fullpath(real_path_mem, common::SafeCStr(file), PATH_MAX);
#else
  real_path_ret = realpath(common::SafeCStr(file), real_path_mem);
#endif

  if (real_path_ret == nullptr) {
    MS_LOG(ERROR) << "File: " << file << " is not exist.";
    return buffer;
  }

  std::string real_path(real_path_mem);
  std::ifstream ifs(real_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "File: " << real_path << " is not exist";
    return buffer;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "File: " << real_path << "open failed";
    return buffer;
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  buffer.ResizeData(size);
  if (buffer.DataSize() != size) {
    MS_LOG(ERROR) << "Malloc buf failed, file: " << real_path;
    ifs.close();
    return buffer;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}

Graph Serialization::LoadModel(const void *model_data, size_t data_size, ModelType model_type) {
  if (model_type == kMindIR) {
    FuncGraphPtr anf_graph = nullptr;
    try {
      anf_graph = ConvertStreamToFuncGraph(reinterpret_cast<const char *>(model_data), data_size);
    } catch (const std::exception &) {
      MS_LOG(EXCEPTION) << "Load MindIR failed.";
    }

    return Graph(std::make_shared<Graph::GraphData>(anf_graph, kMindIR));
  } else if (model_type == kOM) {
    return Graph(std::make_shared<Graph::GraphData>(Buffer(model_data, data_size), kOM));
  }
  MS_LOG(EXCEPTION) << "Unsupported ModelType " << model_type;
}

Graph Serialization::LoadModel(const std::string &file, ModelType model_type) {
  Buffer data = ReadFile(file);
  if (data.Data() == nullptr) {
    MS_LOG(EXCEPTION) << "Read file " << file << " failed.";
  }
  if (model_type == kMindIR) {
    FuncGraphPtr anf_graph = nullptr;
    try {
      anf_graph = ConvertStreamToFuncGraph(reinterpret_cast<const char *>(data.Data()), data.DataSize());
    } catch (const std::exception &) {
      MS_LOG(EXCEPTION) << "Load MindIR failed.";
    }

    return Graph(std::make_shared<Graph::GraphData>(anf_graph, kMindIR));
  } else if (model_type == kOM) {
    return Graph(std::make_shared<Graph::GraphData>(data, kOM));
  }
  MS_LOG(EXCEPTION) << "Unsupported ModelType " << model_type;
}

Status Serialization::LoadCheckPoint(const std::string &ckpt_file, std::map<std::string, Buffer> *parameters) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::SetParameters(const std::map<std::string, Buffer> &parameters, Model *model) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, Buffer *model_data) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, const std::string &model_file) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}
}  // namespace mindspore
