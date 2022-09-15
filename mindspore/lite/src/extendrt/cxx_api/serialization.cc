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
#include <sstream>
#include "utils/log_adapter.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "extendrt/cxx_api/graph/graph_data.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "extendrt/cxx_api/dlutils.h"
#endif
#include "utils/crypto.h"
#include "extendrt/cxx_api/file_utils.h"

namespace mindspore {
static Status RealPath(const std::string &file, std::string *realpath_str) {
  MS_EXCEPTION_IF_NULL(realpath_str);
  char real_path_mem[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  auto real_path_ret = _fullpath(real_path_mem, common::SafeCStr(file), PATH_MAX);
#else
  auto real_path_ret = realpath(common::SafeCStr(file), real_path_mem);
#endif
  if (real_path_ret == nullptr) {
    return Status(kMEInvalidInput, "File: " + file + " does not exist.");
  }
  *realpath_str = real_path_mem;
  return kSuccess;
}

Buffer ReadFile(const std::string &file) {
  Buffer buffer;
  if (file.empty()) {
    MS_LOG(ERROR) << "Pointer file is nullptr";
    return buffer;
  }

  std::string real_path;
  auto status = RealPath(file, &real_path);
  if (status != kSuccess) {
    MS_LOG(ERROR) << status.GetErrDescription();
    return buffer;
  }

  std::ifstream ifs(real_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "File: " << real_path << " does not exist";
    return buffer;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "File: " << real_path << " open failed";
    return buffer;
  }

  (void)ifs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(ifs.tellg());
  buffer.ResizeData(size);
  if (buffer.DataSize() != size) {
    MS_LOG(ERROR) << "Malloc buf failed, file: " << real_path;
    ifs.close();
    return buffer;
  }

  (void)ifs.seekg(0, std::ios::beg);
  (void)ifs.read(reinterpret_cast<char *>(buffer.MutableData()), static_cast<std::streamsize>(size));
  ifs.close();

  return buffer;
}

Key::Key(const char *dec_key, size_t key_len) {
  len = 0;
  if (key_len >= max_key_len) {
    MS_LOG(ERROR) << "Invalid key len " << key_len << " is more than max key len " << max_key_len;
    return;
  }

  auto sec_ret = memcpy_s(key, max_key_len, dec_key, key_len);
  if (sec_ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed, src_len = " << key_len << ", dst_len = " << max_key_len << ", ret = " << sec_ret;
    return;
  }

  len = key_len;
}

Status Serialization::Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph,
                           const Key &dec_key, const std::vector<char> &dec_mode) {
  std::stringstream err_msg;
  if (graph == nullptr) {
    err_msg << "Output args graph is nullptr.";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kMEInvalidInput, err_msg.str());
  }
  if (model_type == kMindIR) {
    FuncGraphPtr anf_graph = nullptr;
    try {
      if (dec_key.len > dec_key.max_key_len) {
        err_msg << "The key length exceeds maximum length: " << dec_key.max_key_len;
        MS_LOG(ERROR) << err_msg.str();
        return Status(kMEInvalidInput, err_msg.str());
      } else if (dec_key.len == 0) {
        if (IsCipherFile(reinterpret_cast<const unsigned char *>(model_data))) {
          err_msg << "Load model failed. The model_data may be encrypted, please pass in correct key.";
          MS_LOG(ERROR) << err_msg.str();
          return Status(kMEInvalidInput, err_msg.str());
        } else {
          anf_graph = ConvertStreamToFuncGraph(reinterpret_cast<const char *>(model_data), data_size, true);
        }
      } else {
        size_t plain_data_size;
        auto plain_data = mindspore::Decrypt(&plain_data_size, reinterpret_cast<const unsigned char *>(model_data),
                                             data_size, dec_key.key, dec_key.len, CharToString(dec_mode));
        if (plain_data == nullptr) {
          err_msg << "Load model failed. Please check the valid of dec_key and dec_mode.";
          MS_LOG(ERROR) << err_msg.str();
          return Status(kMEInvalidInput, err_msg.str());
        }
        anf_graph = ConvertStreamToFuncGraph(reinterpret_cast<const char *>(plain_data.get()), plain_data_size, true);
      }
    } catch (const std::exception &e) {
      err_msg << "Load model failed. Please check the valid of dec_key and dec_mode." << e.what();
      MS_LOG(ERROR) << err_msg.str();
      return Status(kMEInvalidInput, err_msg.str());
    }

    *graph = Graph(std::make_shared<Graph::GraphData>(anf_graph, kMindIR));
    return kSuccess;
  } else if (model_type == kOM) {
    *graph = Graph(std::make_shared<Graph::GraphData>(Buffer(model_data, data_size), kOM));
    return kSuccess;
  }

  err_msg << "Unsupported ModelType " << model_type;
  MS_LOG(ERROR) << err_msg.str();
  return Status(kMEInvalidInput, err_msg.str());
}

Status Serialization::Load(const std::vector<char> &file, ModelType model_type, Graph *graph) {
  return Load(file, model_type, graph, Key{}, StringToChar(kDecModeAesGcm));
}

Status Serialization::Load(const std::vector<char> &file, ModelType model_type, Graph *graph, const Key &dec_key,
                           const std::vector<char> &dec_mode) {
  std::stringstream err_msg;
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Output args graph is nullptr.";
    return Status(kMEInvalidInput, "Output args graph is nullptr.");
  }

  std::string file_path;
  auto status = RealPath(CharToString(file), &file_path);
  if (status != kSuccess) {
    MS_LOG(ERROR) << status.GetErrDescription();
    return status;
  }

  if (model_type == kMindIR) {
    FuncGraphPtr anf_graph;
    if (dec_key.len > dec_key.max_key_len) {
      err_msg << "The key length exceeds maximum length: " << dec_key.max_key_len;
      MS_LOG(ERROR) << err_msg.str();
      return Status(kMEInvalidInput, err_msg.str());
    } else if (dec_key.len == 0 && IsCipherFile(file_path)) {
      err_msg << "Load model failed. The file may be encrypted, please pass in correct key.";
      MS_LOG(ERROR) << err_msg.str();
      return Status(kMEInvalidInput, err_msg.str());
    }
    MindIRLoader mindir_loader(true, dec_key.len == 0 ? nullptr : dec_key.key, dec_key.len, CharToString(dec_mode),
                               false);
    anf_graph = mindir_loader.LoadMindIR(file_path);
    if (anf_graph == nullptr) {
      err_msg << "Load model failed.";
      MS_LOG(ERROR) << err_msg.str();
      return Status(kMEInvalidInput, err_msg.str());
    }
    auto graph_data = std::make_shared<Graph::GraphData>(anf_graph, kMindIR);
#if !defined(_WIN32) && !defined(_WIN64)
    // Config preprocessor, temporary way to let mindspore.so depends on _c_dataengine
    std::vector<std::string> preprocessor = mindir_loader.LoadPreprocess(file_path);
    if (!preprocessor.empty()) {
      std::string dataengine_so_path;
      Status dlret = DLSoPath("libmindspore.so", "_c_dataengine", &dataengine_so_path);
      CHECK_FAIL_AND_RELEASE(dlret, nullptr, "Parse dataengine_so failed: " + dlret.GetErrDescription());

      void *handle = nullptr;
      void *function = nullptr;
      dlret = DLSoOpen(dataengine_so_path, "ParseMindIRPreprocess_C", &handle, &function);
      CHECK_FAIL_AND_RELEASE(dlret, handle, "Parse ParseMindIRPreprocess_C failed: " + dlret.GetErrDescription());
      auto ParseMindIRPreprocessFun =
        (void (*)(const std::vector<std::string> &, std::vector<std::shared_ptr<mindspore::dataset::Execute>> *,
                  Status *))(function);

      std::vector<std::shared_ptr<dataset::Execute>> data_graph;
      ParseMindIRPreprocessFun(preprocessor, &data_graph, &dlret);
      CHECK_FAIL_AND_RELEASE(dlret, handle, "Load preprocess failed: " + dlret.GetErrDescription());
      DLSoClose(handle);
      if (!data_graph.empty()) {
        graph_data->SetPreprocess(data_graph);
      }
    }
#endif
    *graph = Graph(graph_data);
    return kSuccess;
  } else if (model_type == kOM) {
    Buffer data = ReadFile(file_path);
    if (data.Data() == nullptr) {
      err_msg << "Read file " << file_path << " failed.";
      MS_LOG(ERROR) << err_msg.str();
      return Status(kMEInvalidInput, err_msg.str());
    }
    *graph = Graph(std::make_shared<Graph::GraphData>(data, kOM));
    return kSuccess;
  }

  err_msg << "Unsupported ModelType " << model_type;
  MS_LOG(ERROR) << err_msg.str();
  return Status(kMEInvalidInput, err_msg.str());
}

Status Serialization::Load(const std::vector<std::vector<char>> &files, ModelType model_type,
                           std::vector<Graph> *graphs, const Key &dec_key, const std::vector<char> &dec_mode) {
  std::stringstream err_msg;
  if (graphs == nullptr) {
    MS_LOG(ERROR) << "Output args graph is nullptr.";
    return Status(kMEInvalidInput, "Output args graph is nullptr.");
  }

  if (files.size() == 1) {
    std::vector<Graph> result(files.size());
    auto ret = Load(files[0], model_type, &result[0], dec_key, dec_mode);
    *graphs = std::move(result);
    return ret;
  }

  std::vector<std::string> files_path;
  for (const auto &file : files) {
    std::string file_path;
    auto status = RealPath(CharToString(file), &file_path);
    if (status != kSuccess) {
      MS_LOG(ERROR) << status.GetErrDescription();
      return status;
    }
    files_path.emplace_back(std::move(file_path));
  }

  if (model_type == kMindIR) {
    if (dec_key.len > dec_key.max_key_len) {
      err_msg << "The key length exceeds maximum length: " << dec_key.max_key_len;
      MS_LOG(ERROR) << err_msg.str();
      return Status(kMEInvalidInput, err_msg.str());
    }
    MindIRLoader mindir_loader(true, dec_key.len == 0 ? nullptr : dec_key.key, dec_key.len, CharToString(dec_mode),
                               true);
    auto anf_graphs = mindir_loader.LoadMindIRs(files_path);
    if (anf_graphs.size() != files_path.size()) {
      err_msg << "Load model failed, " << files_path.size() << " files got " << anf_graphs.size() << " graphs.";
      MS_LOG(ERROR) << err_msg.str();
      return Status(kMEInvalidInput, err_msg.str());
    }
#if !defined(_WIN32) && !defined(_WIN64)
    // Dataset so loading
    std::string dataengine_so_path;
    Status dlret = DLSoPath("libmindspore.so", "_c_dataengine", &dataengine_so_path);
    CHECK_FAIL_AND_RELEASE(dlret, nullptr, "Parse dataengine_so failed: " + dlret.GetErrDescription());

    void *handle = nullptr;
    void *function = nullptr;
    dlret = DLSoOpen(dataengine_so_path, "ParseMindIRPreprocess_C", &handle, &function);
    CHECK_FAIL_AND_RELEASE(dlret, handle, "Parse ParseMindIRPreprocess_C failed: " + dlret.GetErrDescription());

    auto ParseMindIRPreprocessFun =
      (void (*)(const std::vector<std::string> &, std::vector<std::shared_ptr<mindspore::dataset::Execute>> *,
                Status *))(function);
#endif
    std::vector<Graph> results;
    for (size_t i = 0; i < anf_graphs.size(); ++i) {
      if (anf_graphs[i] == nullptr) {
        if (dec_key.len == 0 && IsCipherFile(files_path[i])) {
          err_msg << "Load model failed. The file " << files_path[i] << " be encrypted, please pass in correct key.";
        } else {
          err_msg << "Load model " << files_path[i] << " failed.";
        }
        MS_LOG(ERROR) << err_msg.str();
        return Status(kMEInvalidInput, err_msg.str());
      }
      auto graph_data = std::make_shared<Graph::GraphData>(anf_graphs[i], kMindIR);
#if !defined(_WIN32) && !defined(_WIN64)
      // Config preprocessor, temporary way to let mindspore.so depends on _c_dataengine
      std::vector<std::string> preprocessor = mindir_loader.LoadPreprocess(files_path[i]);
      if (!preprocessor.empty()) {
        std::vector<std::shared_ptr<dataset::Execute>> data_graph;
        ParseMindIRPreprocessFun(preprocessor, &data_graph, &dlret);
        CHECK_FAIL_AND_RELEASE(dlret, handle, "Load preprocess failed: " + dlret.GetErrDescription());
        if (!data_graph.empty()) {
          graph_data->SetPreprocess(data_graph);
        }
      }
#endif
      results.emplace_back(graph_data);
    }
#if !defined(_WIN32) && !defined(_WIN64)
    // Dataset so release
    DLSoClose(handle);
#endif
    *graphs = std::move(results);
    return kSuccess;
  }

  err_msg << "Unsupported ModelType " << model_type;
  MS_LOG(ERROR) << err_msg.str();
  return Status(kMEInvalidInput, err_msg.str());
}

Status Serialization::SetParameters(const std::map<std::vector<char>, Buffer> &, Model *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::ExportModel(const Model &, ModelType, Buffer *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::ExportModel(const Model &, ModelType, const std::vector<char> &, QuantizationType, bool,
                                  const std::vector<std::vector<char>> &output_tensor_name) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}
}  // namespace mindspore
