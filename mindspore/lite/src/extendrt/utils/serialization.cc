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
#include "extendrt/utils/serialization.h"
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
#include "include/api/types.h"

namespace mindspore::infer {
static mindspore::Status RealPath(const std::string &file, std::string *realpath_str) {
  MS_EXCEPTION_IF_NULL(realpath_str);
  char real_path_mem[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  auto real_path_ret = _fullpath(real_path_mem, common::SafeCStr(file), PATH_MAX);
#else
  auto real_path_ret = realpath(common::SafeCStr(file), real_path_mem);
#endif
  if (real_path_ret == nullptr) {
    return mindspore::Status(kMEInvalidInput, "File: " + file + " does not exist.");
  }
  *realpath_str = real_path_mem;
  return kSuccess;
}

mindspore::Buffer ReadFile(const std::string &file) {
  mindspore::Buffer buffer;
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

mindspore::FuncGraphPtr Serialization::ConvertStreamToFuncGraph(const char *buf, const size_t buf_size, bool is_lite,
                                                                const std::string &mindir_path) {
  MS_EXCEPTION_IF_NULL(buf);
  std::string str(buf, buf_size);
  mind_ir::ModelProto model;
  if (!model.ParseFromString(str)) {
    MS_LOG(ERROR) << "Parse model from buffer fail!";
  }
  mindspore::MSANFModelParser model_parser;
  model_parser.SetMindIRPath(mindir_path);
  if (is_lite) {
    model_parser.SetLite();
  }
  mindspore::FuncGraphPtr dstgraph_ptr = model_parser.Parse(model);
  return dstgraph_ptr;
}

mindspore::Status Serialization::Load(const void *model_data, size_t data_size, mindspore::ModelType model_type,
                                      mindspore::Graph *graph, const mindspore::Key &dec_key,
                                      const std::string &dec_mode, const std::string &mindir_path) {
  std::stringstream err_msg;
  if (graph == nullptr) {
    err_msg << "Output args graph is nullptr.";
    MS_LOG(ERROR) << err_msg.str();
    return mindspore::Status(kMEInvalidInput, err_msg.str());
  }
  if (model_type == kMindIR) {
    mindspore::FuncGraphPtr anf_graph = nullptr;
    try {
      if (dec_key.len > dec_key.max_key_len) {
        err_msg << "The key length exceeds maximum length: " << dec_key.max_key_len;
        MS_LOG(ERROR) << err_msg.str();
        return mindspore::Status(kMEInvalidInput, err_msg.str());
      } else if (dec_key.len == 0) {
        if (mindspore::IsCipherFile(reinterpret_cast<const unsigned char *>(model_data))) {
          err_msg << "Load model failed. The model_data may be encrypted, please pass in correct key.";
          MS_LOG(ERROR) << err_msg.str();
          return mindspore::Status(kMEInvalidInput, err_msg.str());
        } else {
          anf_graph =
            ConvertStreamToFuncGraph(reinterpret_cast<const char *>(model_data), data_size, true, mindir_path);
        }
      } else {
        size_t plain_data_size;
        auto plain_data = mindspore::Decrypt(&plain_data_size, reinterpret_cast<const unsigned char *>(model_data),
                                             data_size, dec_key.key, dec_key.len, dec_mode);
        if (plain_data == nullptr) {
          err_msg << "Load model failed. Please check the valid of dec_key and dec_mode.";
          MS_LOG(ERROR) << err_msg.str();
          return mindspore::Status(kMEInvalidInput, err_msg.str());
        }
        anf_graph = ConvertStreamToFuncGraph(reinterpret_cast<const char *>(plain_data.get()), plain_data_size, true,
                                             mindir_path);
      }
    } catch (const std::exception &e) {
      err_msg << "Load model failed. Please check the valid of dec_key and dec_mode." << e.what();
      MS_LOG(ERROR) << err_msg.str();
      return mindspore::Status(kMEInvalidInput, err_msg.str());
    }

    *graph = mindspore::Graph(std::make_shared<mindspore::Graph::GraphData>(anf_graph, kMindIR));
    return kSuccess;
  } else if (model_type == kOM) {
    *graph =
      mindspore::Graph(std::make_shared<mindspore::Graph::GraphData>(mindspore::Buffer(model_data, data_size), kOM));
    return kSuccess;
  }

  err_msg << "Unsupported ModelType " << model_type;
  MS_LOG(ERROR) << err_msg.str();
  return mindspore::Status(kMEInvalidInput, err_msg.str());
}
}  // namespace mindspore::infer
