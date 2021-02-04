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

#include "load_mindir/load_model.h"
#include <memory>
#include <algorithm>
#include <fstream>

#include "load_mindir/anf_model_parser.h"

using std::string;
using std::vector;

namespace mindspore {
std::shared_ptr<std::vector<char>> ReadProtoFile(const std::string &file) {
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }

  char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, file.c_str(), PATH_MAX) == nullptr) {
    MS_LOG(ERROR) << "Get realpath failed, mind ir file is" << file;
    return nullptr;
  }
#else
  if (realpath(file.c_str(), real_path) == nullptr) {
    MS_LOG(ERROR) << "Get realpath failed, mind ir file is" << file;
    return nullptr;
  }
#endif

  std::ifstream ifs(real_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << "open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  std::shared_ptr<std::vector<char>> buf(new (std::nothrow) std::vector<char>(size));
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << real_path;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf->data(), size);
  ifs.close();

  return buf;
}

std::shared_ptr<FuncGraph> LoadMindIR(const std::string &file_name, bool is_lite) {
  auto graphBuf = ReadProtoFile(file_name);
  if (graphBuf == nullptr) {
    MS_LOG(ERROR) << "Read Mind IR failed, file name is " << file_name.c_str();
    return nullptr;
  }

  try {
    auto graph = ConvertStreamToFuncGraph(graphBuf->data(), graphBuf->size(), is_lite);
    return graph;
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "Load graph model failed, file name is " << file_name.c_str();
    return nullptr;
  }
}

std::shared_ptr<FuncGraph> ConvertStreamToFuncGraph(const char *buf, const size_t buf_size, bool is_lite) {
  MS_EXCEPTION_IF_NULL(buf);
  std::string str((const char *)buf, buf_size);
  mind_ir::ModelProto model_;
  if (!model_.ParseFromString(str)) {
    MS_LOG(ERROR) << "Parse model from buffer fail!";
  }
  MSANFModelParser model_parser;
  if (is_lite) {
    model_parser.SetLite();
  }
  FuncGraphPtr dstgraph_ptr = model_parser.Parse(model_);
  return dstgraph_ptr;
}

}  // namespace mindspore
