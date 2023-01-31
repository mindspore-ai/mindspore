/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include <string>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "load_mindir/load_model.h"
#include "utils/crypto.h"
#include "utils/os.h"
#include "include/common/debug/common.h"

using std::string;
using std::vector;

namespace mindspore {
bool get_all_files(const std::string &dir_in, std::vector<std::string> *files) {
  if (dir_in.empty()) {
    return false;
  }
  struct stat s;
  int ret = stat(dir_in.c_str(), &s);
  if (ret != 0) {
    MS_LOG(ERROR) << "stat error, ret is : " << ret;
    return false;
  }
  if (!S_ISDIR(s.st_mode)) {
    return false;
  }
  DIR *open_dir = opendir(dir_in.c_str());
  if (open_dir == nullptr) {
    MS_LOG(EXCEPTION) << "open dir " << dir_in.c_str() << " failed";
  }
  dirent *p = nullptr;
  while ((p = readdir(open_dir)) != nullptr) {
    struct stat st;
    if (p->d_name[0] != '.') {
      std::string name = dir_in + std::string("/") + std::string(p->d_name);
      ret = stat(name.c_str(), &st);
      if (ret != 0) {
        MS_LOG(ERROR) << "stat error, ret is : " << ret;
        closedir(open_dir);
        return false;
      }
      if (S_ISDIR(st.st_mode)) {
        if (!get_all_files(name, files)) {
          MS_LOG(ERROR) << "Get files failed, ret is : " << ret;
          closedir(open_dir);
          return false;
        }
      } else if (S_ISREG(st.st_mode)) {
        files->push_back(name);
      }
    }
  }
  closedir(open_dir);
  return true;
}

int endsWith(const string s, const string sub) { return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0; }

bool MindIRLoader::ParseModelProto(mind_ir::ModelProto *model, const std::string &path) {
  if (dec_key_ != nullptr) {
    size_t plain_len;
    auto plain_data = Decrypt(&plain_len, path, dec_key_, key_len_, dec_mode_);
    if (plain_data == nullptr) {
      MS_LOG(ERROR)
        << "Decrypt MindIR file failed, please check the correctness of the dec_key or dec_mode or the file integrity.";
      return false;
    }
    if (!model->ParseFromArray(reinterpret_cast<char *>(plain_data.get()), static_cast<int32_t>(plain_len))) {
      MS_LOG(ERROR) << "Load MindIR file failed, please check the correctness of the file, dec_key or dec_mode.";
      return false;
    }
  } else {
    std::fstream input_graph(path, std::ios::in | std::ios::binary);
    if (!input_graph || !model->ParseFromIstream(&input_graph)) {
      MS_LOG(ERROR) << "Load MindIR file failed, please check the correctness of the file.";
      return false;
    }
  }
  return true;
}

bool MindIRLoader::ParseGraphProto(mind_ir::GraphProto *graph, const std::string &path) {
  if (dec_key_ != nullptr) {
    size_t plain_len;
    auto plain_data = Decrypt(&plain_len, path, dec_key_, key_len_, dec_mode_);
    if (plain_data == nullptr) {
      MS_LOG(ERROR)
        << "Decrypt MindIR file failed, please check the correctness of the dec_key or dec_mode or the file integrity.";
      return false;
    }
    if (!graph->ParseFromArray(reinterpret_cast<char *>(plain_data.get()), static_cast<int32_t>(plain_len))) {
      MS_LOG(ERROR) << "Load variable file failed, please check the correctness of the mindir's variable file, "
                       "dec_key or dec_mode";
      return false;
    }
  } else {
    std::fstream input_param(path, std::ios::in | std::ios::binary);
    if (!input_param || !graph->ParseFromIstream(&input_param)) {
      MS_LOG(ERROR) << "Load variable file failed, please check the correctness of mindir's variable file.";
      return false;
    }
  }
  return true;
}

std::vector<std::string> MindIRLoader::LoadPreprocess(const std::string &file_name) {
  if (file_name.length() > PATH_MAX) {
    MS_LOG(ERROR) << "The length of the file name exceeds the limit.";
    return {};
  }
  char abs_path_buff[PATH_MAX];

#ifdef _WIN32
  _fullpath(abs_path_buff, file_name.c_str(), PATH_MAX);
#else
  if (!realpath(file_name.c_str(), abs_path_buff)) {
    MS_LOG(ERROR) << "Load MindIR get absolute path failed";
  }
#endif

  // Read graph
  mind_ir::ModelProto origin_model;
  if (!ParseModelProto(&origin_model, std::string(abs_path_buff))) {
    MS_LOG(ERROR) << "Load MindIR file failed, please check the correctness of the file.";
    return {};
  }

  // Read dataset preprocessor
  auto preprocessor = origin_model.preprocessor();

  // Separate columns and parse
  std::vector<std::string> input_columns;
  for (auto i = 0; i < preprocessor.op_size(); i++) {
    std::string column = preprocessor.op()[i].input_columns();
    if (std::find(input_columns.begin(), input_columns.end(), column) == input_columns.end()) {
      input_columns.push_back(column);
    }
  }

  // Each column has one string to indicate its preprocess behaviour
  std::vector<std::string> map_jsons;
  for (std::string &column : input_columns) {
    nlohmann::json dataset_json;
    nlohmann::json child_dataset_json;
    for (auto i = preprocessor.op_size() - 1; i >= 0; i--) {
      if (preprocessor.op()[i].input_columns() == column) {
        child_dataset_json["input_columns"] = nlohmann::json::parse(preprocessor.op()[i].input_columns());
        child_dataset_json["op_type"] = nlohmann::json::parse(preprocessor.op()[i].op_type());
        child_dataset_json["operations"] = nlohmann::json::parse(preprocessor.op()[i].operations());
        child_dataset_json["output_columns"] = nlohmann::json::parse(preprocessor.op()[i].output_columns());
        child_dataset_json["offload"] = preprocessor.op()[i].offload();

        dataset_json["children"] = child_dataset_json;
        child_dataset_json = dataset_json;
      }
    }
    map_jsons.push_back(dataset_json["children"].dump());
  }
  return map_jsons;
}

std::vector<FuncGraphPtr> MindIRLoader::LoadMindIRs(const std::vector<std::string> &file_names) {
  std::vector<FuncGraphPtr> funcgraph_vec;
  MS_LOG(DEBUG) << "Load multiple MindIR files.";
  for (const auto &file_name : file_names) {
    MS_LOG(DEBUG) << "Load " << file_name;
    funcgraph_vec.push_back(LoadMindIR(file_name));
  }
  return funcgraph_vec;
}

void MindIRLoader::InitModelParser(MSANFModelParser *model_parser) {
  model_parser->SetMindIRDecKey(dec_key_);
  model_parser->SetMindIRKeySize(key_len_);
  model_parser->SetMindIRDecMode(dec_mode_);

  if (!inc_load_) {
    MSANFModelParser::LoadTensorMapClear();
  } else {
    model_parser->SetIncLoad();
  }
  if (is_lite_) {
    model_parser->SetLite();
  }
}

FuncGraphPtr MindIRLoader::LoadMindIR(const void *buffer, const size_t &size) {
  /* mindir -> func_graph
   * only support lite */
  mind_ir::ModelProto model;
  auto ret = model.ParseFromArray(buffer, SizeToInt(size));
  if (!ret) {
    MS_LOG(ERROR) << "ParseFromArray failed.";
    return nullptr;
  }

  MSANFModelParser model_parser;
  InitModelParser(&model_parser);
  FuncGraphPtr func_graph = model_parser.Parse(model);

  return func_graph;
}

FuncGraphPtr MindIRLoader::LoadMindIR(const std::string &file_name) {
  if (file_name.length() > PATH_MAX) {
    MS_LOG(ERROR) << "The length of the file name exceeds the limit.";
    return nullptr;
  }
  char abs_path_buff[PATH_MAX];
  vector<string> files;

#ifdef _WIN32
  _fullpath(abs_path_buff, file_name.c_str(), PATH_MAX);
#else
  if (!realpath(file_name.c_str(), abs_path_buff)) {
    MS_LOG(ERROR) << "Load MindIR get absolute path of " << file_name << " failed, errno is: " << ErrnoToString(errno);
  }
#endif
  // Read graph
  mind_ir::ModelProto origin_model;
  if (!ParseModelProto(&origin_model, std::string(abs_path_buff))) {
    return nullptr;
  }
  // Load parameter into graph
  if (endsWith(std::string(abs_path_buff), "_graph.mindir") && (origin_model.graph().parameter_size() == 0)) {
    if (strlen(abs_path_buff) < strlen("graph.mindir")) {
      MS_LOG(ERROR) << "The abs_path_buff length is less than 'graph.mindir'.";
      return nullptr;
    }
    size_t path_len = strlen(abs_path_buff) - strlen("graph.mindir");
    string var_path = std::string(abs_path_buff).substr(0, path_len);
    var_path += "variables";
    std::ifstream ifs(var_path);
    if (ifs.good()) {
      MS_LOG(DEBUG) << "MindIR file has variables path, load parameter into graph.";
      (void)get_all_files(var_path, &files);
    } else {
      MS_LOG(ERROR) << "Load graph's variable folder failed, please check the correctness of variable folder.";
      return nullptr;
    }

    size_t file_size = files.size();
    mind_ir::GraphProto *mod_graph = origin_model.mutable_graph();
    for (size_t file_index = 0; file_index < file_size; file_index++) {
      mind_ir::GraphProto param_graph;
      if (!ParseGraphProto(&param_graph, files[file_index])) {
        return nullptr;
      }

      for (int param_index = 0; param_index < param_graph.parameter_size(); param_index++) {
        mind_ir::TensorProto *param_proto = mod_graph->add_parameter();
        param_proto->set_name(param_graph.parameter(param_index).name());
        param_proto->set_data_type(param_graph.parameter(param_index).data_type());
        param_proto->set_raw_data(param_graph.parameter(param_index).raw_data());
        param_proto->set_compression_type(param_graph.parameter(param_index).compression_type());
        for (const auto &dim : param_graph.parameter(param_index).dims()) {
          param_proto->add_dims(dim);
        }
      }
    }
  }

  MSANFModelParser model_parser;

  auto mindir_path = std::string(abs_path_buff);
  model_parser.SetMindIRPath(mindir_path.substr(0, mindir_path.rfind("/")));
  InitModelParser(&model_parser);
  FuncGraphPtr dstgraph_ptr = model_parser.Parse(origin_model, weights_value_map_);
  if (has_parallel_info_) {
    layout_map_ = model_parser.ParseLayout(origin_model);
  }
  return dstgraph_ptr;
}

FuncGraphPtr MindIRLoader::LoadMindIR(const void *buffer, const size_t &size, const std::string &mindir_path) {
  mind_ir::ModelProto model;
  auto ret = model.ParseFromArray(buffer, SizeToInt(size));
  if (!ret) {
    MS_LOG(ERROR) << "ParseFromArray failed.";
    return nullptr;
  }

  MSANFModelParser model_parser;
  InitModelParser(&model_parser);
  model_parser.SetMindIRPath(mindir_path);
  FuncGraphPtr func_graph = model_parser.Parse(model);
  return func_graph;
}

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

FuncGraphPtr ConvertStreamToFuncGraph(const char *buf, const size_t buf_size, bool is_lite) {
  MS_EXCEPTION_IF_NULL(buf);
  std::string str(buf, buf_size);
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
