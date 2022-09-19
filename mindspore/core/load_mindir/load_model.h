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
#ifndef MINDSPORE_CORE_LOAD_MODEL_H
#define MINDSPORE_CORE_LOAD_MODEL_H

#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ir/func_graph.h"
#include "load_mindir/anf_model_parser.h"
#include "proto/mind_ir.pb.h"

namespace mindspore {
class MS_CORE_API MindIRLoader {
 public:
  MindIRLoader() = default;
  MindIRLoader(bool is_lite, const unsigned char *dec_key, const size_t key_len, const std::string &dec_mode,
               bool inc_load)
      : is_lite_(is_lite), dec_key_(dec_key), key_len_(key_len), dec_mode_(dec_mode), inc_load_(inc_load) {}
  ~MindIRLoader() = default;

  void set_has_parallel_info(bool has_parallel_info) { has_parallel_info_ = has_parallel_info; }
  void set_weights_value_map(const std::map<string, ValuePtr> &weights_value_map) {
    weights_value_map_ = weights_value_map;
  }
  const LayoutMap &layout_map() const { return layout_map_; }
  void InitModelParser(MSANFModelParser *model_parser);
  FuncGraphPtr LoadMindIR(const void *buffer, const size_t &size);
  FuncGraphPtr LoadMindIR(const std::string &file_name);
  std::vector<FuncGraphPtr> LoadMindIRs(const std::vector<std::string> &file_names);
  std::vector<std::string> LoadPreprocess(const std::string &file_name);
  void SetIsLite(bool is_lite) { is_lite_ = is_lite; }

 private:
  bool ParseModelProto(mind_ir::ModelProto *model, const std::string &path);
  bool ParseGraphProto(mind_ir::GraphProto *graph, const std::string &path);
  bool is_lite_ = false;
  const unsigned char *dec_key_ = nullptr;
  size_t key_len_ = 0;
  std::string dec_mode_ = std::string("AES-GCM");
  bool inc_load_ = false;
  std::map<string, ValuePtr> weights_value_map_;
  bool has_parallel_info_ = false;
  LayoutMap layout_map_;
};

std::shared_ptr<std::vector<char>> ReadProtoFile(const std::string &file);
MS_CORE_API FuncGraphPtr ConvertStreamToFuncGraph(const char *buf, const size_t buf_size, bool is_lite = false);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_LOAD_MODEL_H
