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
#ifndef MINDSPORE_CORE_LOAD_MODEL_H
#define MINDSPORE_CORE_LOAD_MODEL_H

#include <vector>
#include <string>
#include <memory>

#include "ir/func_graph.h"
#include "proto/mind_ir.pb.h"

namespace mindspore {
class MindIRLoader {
 public:
  MindIRLoader() = default;
  MindIRLoader(bool is_lite, const unsigned char *dec_key, const size_t key_len, const std::string &dec_mode,
               bool inc_load)
      : is_lite_(is_lite), dec_key_(dec_key), key_len_(key_len), dec_mode_(dec_mode), inc_load_(inc_load) {}
  ~MindIRLoader() = default;

  bool get_need_renormalize() const { return need_renormalize_; }
  void set_need_renormalize(bool need_renormalize) { need_renormalize_ = need_renormalize; }
  std::shared_ptr<FuncGraph> LoadMindIR(const std::string &file_name);
  std::vector<std::shared_ptr<FuncGraph>> LoadMindIRs(const std::vector<std::string> file_names);

 private:
  bool ParseModelProto(mind_ir::ModelProto *model, const std::string &path);
  bool ParseGraphProto(mind_ir::GraphProto *graph, const std::string &path);
  bool is_lite_ = false;
  const unsigned char *dec_key_ = nullptr;
  size_t key_len_ = 0;
  std::string dec_mode_ = std::string("AES-GCM");
  bool inc_load_ = false;
  bool need_renormalize_ = true;
};

std::string LoadPreprocess(const std::string &file_name);
std::shared_ptr<std::vector<char>> ReadProtoFile(const std::string &file);
std::shared_ptr<FuncGraph> ConvertStreamToFuncGraph(const char *buf, const size_t buf_size, bool is_lite = false);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_LOAD_MODEL_H
