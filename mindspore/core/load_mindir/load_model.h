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
#include "proto/mind_ir.pb.h"

namespace mindspore {
class Layout {
 public:
  Layout() = default;

  const std::vector<int64_t> &get_device_arrangement() const { return device_arrangement_; }
  void set_device_arrangement(const std::vector<int64_t> &device_arrangement) {
    device_arrangement_ = device_arrangement;
  }
  const std::vector<int64_t> &get_tensor_map() const { return tensor_map_; }
  void set_tensor_map(const std::vector<int64_t> &tensor_map) { tensor_map_ = tensor_map; }
  const std::vector<int64_t> &get_slice_shape() const { return slice_shape_; }
  void set_slice_shape(const std::vector<int64_t> &slice_shape) { slice_shape_ = slice_shape; }
  int64_t get_field_size() const { return field_size_; }
  void set_field_size(int64_t field_size) { field_size_ = field_size; }
  bool get_uniform_split() const { return uniform_split_; }
  void set_uniform_split(bool uniform_split) { uniform_split_ = uniform_split; }
  const std::string &get_opt_shard_group() const { return opt_shard_group_; }
  void set_opt_shard_group(const std::string &opt_shard_group) { opt_shard_group_ = opt_shard_group; }

 private:
  std::vector<int64_t> device_arrangement_{};
  std::vector<int64_t> tensor_map_{};
  std::vector<int64_t> slice_shape_{};
  int64_t field_size_ = 0;
  bool uniform_split_ = false;
  std::string opt_shard_group_ = "";
};
using LayoutPtr = std::shared_ptr<Layout>;
using LayoutMap = std::map<string, LayoutPtr>;

class MS_CORE_API MindIRLoader {
 public:
  MindIRLoader() = default;
  MindIRLoader(bool is_lite, const unsigned char *dec_key, const size_t key_len, const std::string &dec_mode,
               bool inc_load)
      : is_lite_(is_lite), dec_key_(dec_key), key_len_(key_len), dec_mode_(dec_mode), inc_load_(inc_load) {}
  ~MindIRLoader() = default;

  bool get_need_renormalize() const { return need_renormalize_; }
  void set_need_renormalize(bool need_renormalize) { need_renormalize_ = need_renormalize; }
  void set_has_parallel_info(bool has_parallel_info) { has_parallel_info_ = has_parallel_info; }
  void set_weights_value_map(const std::map<string, ValuePtr> &weights_value_map) {
    weights_value_map_ = weights_value_map;
  }
  const LayoutMap &layout_map() { return layout_map_; }
  FuncGraphPtr LoadMindIR(const void *buffer, const size_t &size);
  FuncGraphPtr LoadMindIR(const std::string &file_name);
  std::vector<FuncGraphPtr> LoadMindIRs(const std::vector<std::string> file_names);
  std::vector<std::string> LoadPreprocess(const std::string &file_name);

 private:
  bool ParseModelProto(mind_ir::ModelProto *model, const std::string &path);
  bool ParseGraphProto(mind_ir::GraphProto *graph, const std::string &path);
  bool is_lite_ = false;
  const unsigned char *dec_key_ = nullptr;
  size_t key_len_ = 0;
  std::string dec_mode_ = std::string("AES-GCM");
  bool inc_load_ = false;
  bool need_renormalize_ = true;
  std::map<string, ValuePtr> weights_value_map_;
  bool has_parallel_info_ = false;
  LayoutMap layout_map_;
};

std::shared_ptr<std::vector<char>> ReadProtoFile(const std::string &file);
MS_CORE_API FuncGraphPtr ConvertStreamToFuncGraph(const char *buf, const size_t buf_size, bool is_lite = false);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_LOAD_MODEL_H
