/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_CONTEXT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_CONTEXT_H_
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include "src/tensor.h"

namespace mindspore::lite::micro {
class CoderContext {
 public:
  explicit CoderContext(int model_index);

  ~CoderContext() = default;

  std::vector<std::string> init_contents() const { return initialContent_; }

  void set_code_blocks(const std::vector<std::string> &code_block) { code_blocks_ = code_block; }
  std::vector<std::string> code_blocks() const { return code_blocks_; }

  void set_inference_blocks(const std::vector<std::string> &inference_blocks) { inference_blocks_ = inference_blocks; }
  std::vector<std::string> inference_blocks() const { return inference_blocks_; }

  void set_train_blocks(const std::vector<std::string> &train_blocks) { train_blocks_ = train_blocks; }
  std::vector<std::string> train_blocks() const { return train_blocks_; }

  void set_global_code_blocks(const std::vector<std::string> &global_code_blocks) {
    global_code_blocks_ = global_code_blocks;
  }
  std::vector<std::string> global_code_blocks() const { return global_code_blocks_; }

  void set_after_inference_code_blocks(const std::vector<std::string> &after_inference_code_blocks) {
    after_inference_code_blocks_ = after_inference_code_blocks;
  }
  std::vector<std::string> after_inference_code_blocks() const { return after_inference_code_blocks_; }

  void set_weight_buffer_size_code_blocks(const std::vector<std::string> &weight_buffer_size_code_blocks) {
    weight_buffer_size_code_blocks_ = weight_buffer_size_code_blocks;
  }
  std::vector<std::string> weight_buffer_size_code_blocks() const { return weight_buffer_size_code_blocks_; }

  void set_weight_buffer_size(size_t weight_buffer_size) { weight_buffer_size_ = weight_buffer_size; }

  size_t weight_buffer_size() const { return weight_buffer_size_; }

  void set_tensor_map(const std::map<Tensor *, std::string> &tensor_map) {
    tensors_map_.insert(tensor_map.begin(), tensor_map.end());
  }
  std::map<Tensor *, std::string> tensors_map() const { return tensors_map_; }
  void set_saved_weights(const std::map<std::string, Tensor *> &saved_weights) { saved_weights_ = saved_weights; }
  std::map<std::string, Tensor *> saved_weights() const { return saved_weights_; }

  void set_total_buffer_size(size_t size) { total_buffer_size_ = size; }
  size_t total_buffer_size() const { return total_buffer_size_; }

  void set_graph_inputs(const std::vector<Tensor *> &graph_inputs) { graph_inputs_ = graph_inputs; }
  void set_graph_outputs(const std::vector<Tensor *> &graph_outputs) { graph_outputs_ = graph_outputs; }
  void set_graph_eval_outputs(const std::vector<Tensor *> &graph_eval_outputs) {
    graph_eval_outputs_ = graph_eval_outputs;
  }
  void set_graph_train_outputs(const std::vector<Tensor *> &graph_train_outputs) {
    graph_train_outputs_ = graph_train_outputs;
  }

  std::vector<Tensor *> graph_inputs() const { return graph_inputs_; }
  std::vector<Tensor *> graph_outputs() const { return graph_outputs_; }
  std::vector<Tensor *> graph_eval_outputs() const { return graph_eval_outputs_; }
  std::vector<Tensor *> graph_train_outputs() const { return graph_train_outputs_; }

  std::string input_name() { return input_name_; }
  std::string output_name() { return output_name_; }
  std::string buffer_name() { return buffer_name_; }
  std::string weight_name() { return weight_name_; }
  std::string weight_offset_name() { return pack_weight_offset_name_; }
  std::string weight_size_name() { return pack_weight_size_name_; }

  void AppendCode(const std::string &codeBlock);

  void AppendInitCode(const std::string &codeBlock);

  void AppendInitWeightSizeCode(const std::string &codeBlock);

  void AppendInitWeightSizeCode(size_t w_buf_size);

  std::vector<std::string> GetInitWeightSizeCode() const;

  int GetCurModelIndex() { return model_index_; }

  std::set<std::string> c_files() const { return c_files_; }
  void set_c_files(const std::set<std::string> &files) { c_files_.insert(files.begin(), files.end()); }

  std::set<std::string> h_files() const { return h_files_; }
  void set_h_files(const std::set<std::string> &files) { h_files_.insert(files.begin(), files.end()); }

  std::set<std::string> asm_files() const { return asm_files_; }
  void set_asm_files(const std::set<std::string> &files) { asm_files_.insert(files.begin(), files.end()); }

 private:
  std::vector<Tensor *> graph_inputs_;
  std::vector<Tensor *> graph_outputs_;
  std::vector<Tensor *> graph_eval_outputs_;
  std::vector<Tensor *> graph_train_outputs_;
  // primitive const tensors, parsed from model, without packed.
  std::map<std::string, Tensor *> saved_weights_;
  // all tensors, include parsed from model and packed tensors.
  std::map<Tensor *, std::string> tensors_map_;
  // workspace's size.
  size_t total_buffer_size_{0};
  // model's input tensor data's address.
  std::string input_name_;
  // model's output tensor's address
  std::string output_name_;
  // the address of workspace, use for inference or train.
  std::string buffer_name_;
  // model's weight tensors' address.
  std::string weight_name_;
  // model's pack weight tensor offset name
  std::string pack_weight_offset_name_;
  // model's pack weight tensor size name
  std::string pack_weight_size_name_;
  // code blocks store the tensor will be packed runtime
  std::vector<std::string> initialContent_;
  // operator C Lang files list, depended by the net.c. it will be add to CMakeLists.txt
  std::set<std::string> c_files_;
  // when codegen generate the code for ARM64 OR ARM32, we provide server optimized artimetic used the assembly
  // instructions. asm_files store the assembly file names
  std::set<std::string> asm_files_;
  // operator header files
  std::set<std::string> h_files_;

  // net.c's content, include the Inference and Training implementation
  std::vector<std::string> code_blocks_;
  std::vector<std::string> global_code_blocks_;
  std::vector<std::string> train_blocks_;
  std::vector<std::string> inference_blocks_;
  std::vector<std::string> after_inference_code_blocks_;
  std::vector<std::string> weight_buffer_size_code_blocks_;
  std::size_t weight_buffer_size_ = 0;
  int model_index_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_CONTEXT_H_
