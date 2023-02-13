/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_INCLUDE_MODEL_H_
#define MINDSPORE_LITE_INCLUDE_MODEL_H_

#include <memory>
#include <string>
#include <vector>
#include "include/api/visible.h"

namespace mindspore {
namespace schema {
struct Tensor;
}  // namespace schema
namespace lite {
typedef enum { ModelType_MSLite, ModelType_MindIR } LiteModelType;

// LiteGraph can be considered as a light weight and subset of FuncGraph, it can not support the advanced expression of
// FuncGraph, e.g., non-tail recursive.
struct MS_API LiteGraph {
  struct Node {
    std::string name_;
    std::string op_type_;
    int node_type_;
    const void *primitive_ = nullptr;
    std::shared_ptr<void> base_operator_ = nullptr;
    std::vector<uint32_t> input_indices_;
    std::vector<uint32_t> output_indices_;
    int quant_type_;
    int device_type_ = -1;
  };
  struct SubGraph {
    std::string name_;
    std::vector<uint32_t> input_indices_;
    std::vector<uint32_t> output_indices_;
    std::vector<uint32_t> node_indices_;
    std::vector<uint32_t> tensor_indices_;
  };
  std::string name_;
  std::string version_;
  std::vector<uint32_t> input_indices_;
  std::vector<uint32_t> output_indices_;
  std::vector<mindspore::schema::Tensor *> all_tensors_;
  std::vector<Node *> all_nodes_;
  std::vector<SubGraph *> sub_graphs_;
#ifdef ENABLE_MODEL_OBF
  std::vector<uint32_t> all_prims_type_;
  std::vector<uint32_t> all_nodes_stat_;
  bool model_obfuscated_ = false;
  std::vector<unsigned char *> deobf_prims_;
#endif
};
struct MS_API Model {
  LiteGraph graph_;
  char *buf = nullptr;
  size_t buf_size_ = 0;
  LiteModelType model_type_ = mindspore::lite::ModelType_MSLite;

  /// \brief Static method to create a Model pointer.
  static Model *Import(const char *model_buf, size_t size);

  /// \brief Static method to create a Model pointer.
  static Model *Import(const char *filename);

  /// \brief  method to export model to file.
  static int Export(Model *model, const char *filename);

  /// \brief  method to export model to buffer.
  static int Export(Model *model, char *buf, size_t *size);

  /// \brief Free meta graph temporary buffer
  virtual void Free() = 0;

  /// \brief Free all temporary buffer.EG: nodes in the model.
  virtual void Destroy() = 0;

  /// \brief Model destruct, free all memory
  virtual ~Model() = default;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_MODEL_H_
