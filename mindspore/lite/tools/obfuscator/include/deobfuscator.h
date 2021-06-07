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

#ifndef MINDSPORE_LITE_TOOLS_OBFUSCATOR_INCLUDE_DEOBFUSCATOR_H
#define MINDSPORE_LITE_TOOLS_OBFUSCATOR_INCLUDE_DEOBFUSCATOR_H

#include "src/common/common.h"
#include "include/model.h"

#define IV_SIZE 16

namespace mindspore::lite {
struct DeObfuscator {
  Uint32Vector junk_tensor_Indices_;
  Uint32Vector junk_node_Indices_;
  Uint32Vector masking_values_;
  using PrimTypeVector = Vector<schema::PrimitiveType>;
  PrimTypeVector all_prims_type_;
  unsigned char *obf_meta_data_;
  uint32_t all_tensor_size_;
  uint32_t all_node_size_;
  bool with_sub_graph_;
  void Free();
  ~DeObfuscator() { Free(); }
};

int DeObfuscateModel(Model *model, DeObfuscator *model_deobf);
bool DeObfuscatePrimitive(const schema::Primitive *src, uint32_t src_node_stat, unsigned char **dst_prim,
                          schema::PrimitiveType dst_type);
bool InitModelDeObfuscator(Model *model, DeObfuscator *model_deobf, const flatbuffers::Vector<uint8_t> *meta_data,
                           const flatbuffers::Vector<uint8_t> *decrypt_table, size_t node_num);

template <typename T = schema::MetaGraph>
bool IsMetaGraphObfuscated(const T &meta_graph) {
  if (meta_graph.obfMetaData() == nullptr) {
    MS_LOG(INFO) << "obfMetaData is null.";
    return false;
  }
  return true;
}

template <typename T = schema::MetaGraph>
DeObfuscator *GetModelDeObfuscator(const T &meta_graph, Model *model) {
  auto meta_data = meta_graph.obfMetaData();
  auto *model_deobfuscator = new (std::nothrow) DeObfuscator();
  if (model_deobfuscator == nullptr) {
    MS_LOG(ERROR) << "new model deobfuscator fail!";
    return nullptr;
  }
  if (!InitModelDeObfuscator(model, model_deobfuscator, meta_data, nullptr, meta_graph.nodes()->size())) {
    MS_LOG(ERROR) << "init model deobfuscator fail!";
    delete model_deobfuscator;
    return nullptr;
  }
  model_deobfuscator->all_tensor_size_ = meta_graph.allTensors()->size();
  model_deobfuscator->all_node_size_ = meta_graph.nodes()->size();
  if (meta_graph.subGraph() == nullptr) {
    model_deobfuscator->with_sub_graph_ = false;
  } else {
    model_deobfuscator->with_sub_graph_ = true;
  }
  return model_deobfuscator;
}
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_OBFUSCATOR_INCLUDE_DEOBFUSCATOR_H
