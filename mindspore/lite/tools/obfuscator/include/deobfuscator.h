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

#include <vector>
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "include/model.h"
#include "schema/inner/model_generated.h"
#include "include/api/types.h"

#define IV_SIZE 16

namespace mindspore::lite {
struct MS_API DeObfuscator {
  std::vector<uint32_t> junk_tensor_Indices_;
  std::vector<uint32_t> junk_node_Indices_;
  std::vector<uint32_t> masking_values_;
  using PrimTypeVector = std::vector<schema::PrimitiveType>;
  PrimTypeVector all_prims_type_;
  unsigned char *obf_meta_data_;
  uint32_t all_tensor_size_;
  uint32_t all_node_size_;
  uint32_t buf_size_;
  bool with_sub_graph_;
  void Free();
  ~DeObfuscator() { Free(); }
};

MS_API int DeObfuscateModel(Model *model, DeObfuscator *model_deobf);
MS_API bool DeObfuscatePrimitive(const schema::Primitive *src, uint32_t src_node_stat, unsigned char **dst_prim,
                                 schema::PrimitiveType dst_type);
MS_API bool InitModelDeObfuscator(Model *model, DeObfuscator *model_deobf,
                                  const flatbuffers::Vector<uint8_t> *meta_data, size_t node_num);
MS_API bool DeObfuscateTensors(Model *model, DeObfuscator *model_deobf);
MS_API bool DeObfuscateNodes(Model *model, DeObfuscator *model_deobf);
int NodeVerify(const Model *model);
int SubGraphVerify(const Model *model);
MS_API bool ModelVerify(const Model *model);
MS_API int DeObfuscateSubGraph(LiteGraph::SubGraph *subGraph, Model *model, DeObfuscator *model_deobf);
int DeObfuscateNodeIndex(LiteGraph::Node *node, DeObfuscator *model_deobf, uint32_t all_tensors_num);
int DeObfuscateIndex(uint32_t *orig_index, uint32_t modulus, DeObfuscator *model_deobf);

template <typename T = schema::MetaGraph>
MS_API bool IsMetaGraphObfuscated(const T &meta_graph) {
  if (meta_graph.obfMetaData() == nullptr) {
    MS_LOG(INFO) << "obfMetaData is null.";
    return false;
  }
  return true;
}

template <typename T = schema::MetaGraph>
MS_API DeObfuscator *GetModelDeObfuscator(const T &meta_graph, Model *model, size_t buf_size) {
  if (meta_graph.obfMetaData() == nullptr || meta_graph.nodes() == nullptr || meta_graph.allTensors() == nullptr) {
    MS_LOG(ERROR) << "invalid meta_graph!";
    return nullptr;
  }
  auto meta_data = meta_graph.obfMetaData();
  auto *model_deobfuscator = new (std::nothrow) DeObfuscator();
  if (model_deobfuscator == nullptr) {
    MS_LOG(ERROR) << "new model deobfuscator fail!";
    return nullptr;
  }
  model_deobfuscator->buf_size_ = buf_size;
  if (!InitModelDeObfuscator(model, model_deobfuscator, meta_data, meta_graph.nodes()->size())) {
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
