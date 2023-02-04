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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_META_GRAPH_SERIALIZER_H_
#define MINDSPORE_LITE_TOOLS_COMMON_META_GRAPH_SERIALIZER_H_

#include <fstream>
#include <string>
#include "flatbuffers/flatbuffers.h"
#include "schema/inner/model_generated.h"
#include "utils/crypto.h"

namespace mindspore::lite {
class MetaGraphSerializer {
 public:
  // get metagraph packed buffer
  static uint8_t *GetMetaGraphPackedBuff(flatbuffers::FlatBufferBuilder *builder, const schema::MetaGraphT &graph,
                                         size_t *data_size);

  // save serialized fb model
  static int Save(const schema::MetaGraphT &graph, const std::string &output_path, const Byte *key = {},
                  const size_t key_len = 0, const std::string &enc_mode = "");
  static int Save(const schema::MetaGraphT &graph, const std::string &output_path, size_t *size, const Byte *key = {},
                  const size_t key_len = 0, const std::string &enc_mode = "");

 private:
  MetaGraphSerializer() = default;

  virtual ~MetaGraphSerializer();

  bool InitPath(const std::string &real_output_path);

  bool Init(const schema::MetaGraphT &graph, bool save_together = true);

  schema::ExternalDataT *AddExternalData(const char *data, size_t size);

  bool ExtraAndSerializeModelWeight(const schema::MetaGraphT &graph);

  bool SerializeModelAndUpdateWeight(const schema::MetaGraphT &meta_graphT, const Byte *key, const size_t key_len,
                                     const std::string &enc_mode, size_t *size = 0);

  bool SerializeModel(const void *content, size_t size, const Byte *key, const size_t key_len,
                      const std::string &enc_mode);

  int64_t cur_offset_ = 0;
  std::string save_path_;
  std::string model_name_;
  std::string save_model_path_;
  std::string save_data_path_;
  std::fstream *model_fs_ = nullptr;
  std::fstream *data_fs_ = nullptr;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_COMMON_META_GRAPH_SERIALIZER_H_
