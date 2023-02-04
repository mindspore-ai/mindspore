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

#include "tools/common/meta_graph_serializer.h"
#include <sys/stat.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include "flatbuffers/flatbuffers.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "ir/dtype/type_id.h"
#include "src/common/utils.h"
#include "include/errorcode.h"
#include "securec/include/securec.h"
#include "src/common/file_utils.h"

namespace mindspore::lite {
namespace {
constexpr size_t kModelSizeLimit = static_cast<size_t>(2) * 1024 * 1024 * 1024;
constexpr size_t kExternalDataHeadSize = 4096;
constexpr size_t kMagicNumberSize = 4;
constexpr size_t kFlatbuffersBuilderInitSize = 1024;

void ChangeMod(const std::string &file_path) {
#ifndef _MSC_VER
  if (access(file_path.c_str(), F_OK) == 0) {
    (void)chmod(file_path.c_str(), S_IWUSR | S_IRUSR);
  }
#endif
}

std::fstream *ReopenFile(const std::string &file_path, std::ios_base::openmode open_mode = std::ios::in | std::ios::out,
                         std::fstream *fs = nullptr) {
  if (fs == nullptr) {
    ChangeMod(file_path);
    return OpenFile(file_path, open_mode);
  } else {
    fs->close();
    fs->open(file_path, open_mode);
    if (!fs->good()) {
      MS_LOG(DEBUG) << "File is not exist: " << file_path;
      return nullptr;
    }
    if (!fs->is_open()) {
      MS_LOG(DEBUG) << "Can not open file: " << file_path;
      return nullptr;
    }
    return fs;
  }
}
}  // namespace

bool MetaGraphSerializer::InitPath(const std::string &output_path) {
  if (!ParserPathAndModelName(output_path, &this->save_path_, &this->model_name_)) {
    MS_LOG(ERROR) << "parser save path and model name from output_path failed.";
    return false;
  }
#ifdef _WIN32
  save_model_path_ = save_path_ + "\\" + model_name_ + ".ms";
  save_data_path_ = save_path_ + "\\" + model_name_ + ".msw";
#else
  save_model_path_ = save_path_ + "/" + model_name_ + ".ms";
  save_data_path_ = save_path_ + "/" + model_name_ + ".msw";
#endif
  return true;
}

bool MetaGraphSerializer::Init(const schema::MetaGraphT &graph, bool save_together) {
  // init file streams
  ChangeMod(save_model_path_);
  model_fs_ = OpenFile(save_model_path_, std::ios::out | std::ios::binary | std::ios::trunc);
  if (model_fs_ == nullptr) {
    MS_LOG(ERROR) << "Open " << save_model_path_ << " failed";
    return false;
  }
  if (save_together) {
    return true;
  }

  ChangeMod(save_data_path_);
  data_fs_ = OpenFile(save_data_path_, std::ios::out | std::ios::binary | std::ios::trunc);
  if (data_fs_ == nullptr) {
    MS_LOG(ERROR) << "Open " << save_data_path_ << " failed";
    return false;
  }
  // write weight file head
  auto head_data = reinterpret_cast<char *>(malloc(kExternalDataHeadSize));
  if (head_data == nullptr) {
    MS_LOG(ERROR) << "Malloc data for file head failed";
    return false;
  }
  memset(head_data, 0, kExternalDataHeadSize);
  // magic number of weight file: 0x12345678
  auto sum_data = reinterpret_cast<uint32_t *>(head_data);
  sum_data[0] = 0x12345678;
  data_fs_->write(head_data, kExternalDataHeadSize);
  if (data_fs_->bad()) {
    MS_LOG(ERROR) << "Write file head failed";
    free(head_data);
    return false;
  }
  free(head_data);
  cur_offset_ = kExternalDataHeadSize;
  return true;
}

schema::ExternalDataT *MetaGraphSerializer::AddExternalData(const char *data, size_t size) {
  MS_ASSERT(data_fs_ != nullptr);
  auto external_data = new (std::nothrow) schema::ExternalDataT;
  if (external_data == nullptr) {
    MS_LOG(ERROR) << "Create ExternalDataT failed";
    return nullptr;
  }
  external_data->location = model_name_ + ".msw";
  external_data->offset = cur_offset_;
  external_data->length = static_cast<int64_t>(size);
  if (data == nullptr || size == 0) {
    return external_data;
  }
  data_fs_->write(data, static_cast<int64_t>(size));
  if (data_fs_->bad()) {
    MS_LOG(ERROR) << "Write file failed";
    delete external_data;
    return nullptr;
  }
  std::stringstream oss;
  oss << std::hash<char>()(data[0]);
  external_data->checkSum = oss.str();
  cur_offset_ += static_cast<int64_t>(size);
  return external_data;
}

bool MetaGraphSerializer::ExtraAndSerializeModelWeight(const schema::MetaGraphT &graph) {
  if (data_fs_ == nullptr) {
    MS_LOG(ERROR) << "Weight file stream is not inited";
    return false;
  }
  data_fs_ = ReopenFile(save_data_path_, std::ios::out | std::ios::app, data_fs_);
  if (data_fs_ == nullptr) {
    MS_LOG(ERROR) << "Reopen weight file stream failed";
    return false;
  }
  if (this->cur_offset_ != kExternalDataHeadSize) {
    MS_LOG(ERROR) << "Serialized model weight already";
    return false;
  }
  for (const auto &tensor : graph.allTensors) {
    if (tensor->nodeType == NodeType_CNode) {
      continue;
    }
    if (tensor->dataType == kObjectTypeTensorType) {  // not support control-flow now
      continue;
    }
    auto external_data =
      this->AddExternalData(reinterpret_cast<const char *>(tensor->data.data()), tensor->data.size());
    if (external_data == nullptr) {
      MS_LOG(ERROR) << "Serialized model weight failed";
      return false;
    }
    tensor->data.clear();
    tensor->externalData.emplace_back(external_data);
  }
  return true;
}

bool MetaGraphSerializer::SerializeModelAndUpdateWeight(const schema::MetaGraphT &meta_graphT, const Byte *key,
                                                        const size_t key_len, const std::string &enc_mode,
                                                        size_t *size) {
  // serialize model
  flatbuffers::FlatBufferBuilder builder(kFlatbuffersBuilderInitSize);
  auto offset = schema::MetaGraph::Pack(builder, &meta_graphT);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  *size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (!SerializeModel(content, *size, key, key_len, enc_mode)) {
    MS_LOG(ERROR) << "Serialize graph failed";
    return false;
  }

  // update weight file using check-sum of model-buffer
  auto model_crc32 = std::hash<uint8_t>()(content[0]);
  if (data_fs_ == nullptr) {
    MS_LOG(ERROR) << "Weight file stream is not inited";
    return false;
  }
  data_fs_ = ReopenFile(save_data_path_, std::ios::in | std::ios::out, data_fs_);
  if (data_fs_ == nullptr) {
    MS_LOG(ERROR) << "Reopen weight file stream failed";
    return false;
  }
  data_fs_->seekp(kMagicNumberSize, std::ios::beg);
  data_fs_->write(reinterpret_cast<const char *>(&model_crc32), kMagicNumberSize);
#ifndef _MSC_VER
  chmod(save_data_path_.c_str(), S_IRUSR);
#endif
  return true;
}

uint8_t *MetaGraphSerializer::GetMetaGraphPackedBuff(flatbuffers::FlatBufferBuilder *builder,
                                                     const schema::MetaGraphT &graph, size_t *data_size) {
  auto offset = schema::MetaGraph::Pack(*builder, &graph);
  builder->Finish(offset);
  schema::FinishMetaGraphBuffer(*builder, offset);
  *data_size = builder->GetSize();
  return builder->GetBufferPointer();
}

int MetaGraphSerializer::Save(const schema::MetaGraphT &graph, const std::string &output_path, const Byte *key,
                              const size_t key_len, const std::string &enc_mode) {
  size_t size = 0;
  auto ret = MetaGraphSerializer::Save(graph, output_path, &size, key, key_len, enc_mode);
  return ret;
}

int MetaGraphSerializer::Save(const schema::MetaGraphT &graph, const std::string &output_path, size_t *size,
                              const Byte *key, const size_t key_len, const std::string &enc_mode) {
  MetaGraphSerializer meta_graph_serializer;
  *size = 0;
  flatbuffers::FlatBufferBuilder builder(kFlatbuffersBuilderInitSize);
  auto buffer = meta_graph_serializer.GetMetaGraphPackedBuff(&builder, graph, size);
  if (!meta_graph_serializer.InitPath(output_path)) {
    MS_LOG(ERROR) << "Init path failed";
    return RET_ERROR;
  }
  size_t tensors_size = 0;
  for (auto &tensor : graph.allTensors) {
    tensors_size += tensor->data.size();
  }

  auto save_together = (tensors_size < kModelSizeLimit && *size < kModelSizeLimit);
  if (!meta_graph_serializer.Init(graph, save_together)) {
    MS_LOG(ERROR) << "Init MetaGraphSerializer failed";
    return RET_ERROR;
  }
  if (save_together) {
    if (!meta_graph_serializer.SerializeModel(buffer, *size, key, key_len, enc_mode)) {
      MS_LOG(ERROR) << "Serialize graph failed";
      return RET_ERROR;
    }
  } else {
    if (!meta_graph_serializer.ExtraAndSerializeModelWeight(graph)) {
      MS_LOG(ERROR) << "Serialize graph weight failed";
      return RET_ERROR;
    }
    size_t model_size = 0;
    if (!meta_graph_serializer.SerializeModelAndUpdateWeight(graph, key, key_len, enc_mode, &model_size)) {
      MS_LOG(ERROR) << "Serialize graph and adjust weight failed";
      return RET_ERROR;
    }
    *size = model_size + tensors_size;
  }
  return RET_OK;
}

MetaGraphSerializer::~MetaGraphSerializer() {
  if (model_fs_ != nullptr) {
    model_fs_->close();
    delete model_fs_;
  }
  if (data_fs_ != nullptr) {
    data_fs_->close();
    delete data_fs_;
  }
}

bool MetaGraphSerializer::SerializeModel(const void *content, size_t size, const Byte *key, const size_t key_len,
                                         const std::string &enc_mode) {
  MS_ASSERT(model_fs_ != nullptr);
  if (size == 0 || content == nullptr) {
    MS_LOG(ERROR) << "Input meta graph buffer is nullptr";
    return false;
  }
  if (key_len > 0) {
    size_t encrypt_len;
    auto encrypt_content = Encrypt(&encrypt_len, reinterpret_cast<const Byte *>(content), size, key, key_len, enc_mode);
    if (encrypt_content == nullptr || encrypt_len == 0) {
      MS_LOG(ERROR) << "Encrypt failed.";
      model_fs_->close();
      return false;
    }
    model_fs_->write(reinterpret_cast<const char *>(encrypt_content.get()), encrypt_len);
  } else {
    model_fs_->write((const char *)content, static_cast<int64_t>(size));
  }
  if (model_fs_->bad()) {
    MS_LOG(ERROR) << "Write model file failed: " << save_model_path_;
    return false;
  }
#ifndef _MSC_VER
  chmod(save_model_path_.c_str(), S_IRUSR);
#endif
  return true;
}
}  // namespace mindspore::lite
