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

namespace mindspore::lite {
namespace {
constexpr size_t kModelSizeLimit = 2 * 1024 * 1024;
constexpr size_t kExternalDataHeadSize = 4096;
constexpr size_t kMagicNumberSize = 4;

std::fstream *OpenFile(const std::string &save_path, bool append) {
#ifndef _MSC_VER
  if (access(save_path.c_str(), F_OK) == 0) {
    chmod(save_path.c_str(), S_IWUSR);
  }
#endif
  auto fs = new (std::nothrow) std::fstream();
  if (fs == nullptr) {
    MS_LOG(DEBUG) << "Create file stream failed";
    return nullptr;
  }
  if (append) {
    fs->open(save_path, std::ios::app);
  } else {
    fs->open(save_path);
  }
  if (!fs->is_open()) {
    MS_LOG(DEBUG) << "Can not open output file: " << save_path;
    delete fs;
    return nullptr;
  }
  return fs;
}

bool SerializeModel(const std::string &save_path, const void *content, size_t size) {
  if (size == 0 || content == nullptr) {
    MS_LOG(ERROR) << "Input meta graph buffer is nullptr";
    return false;
  }
#ifndef _MSC_VER
  if (access(save_path.c_str(), F_OK) == 0) {
    chmod(save_path.c_str(), S_IWUSR);
  }
#endif
  std::ofstream output(save_path, std::ofstream::binary);
  if (!output.is_open()) {
    MS_LOG(ERROR) << "Can not open output file: " << save_path;
    return RET_ERROR;
  }

  output.write((const char *)content, size);
  if (output.bad()) {
    output.close();
    MS_LOG(ERROR) << "Write output file : " << save_path << " failed";
    return RET_ERROR;
  }
  output.close();
#ifndef _MSC_VER
  chmod(save_path.c_str(), S_IRUSR);
#endif
  return true;
}
}  // namespace

void MetaGraphSerializer::InitPath(const std::string &output_path) {
  this->save_path_.clear();
  this->model_name_.clear();
  if (output_path.empty()) {
    return;
  }
  auto pos = output_path.find_last_of('/');
  std::string model_name;
  if (pos == std::string::npos) {
    this->save_path_ = "./";
    model_name = output_path;
  } else {
    this->save_path_ = output_path.substr(0, pos + 1);
    model_name = output_path.substr(pos + 1);
  }
  auto suffix_pos = model_name.find_last_of('.');
  if (suffix_pos == std::string::npos) {
    this->model_name_ = model_name;
  } else {
    if (model_name.substr(suffix_pos + 1) == "ms") {
      this->model_name_ = model_name.substr(0, suffix_pos);
    } else {
      this->model_name_ = model_name;
    }
  }
  save_model_path_ = save_path_ + model_name_ + ".ms";
  save_data_path_ = save_path_ + model_name_ + ".msw";
}

bool MetaGraphSerializer::Init(const schema::MetaGraphT &graph, const std::string &output_path) {
  InitPath(output_path);
  // delete exist file
  struct stat file_state {};
  if (stat(save_model_path_.c_str(), &file_state) == 0) {
    remove(save_model_path_.c_str());
  }
  if (stat(save_data_path_.c_str(), &file_state) == 0) {
    remove(save_data_path_.c_str());
  }
  // write weight file head
  auto file = OpenFile(save_data_path_, true);
  if (file == nullptr) {
    MS_LOG(ERROR) << "Open file failed: " << save_data_path_;
    return false;
  }
  auto head_data = reinterpret_cast<char *>(malloc(kExternalDataHeadSize));
  if (head_data == nullptr) {
    MS_LOG(ERROR) << "Malloc data for file head failed";
    return false;
  }
  auto ret = memset_s(head_data, kExternalDataHeadSize, 0, kExternalDataHeadSize);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memset failed: " << ret;
    return false;
  }
  // magic number of weight file: 0x12345678
  head_data[0] = 0x12;
  head_data[1] = 0x34;
  head_data[2] = 0x56;
  head_data[3] = 0x78;
  file->write(head_data, kExternalDataHeadSize);
  if (file->bad()) {
    MS_LOG(ERROR) << "Write file head failed";
    free(head_data);
    file->close();
    delete file;
    return false;
  }
  free(head_data);
  file->close();
  delete file;
  cur_offset_ = kExternalDataHeadSize;
  return true;
}

schema::ExternalDataT *MetaGraphSerializer::AddExternalData(std::fstream *file, const char *data, size_t size) {
  MS_ASSERT(file != nullptr);
  auto external_data = new (std::nothrow) schema::ExternalDataT;
  if (external_data == nullptr) {
    MS_LOG(ERROR) << "Create ExternalDataT failed";
    return nullptr;
  }
  external_data->location = save_data_path_;
  external_data->offset = cur_offset_;
  external_data->length = static_cast<int64_t>(size);
  if (data == nullptr || size == 0) {
    return external_data;
  }
  file->write(data, static_cast<int64_t>(size));
  if (file->bad()) {
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
  if (this->cur_offset_ != kExternalDataHeadSize) {
    MS_LOG(ERROR) << "Serialized model weight already";
    return false;
  }
  auto file = OpenFile(save_data_path_, true);
  if (file == nullptr) {
    MS_LOG(ERROR) << "Open file failed: " << save_data_path_;
    return false;
  }
  for (const auto &tensor : graph.allTensors) {
    if (tensor->nodeType == NodeType_CNode) {
      continue;
    }
    if (tensor->dataType == kObjectTypeTensorType) {
      continue;
    }
    auto external_data =
      this->AddExternalData(file, reinterpret_cast<const char *>(tensor->data.data()), tensor->data.size());
    if (external_data == nullptr) {
      MS_LOG(ERROR) << "Serialized model weight failed";
      file->close();
      delete file;
      return false;
    }
    tensor->data.clear();
    tensor->externalData.emplace_back(external_data);
  }
  file->close();
  delete file;
  return true;
}

bool MetaGraphSerializer::SerializeModelAndUpdateWeight(const schema::MetaGraphT &meta_graphT) {
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, &meta_graphT);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return false;
  }
  auto model_crc32 = std::hash<uint8_t>()(content[0]);
  if (!SerializeModel(save_model_path_, content, size)) {
    MS_LOG(ERROR) << "Serialize graph failed";
    return false;
  }
  auto file = OpenFile(save_data_path_, false);
  if (file == nullptr) {
    MS_LOG(ERROR) << "Open file failed: " << save_data_path_;
    return false;
  }
  file->seekp(kMagicNumberSize, std::ios::beg);
  file->write(reinterpret_cast<const char *>(&model_crc32), kMagicNumberSize);
  file->close();
  delete file;
#ifndef _MSC_VER
  chmod(save_data_path_.c_str(), S_IRUSR);
#endif
  std::cout << "sum: " << model_crc32 << std::endl;
  return true;
}

int MetaGraphSerializer::Save(const schema::MetaGraphT &graph, const std::string &output_path) {
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, &graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  auto save_together = size < kModelSizeLimit;
  if (!save_together && false) {
    MetaGraphSerializer meta_graph_serializer;
    if (!meta_graph_serializer.Init(graph, output_path)) {
      MS_LOG(ERROR) << "Init MetaGraphSerializer failed";
      return RET_ERROR;
    }
    if (!meta_graph_serializer.ExtraAndSerializeModelWeight(graph)) {
      MS_LOG(ERROR) << "Serialize graph weight failed";
      return RET_ERROR;
    }
    if (!meta_graph_serializer.SerializeModelAndUpdateWeight(graph)) {
      MS_LOG(ERROR) << "Serialize graph and adjust weight failed";
      return RET_ERROR;
    }
  } else {
    std::string file_name = output_path;
    if (file_name.substr(file_name.find_last_of('.') + 1) != "ms") {
      file_name = file_name + ".ms";
    }
    auto content = builder.GetBufferPointer();
    if (content == nullptr) {
      MS_LOG(ERROR) << "GetBufferPointer nullptr";
      return RET_ERROR;
    }
    if (!SerializeModel(file_name, content, size)) {
      MS_LOG(ERROR) << "Serialize graph failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
