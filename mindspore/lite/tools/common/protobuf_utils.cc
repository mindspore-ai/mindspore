/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/common/protobuf_utils.h"
#include <fstream>
#include <string>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/coded_stream.h"
#include "src/common/file_utils.h"

namespace mindspore {
namespace lite {
static const int PROTO_READ_BYTES_LIMIT = INT_MAX;  // Max size of 2 GB minus 1 byte.
static const int WARNING_THRESHOLD = 536870912 * 2;

bool ReadProtoFromCodedInputStream(google::protobuf::io::CodedInputStream *coded_stream,
                                   google::protobuf::Message *proto) {
  if (proto == nullptr) {
    MS_LOG(ERROR) << "incorrect parameter. nullptr == proto";
    return false;
  }
  coded_stream->SetTotalBytesLimit(PROTO_READ_BYTES_LIMIT, WARNING_THRESHOLD);
  return proto->ParseFromCodedStream(coded_stream);
}

STATUS ReadProtoFromText(const char *file, google::protobuf::Message *message) {
  if (file == nullptr || message == nullptr) {
    return RET_ERROR;
  }

  std::string realPath = RealPath(file);
  if (realPath.empty()) {
    MS_LOG(ERROR) << "Proto file path " << file << " is  not valid";
    return RET_ERROR;
  }

  std::ifstream fs(realPath.c_str(), std::ifstream::in);

  if (!fs.is_open()) {
    MS_LOG(ERROR) << "Open proto file " << file << " failed.";
    return RET_ERROR;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  bool status = google::protobuf::TextFormat::Parse(&input, message);
  if (!status) {
    MS_LOG(ERROR) << "call [google::protobuf::TextFormat::Parse] func status fail, please check your text file.";
    fs.close();
    return RET_ERROR;
  }

  fs.close();
  return RET_OK;
}

STATUS ReadProtoFromBinaryFile(const char *file, google::protobuf::Message *message) {
  if (file == nullptr || message == nullptr) {
    return RET_ERROR;
  }

  std::string realPath = RealPath(file);
  if (realPath.empty()) {
    MS_LOG(ERROR) << "Binary proto file path " << file << " is not valid";
    return RET_ERROR;
  }

  std::ifstream fs(realPath, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    MS_LOG(ERROR) << "Open binary proto file " << file << " failed.";
    return RET_ERROR;
  }

  google::protobuf::io::IstreamInputStream istream(&fs);
  google::protobuf::io::CodedInputStream coded_stream(&istream);

  bool success = ReadProtoFromCodedInputStream(&coded_stream, message);
  fs.close();

  if (!success) {
    MS_LOG(DEBUG) << "Parse " << file << " failed.";
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
