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

#include <fcntl.h>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include "utils/base_ref_extends.h"
#include "utils/load_onnx/anf_model_parser.h"
#include "utils/load_onnx/anf_converter.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "proto/onnx.pb.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace lite {

const char WHITESPACE[] = "\t\n\v\f\r ";
const int FLAG_PREFIX_LEN = 2;

void AnfConverter::Trim(std::string *input) {
  if (input == nullptr) {
    return;
  }
  if (input->empty()) {
    return;
  }
  input->erase(0, input->find_first_not_of(WHITESPACE));
  input->erase(input->find_last_not_of(WHITESPACE) + 1);
}

int AnfConverter::ValidateFileStr(const std::string &modelFile, std::string fileType) {
  if (modelFile.size() > fileType.size()) {
    if (modelFile.substr(modelFile.size() - fileType.size()) == fileType) {
      return 0;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

bool AnfConverter::ReadOnnxFromBinary(const std::string &modelFile, google::protobuf::Message *onnx_model) {
  std::unique_ptr<char> onnx_file(new (std::nothrow) char[PATH_MAX]{0});
  int fd = open(onnx_file.get(), O_RDONLY);
  if (fd < 0) {
    MS_LOG(EXCEPTION) << "failed to open file";
  }
  google::protobuf::io::FileInputStream input(fd);
  google::protobuf::io::CodedInputStream code_input(&input);
  code_input.SetTotalBytesLimit(INT_MAX, 536870912);
  bool ret = onnx_model->ParseFromCodedStream(&code_input);
  if (!ret) {
    MS_LOG(ERROR) << "load onnx file failed";
    return false;
  }
  (void)close(fd);
  MS_LOG(INFO) << "enter ReadProtoFromBinary success!" << std::endl;
  return true;
}

std::shared_ptr<FuncGraph> AnfConverter::RunAnfConverter(const std::string &file_path) {
  std::string modelFile;

  std::string tmp = file_path;
  Trim(&tmp);
  const std::string flagItem(tmp);

  size_t pos = flagItem.find_first_of("=");
  if (pos == std::string::npos) {
    MS_LOG(ERROR) << "Trans data not support input format!";
  } else {
    modelFile = flagItem.substr(pos + 1);
    std::cout << "input protobuf file path is: " << modelFile << std::endl;
  }

  if (ValidateFileStr(modelFile, ".pb") != 0) {
    MS_LOG(EXCEPTION) << "INPUT ILLEGAL: modelFile must be *.pb";
  }

  onnx::ModelProto model_;
  ReadOnnxFromBinary(modelFile, &model_);
  MSANFModelParser model_parser;
  FuncGraphPtr dstgraph_ptr = model_parser.Parse(model_);
  return dstgraph_ptr;
}

std::shared_ptr<FuncGraph> AnfConverter::RunAnfConverter(const char *buf, const size_t buf_size) {
  Py_Initialize();
  MS_EXCEPTION_IF_NULL(buf);
  std::string str((const char *)buf, buf_size);
  onnx::ModelProto model_;
  if (!model_.ParseFromString(str)) {
    MS_LOG(EXCEPTION) << "Parse model from buffer fail!";
  }
  MSANFModelParser model_parser;
  FuncGraphPtr dstgraph_ptr = model_parser.Parse(model_);
  return dstgraph_ptr;
}
}  // namespace lite
}  // namespace mindspore
