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

#ifndef MINDSPORE_CCSRC_UTILS_LOAD_ONNX_ANF_CONVERTER_H
#define MINDSPORE_CCSRC_UTILS_LOAD_ONNX_ANF_CONVERTER_H
#include <string>
#include <memory>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "proto/onnx.pb.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace lite {
class AnfConverter {
 public:
  static std::shared_ptr<FuncGraph> RunAnfConverter(const std::string &file_path);
  static std::shared_ptr<FuncGraph> RunAnfConverter(const char *buf, const size_t buf_size);

 private:
  static void Trim(std::string *input);
  static int ValidateFileStr(const std::string &modelFile, std::string fileType);
  static bool ReadOnnxFromBinary(const std::string &modelFile, google::protobuf::Message *onnx_model);
};
}  // namespace lite
}  // namespace mindspore
#endif
