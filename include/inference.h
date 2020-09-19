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

#ifndef MINDSPORE_INCLUDE_MS_SESSION_H
#define MINDSPORE_INCLUDE_MS_SESSION_H

#include <memory>
#include <vector>
#include <string>
#include "include/infer_tensor.h"
#include "include/infer_log.h"

namespace mindspore {
namespace inference {
enum StatusCode { SUCCESS = 0, FAILED, INVALID_INPUTS };

class Status {
 public:
  Status() : status_code_(FAILED) {}
  Status(enum StatusCode status_code, const std::string &status_msg = "")
      : status_code_(status_code), status_msg_(status_msg) {}
  ~Status() = default;

  bool IsSuccess() const { return status_code_ == SUCCESS; }
  enum StatusCode StatusCode() const { return status_code_; }
  std::string StatusMessage() const { return status_msg_; }
  bool operator==(const Status &other) const { return status_code_ == other.status_code_; }
  bool operator==(enum StatusCode other_code) const { return status_code_ == other_code; }
  bool operator!=(const Status &other) const { return status_code_ != other.status_code_; }
  bool operator!=(enum StatusCode other_code) const { return status_code_ != other_code; }
  operator bool() const = delete;
  Status &operator<(const LogStream &stream) noexcept __attribute__((visibility("default"))) {
    status_msg_ = stream.sstream_->str();
    return *this;
  }

 private:
  enum StatusCode status_code_;
  std::string status_msg_;
};

class MS_API InferSession {
 public:
  InferSession() = default;
  virtual ~InferSession() = default;
  virtual Status InitEnv(const std::string &device_type, uint32_t device_id) = 0;
  virtual Status FinalizeEnv() = 0;
  virtual Status LoadModelFromFile(const std::string &file_name, uint32_t &model_id) = 0;
  virtual Status UnloadModel(uint32_t model_id) = 0;
  // override this method to avoid request/reply data copy
  virtual Status ExecuteModel(uint32_t model_id, const RequestBase &request, ReplyBase &reply) = 0;

  virtual Status ExecuteModel(uint32_t model_id, const std::vector<InferTensor> &inputs,
                              std::vector<InferTensor> &outputs) {
    VectorInferTensorWrapRequest request(inputs);
    VectorInferTensorWrapReply reply(outputs);
    return ExecuteModel(model_id, request, reply);
  }
  // default not support input data preprocess(decode, resize, crop, crop&paste, etc.)
  virtual Status ExecuteModel(uint32_t /*model_id*/,
                              const ImagesRequestBase & /*images_inputs*/,  // images for preprocess
                              const RequestBase & /*request*/, ReplyBase & /*reply*/) {
    return FAILED;
  }
  virtual Status GetModelInputsInfo(uint32_t graph_id, std::vector<inference::InferTensor> *tensor_list) const {
    Status status(SUCCESS);
    return status;
  }
  static std::shared_ptr<InferSession> CreateSession(const std::string &device, uint32_t device_id);
};
}  // namespace inference
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_MS_SESSION_H
