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

namespace mindspore {
namespace inference {

class MS_API InferSession {
 public:
  InferSession() = default;
  virtual ~InferSession() = default;
  virtual bool InitEnv(const std::string &device_type, uint32_t device_id) = 0;
  virtual bool FinalizeEnv() = 0;
  virtual bool LoadModelFromFile(const std::string &file_name, uint32_t &model_id) = 0;
  virtual bool UnloadModel(uint32_t model_id) = 0;
  // override this method to avoid request/reply data copy
  virtual bool ExecuteModel(uint32_t model_id, const RequestBase &request, ReplyBase &reply) = 0;

  virtual bool ExecuteModel(uint32_t model_id, const std::vector<InferTensor> &inputs,
                            std::vector<InferTensor> &outputs) {
    VectorInferTensorWrapRequest request(inputs);
    VectorInferTensorWrapReply reply(outputs);
    return ExecuteModel(model_id, request, reply);
  }

  static std::shared_ptr<InferSession> CreateSession(const std::string &device, uint32_t device_id);
};

}  // namespace inference
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_MS_SESSION_H
