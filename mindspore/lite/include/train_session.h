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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_SESSION_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_SESSION_H_
#include <vector>
#include <string>
#include <unordered_map>
// #include "include/lite_session.h"
#include "src/lite_session.h"

namespace mindspore {
namespace lite {
struct Model;
}

namespace session {

class TrainSession : public lite::LiteSession {
 public:
  TrainSession();
  ~TrainSession() = default;

  int RunGraph(const session::KernelCallBack &before = nullptr,
               const session::KernelCallBack &after = nullptr) override;

  int CompileGraph(lite::Model *model) override;
  virtual void ReplaceOps();
  virtual void *ExportToBuf(void *buf, size_t *len) const;

  // todo: output tensors by tensor name
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> GetOutputMap() const;
  std::vector<tensor::MSTensor *> GetOutputsByName(const std::string &node_name) const;

  virtual void train();
  bool is_train() { return train_mode_ == true; }
  virtual void eval();
  bool is_eval() { return train_mode_ == false; }

 protected:
  bool train_mode_ = false;
  lite::Model *model_ = nullptr;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> ext_output_map_;

  // private:
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_SESSION_H_
