
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

#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_SESSION_H_
#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_SESSION_H_

#include "include/errorcode.h"
#include "include/lite_session.h"

#include "tensor.h"

namespace mindspore {
namespace lite {

#define MS_ERROR_IF_NULL(ptr)            \
  do {                                   \
    if ((ptr) == nullptr) {              \
      return mindspore::lite::RET_ERROR; \
    }                                    \
  } while (0)

class LiteSession : public session::LiteSession {
 public:
  LiteSession() = default;

  ~LiteSession() override;

  void BindThread(bool if_bind) override {}

  int CompileGraph(lite::Model *model) override;

  std::vector<tensor::MSTensor *> GetInputs() const override;

  mindspore::tensor::MSTensor *GetInputsByTensorName(const std::string &tensor_name) const override { return nullptr; }

  int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;

  std::vector<tensor::MSTensor *> GetOutputsByNodeName(const std::string &node_name) const override;

  std::unordered_map<std::string, mindspore::tensor::MSTensor *> GetOutputs() const override;

  std::vector<std::string> GetOutputTensorNames() const override;

  mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const override;

  int Resize(const std::vector<tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims) override;

  int InitRuntimeBuffer();

 private:
  int SetInputsData(const std::vector<MTensor *> &inputs) const;
  std::vector<MTensor *> inputs_;
  std::vector<MTensor *> outputs_;
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> output_tensor_map_;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> output_node_map_;

  void *runtime_buffer_;
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_SESSION_H_

