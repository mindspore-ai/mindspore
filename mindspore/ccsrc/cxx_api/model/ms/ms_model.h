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
#ifndef MINDSPORE_CCSRC_SESSION_SESSION_H
#define MINDSPORE_CCSRC_SESSION_SESSION_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <map>

#include "backend/session/session_basic.h"
#include "ir/anf.h"
#include "include/api/status.h"
#include "cxx_api/model/model_impl.h"

#ifdef ENABLE_D
#include "runtime/context.h"
#endif

namespace mindspore {
namespace api {
class MsModel : public ModelImpl {
 public:
  explicit MsModel(uint32_t device_id);
  ~MsModel();

  Status LoadModel(const Buffer &model_data, ModelType type,
                   const std::map<std::string, std::string> &options) override;
  Status LoadModel(const std::string &file_name, ModelType type,
                   const std::map<std::string, std::string> &options) override;
  Status UnloadModel() override;

  Status Train(const DataSet &dataset, std::map<std::string, Buffer> *outputs) override;
  Status Eval(const DataSet &dataset, std::map<std::string, Buffer> *outputs) override;
  Status Predict(const std::map<std::string, Buffer> &inputs, std::map<std::string, Buffer> *outputs) override;

  Status GetInputsInfo(std::vector<Tensor> *tensor_list) const override;
  Status GetOutputsInfo(std::vector<Tensor> *tensor_list) const override;

  Status InitEnv(const std::unordered_map<std::string, std::string> &other_options);
  Status FinalizeEnv();

 private:
  std::shared_ptr<session::SessionBasic> session_impl_ = nullptr;
  uint32_t graph_id_;
  std::string device_type_;
  int32_t device_id_ = 0;
#ifdef ENABLE_D
  rtContext_t context_ = nullptr;
#endif
  std::vector<tensor::TensorPtr> inputs_;
  std::vector<tensor::TensorPtr> outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool load_flag_ = false;

  std::shared_ptr<FuncGraph> LoadModel(const char *model_buf, size_t size, const std::string &device);
  Buffer ReadFile(const std::string &file);
  static void RegAllOp();
  Status CompileGraph(std::shared_ptr<FuncGraph> funcGraphPtr);
  Status CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs) const;
  std::vector<tensor::TensorPtr> RunGraph(const std::vector<tensor::TensorPtr> &inputs);
  Status ExecuteModel(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs);
};

API_REG_MODEL(AscendMS, MsModel);
}  // namespace api
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_SESSION_SESSION_BASIC_H
