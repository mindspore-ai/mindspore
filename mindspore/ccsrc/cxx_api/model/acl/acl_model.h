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
#ifndef MINDSPORE_CCSRC_CXX_API_ACL_MODEL_H
#define MINDSPORE_CCSRC_CXX_API_ACL_MODEL_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <map>

#include "ir/anf.h"
#include "include/api/status.h"
#include "cxx_api/model/model_impl.h"
#include "cxx_api/model/acl/dvpp_process.h"
#include "cxx_api/model/acl/model_process.h"
#include "cxx_api/model/acl/model_converter.h"
#include "cxx_api/model/acl/acl_model_options.h"
#include "ir/tensor.h"

namespace mindspore::api {
class AclModel : public ModelImpl {
 public:
  explicit AclModel(uint32_t device_id)
      : init_flag_(false),
        load_flag_(false),
        device_type_("AscendCL"),
        device_id_(device_id),
        context_(nullptr),
        stream_(nullptr),
        acl_env_(nullptr),
        model_process_(),
        dvpp_process_(),
        model_converter_(),
        options_(nullptr) {}
  ~AclModel() = default;

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

 private:
  bool init_flag_;
  bool load_flag_;
  std::string device_type_;
  int32_t device_id_;
  aclrtContext context_;
  aclrtStream stream_;

  class AclEnvGuard;
  std::shared_ptr<AclEnvGuard> acl_env_;
  static std::weak_ptr<AclEnvGuard> global_acl_env_;
  static std::mutex global_acl_env_mutex_;

  ModelProcess model_process_;
  DvppProcess dvpp_process_;
  ModelConverter model_converter_;
  std::unique_ptr<AclModelOptions> options_;

  Status InitEnv();
  Status FinalizeEnv();
};

class AclModel::AclEnvGuard {
 public:
  explicit AclEnvGuard(const std::string &cfg_file);
  ~AclEnvGuard();
  aclError GetErrno() const { return errno_; }

 private:
  aclError errno_;
};

API_REG_MODEL(AscendCL, AclModel);
}  // namespace mindspore::api
#endif  // MINDSPORE_CCSRC_CXX_API_ACL_MODEL_H
