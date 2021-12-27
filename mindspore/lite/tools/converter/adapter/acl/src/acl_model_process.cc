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

#include "tools/converter/adapter/acl/src/acl_model_process.h"
#include "src/runtime/kernel/ascend/src/acl_env_guard.h"
#include "src/common/log_util.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace lite {
AclModelProcess::AclModelProcess(const Buffer &om_data, const acl::AclModelOptionCfg &options)
    : om_data_(om_data), options_(options), model_desc_(nullptr), is_load_(false) {}

STATUS AclModelProcess::Init() {
  auto acl_env = kernel::acl::AclEnvGuard::GetAclEnv("");
  if (acl_env == nullptr) {
    MS_LOG(ERROR) << "Acl init failed.";
    return lite::RET_ERROR;
  }
  aclError ret = aclrtSetDevice(options_.device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl open device " << options_.device_id << " failed, ret=" << ret;
    return lite::RET_ERROR;
  }
  MS_LOG(DEBUG) << "Open device " << options_.device_id << " success.";
  return lite::RET_OK;
}

STATUS AclModelProcess::Load() {
  if (is_load_) {
    MS_LOG(WARNING) << "Model is loaded, no need to load again.";
    return lite::RET_OK;
  }
  if (Init() != lite::RET_OK) {
    MS_LOG(ERROR) << "Model process init failed.";
    return lite::RET_ERROR;
  }
  auto acl_ret = aclmdlLoadFromMem(om_data_.Data(), om_data_.DataSize(), &model_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed, ret = " << acl_ret;
    return lite::RET_ERROR;
  }
  model_desc_ = aclmdlCreateDesc();
  acl_ret = aclmdlGetDesc(model_desc_, model_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Get model desc failed, ret = " << acl_ret;
    return lite::RET_ERROR;
  }
  is_load_ = true;
  return lite::RET_OK;
}

STATUS AclModelProcess::GetInputsShape(std::vector<std::vector<int64_t>> *inputs_shape) {
  CHECK_NULL_RETURN(inputs_shape);
  if (!is_load_) {
    MS_LOG(WARNING) << "Model is not loaded, now load model first.";
    if (Load() != lite::RET_OK) {
      MS_LOG(ERROR) << "Load model failed.";
      return lite::RET_ERROR;
    }
  }
  size_t input_size = aclmdlGetNumInputs(model_desc_);
  MS_LOG(INFO) << "Input size of model " << input_size;
  for (size_t i = 0; i < input_size; ++i) {
    aclmdlIODims dims;
    auto ret = aclmdlGetInputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed, ret = " << ret;
      return lite::RET_ERROR;
    }
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    inputs_shape->push_back(shape);
  }
  return lite::RET_OK;
}

STATUS AclModelProcess::UnLoad() {
  if (!is_load_) {
    MS_LOG(WARNING) << "Model is not loaded, please load model first";
    return lite::RET_OK;
  }
  auto ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
    return lite::RET_ERROR;
  }
  ret = aclmdlDestroyDesc(model_desc_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Destroy model desc failed, ret = " << ret;
    return lite::RET_ERROR;
  }
  ret = aclrtResetDevice(options_.device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Reset device " << options_.device_id << " failed.";
  }
  return lite::RET_OK;
}
}  // namespace lite
}  // namespace mindspore
