/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend/model/model_infer.h"
#include "common/log_adapter.h"
#include "acl/acl.h"

namespace mindspore::kernel {
namespace acl {
namespace {
std::mutex g_context_mutex;
}
ModelInfer::ModelInfer(const AclModelOptionsPtr &options)
    : init_flag_(false),
      device_type_("AscendCL"),
      context_(nullptr),
      options_(options),
      model_process_(options),
      acl_env_(nullptr) {}

bool ModelInfer::Init() {
  if (init_flag_) {
    MS_LOG(INFO) << "Acl has been initialized, skip.";
    return true;
  }
  if (options_ == nullptr) {
    MS_LOG(ERROR) << "Acl options is nullptr.";
    return false;
  }
  std::string acl_init_option = !options_->dump_path.empty()
                                  ? options_->dump_path
                                  : (!options_->profiling_path.empty() ? options_->profiling_path : std::string());
  acl_env_ = AclEnvGuard::GetAclEnv(acl_init_option);
  if (acl_env_ == nullptr) {
    MS_LOG(ERROR) << "Acl init failed.";
    return false;
  }
  std::lock_guard<std::mutex> lock(g_context_mutex);
  int32_t device_id = options_->device_id;
  aclError ret = aclrtSetDevice(device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl open device " << device_id << " failed.";
    return false;
  }
  MS_LOG(INFO) << "Open device " << device_id << " success.";

  ret = aclrtCreateContext(&context_, device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl create context failed.";
    return false;
  }
  MS_LOG(INFO) << "Create context success.";

  aclrtRunMode run_mode;
  ret = aclrtGetRunMode(&run_mode);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl get run mode failed.";
    return false;
  }
  bool is_device = (run_mode == ACL_DEVICE);
  model_process_.SetIsDevice(is_device);
  MS_LOG(INFO) << "Get run mode success is device input/output " << is_device;

  MS_LOG(INFO) << "Init model success, device id " << device_id;
  init_flag_ = true;
  return true;
}

bool ModelInfer::Finalize() {
  if (!init_flag_) {
    MS_LOG(INFO) << "Init is not ok, no need to finalize.";
    return true;
  }
  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed.";
    return false;
  }
  if (!model_process_.UnLoad()) {
    MS_LOG(ERROR) << "Unload model inner failed.";
  }
  std::lock_guard<std::mutex> lock(g_context_mutex);
  if (context_ != nullptr) {
    rt_ret = aclrtDestroyContext(context_);
    if (rt_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy context failed.";
    }
    context_ = nullptr;
  }
  MS_LOG(INFO) << "End to destroy context.";

  rt_ret = aclrtResetDevice(options_->device_id);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Reset device " << options_->device_id << " failed.";
  }
  MS_LOG(INFO) << "End to reset device " << options_->device_id;
  init_flag_ = false;
  return true;
}

bool ModelInfer::Load(const void *om_data, size_t om_data_size) {
  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed, ret = " << rt_ret;
    return false;
  }
  if (!model_process_.Load(om_data, om_data_size)) {
    MS_LOG(ERROR) << "Load model model failed.";
    return false;
  }
  return true;
}

bool ModelInfer::Inference(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs) {
  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed, ret = " << rt_ret;
    return false;
  }
  return model_process_.PredictFromHost(inputs, outputs);
}

std::vector<Format> ModelInfer::GetInputFormat() { return model_process_.GetInputFormat(); }
const std::vector<ShapeVector> ModelInfer::GetOutputShape() { return model_process_.GetOutputShape(); }
const std::vector<ShapeVector> ModelInfer::GetInputShape() { return model_process_.GetInputShape(); }
const std::vector<TypeId> ModelInfer::GetInputDataType() { return model_process_.GetInputDataType(); }
const std::vector<TypeId> ModelInfer::GetOutputDataType() { return model_process_.GetOutputDataType(); }
std::vector<Format> ModelInfer::GetOutputFormat() { return model_process_.GetOutputFormat(); }

bool ModelInfer::Resize(const std::vector<ShapeVector> &new_shapes) {
  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed, ret = " << rt_ret;
    return false;
  }
  return model_process_.Resize(new_shapes);
}
}  // namespace acl
}  // namespace mindspore::kernel
