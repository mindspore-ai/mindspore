/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/custom_v2_aclnn_kernel.h"
#include <algorithm>
#include <tuple>
#include "kernel/ascend/acl_ir/custom/custom_op_api_exec.h"
#include "kernel/ascend/acl_ir/custom/custom_op_api_cache.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
namespace custom {

void CustomV2AclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "Start get custom v2 workspace info, op_type: " << op_type_;
  MS_VLOG(VL_CUSTOM_OP) << "Start get custom v2 workspace info, op_type: " << op_type_;
  GetCustomInputTypes();
  auto dynamic_inputs = GetCustomInputs(inputs);
  auto dynamic_outputs = GetCustomOutputs(outputs);
  if (input_output_types_.size() != (dynamic_inputs.size() + dynamic_outputs.size())) {
    MS_LOG(EXCEPTION) << "Custom op " << op_type_ << " inputs type size " << input_output_types_.size()
                      << " is exceeds I/O size:" << dynamic_inputs.size() + dynamic_outputs.size();
  }
  GetWorkspaceForResize(dynamic_inputs, dynamic_outputs);
  MS_LOG(DEBUG) << "End get custom v2 workspace info, op_type: " << op_type_;
  MS_VLOG(VL_CUSTOM_OP) << "End get custom v2 workspace info, op_type: " << op_type_;
}

bool CustomV2AclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "Start launch custom v2, op_type: " << op_type_;
  MS_VLOG(VL_CUSTOM_OP) << "Start launch custom v2, op_type: " << op_type_;
  auto dyn_inputs = GetCustomInputs(inputs);
  auto dyn_outputs = GetCustomOutputs(outputs);
  RunOp(stream_ptr, workspace, dyn_inputs, dyn_outputs);
  MS_LOG(DEBUG) << "End launch custom v2, op_type: " << op_type_;
  MS_VLOG(VL_CUSTOM_OP) << "End launch custom v2, op_type: " << op_type_;
  return true;
}

std::vector<std::vector<KernelTensor *>> CustomV2AclnnKernelMod::GetCustomInputs(
  const std::vector<KernelTensor *> &inputs) {
  std::vector<std::vector<KernelTensor *>> dynamic_inputs;
  MS_EXCEPTION_IF_NULL(primitive_);
  if (!primitive_->HasAttr(kAttrDynInputSizes)) {
    for (const auto &item : inputs) {
      (void)dynamic_inputs.emplace_back(std::vector<KernelTensor *>({item}));
    }
    return dynamic_inputs;
  }

  auto value = primitive_->GetAttr(kAttrDynInputSizes);
  MS_EXCEPTION_IF_NULL(value);
  auto dynamic_input_sizes = GetValue<std::vector<int64_t>>(value);
  int64_t offset = 0;
  for (const auto &item : dynamic_input_sizes) {
    std::vector<KernelTensor *> dynamic_input;
    if (item > 0) {
      std::copy(inputs.begin() + offset, inputs.begin() + offset + item, std::back_inserter(dynamic_input));
      offset = offset + item;
    } else {
      std::copy(inputs.begin() + offset, inputs.begin() + offset + 1, std::back_inserter(dynamic_input));
      offset = offset + 1;
    }
    (void)dynamic_inputs.emplace_back(dynamic_input);
  }
  return dynamic_inputs;
}

std::vector<std::vector<KernelTensor *>> CustomV2AclnnKernelMod::GetCustomOutputs(
  const std::vector<KernelTensor *> &outputs) {
  std::vector<std::vector<KernelTensor *>> dynamic_outputs;
  if (input_output_types_[input_output_types_.size() - 1] == CustomSupportType::kTypeTensorList) {
    (void)dynamic_outputs.emplace_back(outputs);
  } else {
    for (const auto &item : outputs) {
      (void)dynamic_outputs.emplace_back(std::vector<KernelTensor *>({item}));
    }
  }
  return dynamic_outputs;
}

std::vector<void *> CustomV2AclnnKernelMod::ConvertTypes(const std::vector<std::vector<KernelTensor *>> &inputs,
                                                         size_t offset) {
  if (input_output_types_.size() < (inputs.size() + offset)) {
    MS_LOG(EXCEPTION) << "Custom op " << op_type_ << " inputs type size " << input_output_types_.size()
                      << " is less than I/O size:" << inputs.size() + offset;
  }
  std::vector<void *> convert_inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    KernelTensor *input;
    auto dyn_input = inputs[i];
    if (dyn_input.empty()) {
      MS_LOG(EXCEPTION) << "Custom op [" << op_type_ << "] input-" << i << " is empty!";
    } else {
      input = dyn_input[0];
      MS_EXCEPTION_IF_NULL(input);
    }

    auto type = input_output_types_[i + offset];
    MS_LOG(INFO) << "Convert custom op [" << op_type_ << "] input-" << i
                 << ", input type: " << custom_supported_type_to_string.at(type);
    MS_VLOG(VL_CUSTOM_OP) << "Convert custom op [" << op_type_ << "] input-" << i
                          << ", input type: " << custom_supported_type_to_string.at(type);
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        (void)convert_inputs.emplace_back(device::ascend::ConvertType(input));
        break;
      }
      case CustomSupportType::kTypeTensorList: {
        (void)convert_inputs.emplace_back(device::ascend::ConvertType(dyn_input));
        break;
      }
      case CustomSupportType::kTypeBool: {
        (void)inputs_bool_value_.emplace_back(device::ascend::ConvertKernelTensor<bool>(input));
        (void)convert_inputs.emplace_back(&(inputs_bool_value_.back()));
        break;
      }
      case CustomSupportType::kTypeFloat: {
        (void)inputs_float_value_.emplace_back(device::ascend::ConvertKernelTensor<float>(input));
        (void)convert_inputs.emplace_back(&(inputs_float_value_.back()));
        break;
      }
      case CustomSupportType::kTypeDouble: {
        auto value = (input->dtype_id() == kNumberTypeFloat32)
                       ? static_cast<double>(device::ascend::ConvertKernelTensor<float>(input))
                       : device::ascend::ConvertKernelTensor<double>(input);
        (void)inputs_double_value_.emplace_back(value);
        (void)convert_inputs.emplace_back(&(inputs_double_value_.back()));
        break;
      }
      case CustomSupportType::kTypeInt: {
        (void)inputs_int_value_.emplace_back(device::ascend::ConvertKernelTensor<int64_t>(input));
        (void)convert_inputs.emplace_back(&inputs_int_value_.back());
        break;
      }
      case CustomSupportType::kTypeString: {
        (void)convert_inputs.emplace_back(const_cast<void *>(static_cast<const void *>(
          device::ascend::ConvertType(device::ascend::ConvertKernelTensor<std::string>(input)))));
        break;
      }
      case CustomSupportType::kTypeScalar: {
        auto scalar = device::ascend::ConvertKernelTensor<ScalarPtr>(input);
        (void)convert_inputs.emplace_back(device::ascend::ConvertType(scalar));
        break;
      }
      case CustomSupportType::kTypeIntArray: {
        auto int_vector = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(input);
        (void)convert_inputs.emplace_back(device::ascend::ConvertType(int_vector));
        break;
      }
      case CustomSupportType::kTypeBoolArray: {
        auto bool_vector = device::ascend::ConvertKernelTensor<std::vector<uint8_t>>(input);
        (void)convert_inputs.emplace_back(device::ascend::ConvertType(bool_vector));
        break;
      }
      case CustomSupportType::kTypeFloatArray: {
        auto float_vector = device::ascend::ConvertKernelTensor<std::vector<float>>(input);
        (void)convert_inputs.emplace_back(device::ascend::ConvertType(float_vector));
        break;
      }
      case CustomSupportType::kTypeDType: {
        auto value = input->GetValue();
        MS_EXCEPTION_IF_NULL(value);
        if (value->isa<Type>()) {
          auto type_id = value->cast<TypePtr>()->type_id();
          (void)inputs_type_value_.emplace_back(device::ascend::ConvertType(type_id));
          (void)convert_inputs.emplace_back(&inputs_type_value_.back());
          break;
        } else {
          MS_LOG(EXCEPTION) << "Kernel tensor' value  is not Type, but is " << value->ToString();
        }
      }
      default:
        MS_LOG(EXCEPTION) << "Custom unsupported input type: " << static_cast<int64_t>(type);
    }
  }
  return convert_inputs;
}

bool CustomV2AclnnKernelMod::CallGetWorkSpaceSize(const std::vector<std::vector<KernelTensor *>> &inputs,
                                                  const std::vector<std::vector<KernelTensor *>> &outputs,
                                                  uint64_t *workspace_size_addr, aclOpExecutor **executor_addr,
                                                  void *get_workspace_size_func) {
  inputs_bool_value_.reserve(inputs.size());
  inputs_float_value_.reserve(inputs.size());
  inputs_int_value_.reserve(inputs.size());
  inputs_double_value_.reserve(inputs.size());

  auto convert_inputs = ConvertTypes(inputs, 0);
  auto convert_outputs = ConvertTypes(outputs, inputs.size());

  converted_params_.clear();
  std::copy(convert_inputs.begin(), convert_inputs.end(), std::back_inserter(converted_params_));
  std::copy(convert_outputs.begin(), convert_outputs.end(), std::back_inserter(converted_params_));
  converted_params_.emplace_back(workspace_size_addr);
  converted_params_.emplace_back(executor_addr);

  std::string file_path;
  std::string func_type;
  const auto &exec_info = GetValue<std::string>(primitive_->GetAttr("custom_callback_func"));
  if (auto pos = exec_info.find(":"); pos != std::string::npos) {
    auto path = exec_info.substr(0, pos);
    auto real_path = FileUtils::GetRealPath(path.c_str());
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "For custom '" << op_type_ << "', couldn't find the AOT binary file under path: " << path;
    }
    file_path = real_path.value();
    func_type = exec_info.substr(pos + 1);
  } else {
    MS_LOG(EXCEPTION) << "For custom'" << op_type_ << "', user defined function path '" << exec_info << "' is illegal.";
  }

  auto handle_ = dlopen(file_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    MS_LOG(EXCEPTION) << "For custom '" << op_type_ << ", dlopen file '" << file_path
                      << "' should be successful, but error occurs! Error message is: " << dlerror();
  }

  using FuncType = int (*)(void *, std::vector<void *>, std::vector<void *>, uint64_t *, aclOpExecutor **);
  auto func_name = func_type + "GetWorkSpaceSize";
  auto user_func = reinterpret_cast<FuncType>(dlsym(handle_, func_name.c_str()));
  if (user_func != nullptr) {
    int ret = 0;
    try {
      ret = user_func(get_workspace_size_func, convert_inputs, convert_outputs, workspace_size_addr, executor_addr);
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "For custom '" << op_type_ << "', operator failed when executing user defined func " << func_name
                    << "! "
                    << "Error message is " << e.what();
      return false;
    }
    return ret;
  } else {
    MS_LOG(EXCEPTION) << "For custom " << op_type_ << ", can not find func " << func_name << " in " << file_path;
  }
}

CacheTuple CustomV2AclnnKernelMod::GenCustomExecutorForResize(const std::vector<std::vector<KernelTensor *>> &inputs,
                                                              const std::vector<std::vector<KernelTensor *>> &outputs) {
  auto workspace_api_name = op_type_ + "GetWorkspaceSize";
  const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());
  if (get_workspace_size_func_ptr == nullptr) {
    MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!";
  }
  uint64_t workspace_size = 0;
  device::ascend::aclOpExecutor *executor = nullptr;
  uint64_t *workspace_size_addr = &workspace_size;
  device::ascend::aclOpExecutor **executor_addr = &executor;
  auto workspace_status =
    CallGetWorkSpaceSize(inputs, outputs, workspace_size_addr, executor_addr, get_workspace_size_func_ptr);
  if (workspace_status != 0) {
    CHECK_AND_THROW_RECOVERABLE_ERROR(workspace_api_name);
    MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";
  }

  int32_t repeat_ret = device::ascend::SetExecutorRepeatable(workspace_api_name, executor);
  auto graph_cache = device::ascend::CustomGraphCache(executor, std::move(converted_params_), input_output_types_);
  auto process_cache = device::ascend::ProcessCache(graph_cache);
  return std::make_tuple(workspace_size, executor, process_cache, repeat_ret);
}

void CustomV2AclnnKernelMod::GetWorkspaceForResize(const std::vector<std::vector<KernelTensor *>> &inputs,
                                                   const std::vector<std::vector<KernelTensor *>> &outputs) {
  hash_id_ = device::ascend::CustomAclnnHash(op_type_, inputs, outputs, input_output_types_);
  size_t cur_workspace = 0;
  auto iter = hash_map_.find(hash_id_);
  if (iter != hash_map_.end()) {
    MS_LOG(DEBUG) << "op " << op_type_ << " hit cache with hash id: " << hash_id_;
    MS_VLOG(VL_CUSTOM_OP) << "op " << op_type_ << " hit cache with hash id: " << hash_id_;
    hash_cache_.splice(hash_cache_.begin(), hash_cache_, iter->second);
    cur_workspace = std::get<kWorkspaceIndex>(hash_cache_.front());
  } else {
    auto [workspace, executor, cache, fail_cache] = GenCustomExecutorForResize(inputs, outputs);
    cur_workspace = workspace;
    if (!fail_cache) {
      hash_cache_.emplace_front(hash_id_, executor, cache, workspace);
      hash_map_[hash_id_] = hash_cache_.begin();
      if (hash_cache_.size() > capacity_) {
        hash_map_.erase(std::get<0>(hash_cache_.back()));
        auto release_func = std::get<kReleaseFuncIndex>(hash_cache_.back());
        release_func(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
        hash_cache_.pop_back();
      }
    } else {
      hash_id_ = 0;
      cache(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
    }
  }

  if (cur_workspace != 0) {
    std::vector<size_t> workspace_size_list = {cur_workspace};
    SetWorkspaceSizeList(workspace_size_list);
  }
}

void CustomV2AclnnKernelMod::RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace,
                                   const std::vector<std::vector<KernelTensor *>> &inputs,
                                   const std::vector<std::vector<KernelTensor *>> &outputs) {
  auto [executor, release_func] = GetExecutor(inputs, outputs);
  if (workspace_size_list_.empty()) {
    RUN_CUSTOM_OP_API_ASYNC(op_type_, nullptr, 0, executor, stream_ptr, release_func);
  } else {
    if (workspace.empty()) {
      MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
    }
    auto workspace_tensor = workspace[0];
    if (workspace_tensor->size() != workspace_size_list_[0]) {
      MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                        << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
    }
    RUN_CUSTOM_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr,
                            release_func);
  }
}

std::vector<std::vector<void *>> CustomV2AclnnKernelMod::GetTensorAddress(
  const std::vector<std::vector<KernelTensor *>> &inputs, const std::vector<std::vector<KernelTensor *>> &outputs) {
  if (input_output_types_.size() != (inputs.size() + outputs.size())) {
    MS_LOG(EXCEPTION) << "Custom op " << op_type_ << " inputs type size " << input_output_types_.size()
                      << " is exceeds I/O size:" << inputs.size() + outputs.size();
  }
  std::vector<std::vector<KernelTensor *>> inputs_outputs;
  std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputs_outputs));
  std::copy(outputs.begin(), outputs.end(), std::back_inserter(inputs_outputs));
  std::vector<std::vector<void *>> address_list;
  for (size_t i = 0; i < inputs_outputs.size(); i++) {
    auto dyn_input = inputs_outputs[i];
    KernelTensor *input;
    if (dyn_input.empty()) {
      MS_LOG(EXCEPTION) << "Custom op [" << op_type_ << "] input-" << i << " is empty!";
    } else {
      input = dyn_input[0];
      MS_EXCEPTION_IF_NULL(input);
    }
    auto type = input_output_types_[i];
    MS_LOG(DEBUG) << "Get custom op [" << op_type_ << "] input-" << i
                  << " tensor address, input type: " << custom_supported_type_to_string.at(type);
    MS_VLOG(VL_CUSTOM_OP) << "Get custom op [" << op_type_ << "] input-" << i
                          << " tensor address, input type: " << custom_supported_type_to_string.at(type);
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        address_list.emplace_back(device::ascend::GetAddr(input));
        break;
      }
      case CustomSupportType::kTypeTensorList: {
        address_list.emplace_back(device::ascend::GetAddr(dyn_input));
        break;
      }
      case CustomSupportType::kTypeBool: {
        address_list.emplace_back(device::ascend::GetAddr(device::ascend::ConvertKernelTensor<bool>(input)));
        break;
      }
      case CustomSupportType::kTypeFloat: {
        address_list.emplace_back(device::ascend::GetAddr(device::ascend::ConvertKernelTensor<float>(input)));
        break;
      }
      case CustomSupportType::kTypeDouble: {
        auto value = (input->dtype_id() == kNumberTypeFloat32)
                       ? static_cast<double>(device::ascend::ConvertKernelTensor<float>(input))
                       : device::ascend::ConvertKernelTensor<double>(input);
        address_list.emplace_back(device::ascend::GetAddr(value));
        break;
      }
      case CustomSupportType::kTypeInt: {
        address_list.emplace_back(device::ascend::GetAddr(device::ascend::ConvertKernelTensor<int64_t>(input)));
        break;
      }
      case CustomSupportType::kTypeString: {
        address_list.emplace_back(device::ascend::GetAddr(device::ascend::ConvertKernelTensor<std::string>(input)));
        break;
      }
      case CustomSupportType::kTypeScalar: {
        auto scalar = device::ascend::ConvertKernelTensor<ScalarPtr>(input);
        address_list.emplace_back(device::ascend::GetAddr(scalar));
        break;
      }
      case CustomSupportType::kTypeIntArray:
      case CustomSupportType::kTypeBoolArray:
      case CustomSupportType::kTypeFloatArray: {
        std::vector<void *> addr{nullptr};
        address_list.emplace_back(addr);
        break;
      }
      case CustomSupportType::kTypeDType: {
        auto value = input->GetValue();
        MS_EXCEPTION_IF_NULL(value);
        if (value->isa<Type>()) {
          auto type_id = value->cast<TypePtr>()->type_id();
          (void)address_list.emplace_back(device::ascend::GetAddr(type_id));
          break;
        } else {
          MS_LOG(EXCEPTION) << "Kernel tensor' value  is not Type, but is " << value->ToString();
        }
      }
      default:
        MS_LOG(EXCEPTION) << "Custom unsupported input type: " << static_cast<int64_t>(type);
    }
  }
  return address_list;
}

void CustomV2AclnnKernelMod::UpdateTensorForLaunch(const std::vector<std::vector<KernelTensor *>> &inputs,
                                                   const std::vector<std::vector<KernelTensor *>> &outputs,
                                                   const ProcessCache &cache) {
  const auto &address_list = GetTensorAddress(inputs, outputs);
  cache(device::ascend::ProcessCacheType::kUpdateTensorAddress, address_list);
}

ExecutorTuple CustomV2AclnnKernelMod::GenCustomExecutor(const std::vector<std::vector<KernelTensor *>> &inputs,
                                                        const std::vector<std::vector<KernelTensor *>> &outputs) {
  auto workspace_api_name = op_type_ + "GetWorkspaceSize";
  static device::ascend::ApiCachePool api_cache_pool;
  const char *api_name = api_cache_pool.get(op_type_);
  const auto get_workspace_size_func_ptr = device::ascend::GetOpApiFunc(workspace_api_name.c_str());
  if (get_workspace_size_func_ptr == nullptr) {
    MS_LOG(EXCEPTION) << workspace_api_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!";
  }
  uint64_t workspace_size = 0;
  device::ascend::aclOpExecutor *executor = nullptr;
  std::function<void()> release_func = nullptr;
  uint64_t *workspace_size_addr = &workspace_size;
  device::ascend::aclOpExecutor **executor_addr = &executor;
  uint64_t new_hash_id;
  if (CustomHitCacheSingle(api_name, executor_addr, workspace_size_addr, &new_hash_id, inputs, outputs,
                           input_output_types_)) {
    MS_LOG(DEBUG) << "gen executor aclnn cache hit.";
    MS_VLOG(VL_CUSTOM_OP) << "gen executor aclnn cache hit.";
    return std::make_tuple(workspace_size, executor, release_func, new_hash_id, true);
  }
  MS_LOG(DEBUG) << "gen executor aclnn cache miss.";
  MS_VLOG(VL_CUSTOM_OP) << "gen executor aclnn cache miss.";
  auto init_mem_func = device::ascend::OpApiDefaultResource::GetInstance().init_mem_func();
  if (init_mem_func) {
    init_mem_func(nullptr, false);
  }

  auto workspace_status =
    CallGetWorkSpaceSize(inputs, outputs, workspace_size_addr, executor_addr, get_workspace_size_func_ptr);
  if (workspace_status != 0) {
    MS_LOG(EXCEPTION) << workspace_api_name << " call failed, please check!";
  }
  auto releas_call = device::ascend::CustomReleaseCall(std::move(converted_params_), input_output_types_);
  release_func = std::function<void()>(releas_call);
  auto uninit_mem_func = device::ascend::OpApiDefaultResource::GetInstance().uninit_mem_func();
  if (uninit_mem_func) {
    uninit_mem_func(nullptr, false);
  }
  device::ascend::UninitCacheThreadLocal();
  return std::make_tuple(workspace_size, executor, release_func, new_hash_id, false);
}

std::pair<aclOpExecutor *, std::function<void()>> CustomV2AclnnKernelMod::GetExecutor(
  const std::vector<std::vector<KernelTensor *>> &inputs, const std::vector<std::vector<KernelTensor *>> &outputs) {
  auto iter = hash_map_.find(hash_id_);
  if (capacity_ == 0 || hash_id_ == 0 || iter == hash_map_.end()) {
    aclOpExecutor *executor;
    std::function<void()> release_func;
    std::tie(std::ignore, executor, release_func, hash_id_, std::ignore) = GenCustomExecutor(inputs, outputs);
    return std::make_pair(executor, release_func);
  }
  const auto &cur_run = *(iter->second);
  UpdateTensorForLaunch(inputs, outputs, std::get<kReleaseFuncIndex>(cur_run));
  const auto &executor = std::get<1>(cur_run);
  return std::make_pair(executor, nullptr);
}

void CustomV2AclnnKernelMod::GetCustomInputTypes() {
  input_output_types_.clear();
  if (!primitive_->HasAttr(kCustomInputsType)) {
    MS_LOG(EXCEPTION) << "Can not find attribute [custom_inputs_type] for custom " << op_type_;
  }

  auto inputs_type_value = primitive_->GetAttr(kCustomInputsType);
  if (!inputs_type_value->isa<ValueList>()) {
    MS_LOG(EXCEPTION) << "For custom op [" << op_type_ << "], attribute [" << kCustomInputsType
                      << "] type should be ValueList, but get " << inputs_type_value->ToString();
  }

  auto inputs_type_value_list = inputs_type_value->cast<ValueListPtr>();
  auto input_type_value_list_value = inputs_type_value_list->value();

  for (const auto &item : input_type_value_list_value) {
    auto input_type = GetValue<std::string>(item);
    MS_LOG(DEBUG) << "Custom op " << op_type_ << " input type: " << input_type;
    MS_VLOG(VL_CUSTOM_OP) << "Custom op " << op_type_ << " input type: " << input_type;
    auto iter = string_to_custom_supported_type.find(input_type);
    if (iter == string_to_custom_supported_type.end()) {
      MS_LOG(EXCEPTION) << "Unsupported custom input type: " << input_type;
    }
    (void)input_output_types_.emplace_back(iter->second);
  }
}

CustomV2AclnnKernelMod::~CustomV2AclnnKernelMod() {
  converted_params_.clear();
  input_output_types_.clear();
  inputs_int_value_.clear();
  inputs_float_value_.clear();
  inputs_bool_value_.clear();
  inputs_double_value_.clear();
}

}  // namespace custom
}  // namespace kernel
}  // namespace mindspore
