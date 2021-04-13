/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "runtime/device/ascend/ascend_device_address.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include <set>
#include <algorithm>
#include "runtime/mem.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/convert_tensor_utils.h"
#include "ir/dtype/type.h"
#include "ir/tensor.h"
#include "abstract/utils.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_build.h"
#include "backend/kernel_compiler/tbe/tbe_kernel_parallel_build.h"
#include "utils/utils.h"
#include "common/trans.h"
#include "debug/data_dump/dump_json_parser.h"
#ifdef ENABLE_DEBUGGER
#include "debug/tensor_load.h"
#endif

namespace {
const std::unordered_map<mindspore::TypeId, std::string> type_id_name_map = {
  {mindspore::kNumberTypeBool, "bool"},       {mindspore::kNumberTypeInt8, "int8"},
  {mindspore::kNumberTypeInt16, "int16"},     {mindspore::kNumberTypeInt32, "int32"},
  {mindspore::kNumberTypeInt64, "int64"},     {mindspore::kNumberTypeFloat16, "float16"},
  {mindspore::kNumberTypeFloat32, "float32"}, {mindspore::kNumberTypeUInt8, "uint8"},
  {mindspore::kNumberTypeUInt16, "uint16"},   {mindspore::kNumberTypeUInt32, "uint32"},
  {mindspore::kNumberTypeUInt64, "uint64"}};
const std::set<std::pair<std::string, std::string>> use_trans_data = {
  std::make_pair("float16", mindspore::kOpFormat_NC1HWC0), std::make_pair("float32", mindspore::kOpFormat_NC1HWC0),
  std::make_pair("bool", mindspore::kOpFormat_NC1HWC0),    std::make_pair("float32", mindspore::kOpFormat_FRAC_Z),
  std::make_pair("float16", mindspore::kOpFormat_FRAC_Z),  std::make_pair("float16", mindspore::kOpFormat_FRAC_NZ),
  std::make_pair("float32", mindspore::kOpFormat_FRAC_NZ), std::make_pair("int32", mindspore::kOpFormat_FRAC_NZ),
  std::make_pair("float16", mindspore::kOpFormat_NHWC),    std::make_pair("float32", mindspore::kOpFormat_NHWC),
  std::make_pair("int8", mindspore::kOpFormat_NHWC),       std::make_pair("int16", mindspore::kOpFormat_NHWC),
  std::make_pair("int32", mindspore::kOpFormat_NHWC),      std::make_pair("int64", mindspore::kOpFormat_NHWC),
  std::make_pair("uint8", mindspore::kOpFormat_NHWC),      std::make_pair("uint16", mindspore::kOpFormat_NHWC),
  std::make_pair("uint32", mindspore::kOpFormat_NHWC),     std::make_pair("uint64", mindspore::kOpFormat_NHWC),
  std::make_pair("float16", mindspore::kOpFormat_HWCN),    std::make_pair("float32", mindspore::kOpFormat_HWCN),
  std::make_pair("int8", mindspore::kOpFormat_HWCN),       std::make_pair("int16", mindspore::kOpFormat_HWCN),
  std::make_pair("int32", mindspore::kOpFormat_HWCN),      std::make_pair("int64", mindspore::kOpFormat_HWCN),
  std::make_pair("uint8", mindspore::kOpFormat_HWCN),      std::make_pair("uint16", mindspore::kOpFormat_HWCN),
  std::make_pair("uint32", mindspore::kOpFormat_HWCN),     std::make_pair("uint64", mindspore::kOpFormat_HWCN)};
constexpr auto src_format = "src_format";
constexpr auto dst_format = "dst_format";
constexpr auto src = "src_0";
constexpr auto dst = "dst";
constexpr auto param_type_required = "required";
constexpr auto gen_model_single = "single";
constexpr auto trans_data = "trans_data";
constexpr auto platform_tbe = "TBE";
constexpr auto name = "name";
constexpr auto valid = "valid";
constexpr auto value = "value";
constexpr auto dtype = "dtype";
constexpr auto format_str = "format";
constexpr auto ori_format = "ori_format";
constexpr auto ori_shape = "ori_shape";
constexpr auto param_type = "param_type";
constexpr auto shape_str = "shape";
constexpr auto process_aicore = "aicore";
constexpr auto gen_model_str = "gen_model";
constexpr auto impl_path_str = "impl_path";
constexpr auto attrs_str = "attrs";
constexpr auto inputs_str = "inputs";
constexpr auto outputs_str = "outputs";
constexpr auto kernel_name_str = "kernel_name";
constexpr auto op_info_str = "op_info";
constexpr auto platform_str = "platform";
constexpr auto fractal_z = "FRACTAL_Z";
}  // namespace

namespace mindspore {
namespace device {
namespace ascend {
const int FLOAT_LEN = sizeof(float);
const int FLOAT16_LEN = 2;  // sizeof(float16);
const std::set<std::string> kOpNeedTransFormat = {
  kOpFormat_NHWC,    kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z,   kOpFormat_C1HWNCoC0,
  kOpFormat_FRAC_NZ, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D};

void SyncMemory(void *dst, const void *src, uint64_t size, rtMemcpyKind_t kind) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->SetContext();

  // Only apply asynchronous copy in Pynative && RT_MEMCPY_HOST_TO_DEVICE mode
  if (execution_mode != kPynativeMode || kind != RT_MEMCPY_HOST_TO_DEVICE) {
    auto ret_rt_memcpy = rtMemcpy(dst, size, src, size, kind);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "rtMemcpy failed";
    }
  } else {
    auto ret = runtime_instance->MemcpyAsync(dst, src, size, static_cast<int32_t>(kind));
    if (!ret) {
      MS_EXCEPTION(DeviceProcessError) << "MemcpyAsync failed";
    }
  }
}

bool FloatToHalfAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / FLOAT_LEN;
  if (elem_num != (dst_size / FLOAT16_LEN)) {
    MS_EXCEPTION(ArgumentError) << "FloatToHalf failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<float16> half_data(elem_num);
  FloatToHalf(half_data.data(), src, elem_num);
  SyncMemory(dst, half_data.data(), dst_size, RT_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool Float64ToFloatAndSyncHostToDevice(void *dst, size_t dst_size, const void *src, size_t src_size) {
  if (src_size / 2 != dst_size) {
    MS_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = dst_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  DoubleToFloat(host_tmp.data(), src, elem_num);
  SyncMemory(dst, host_tmp.data(), dst_size, RT_MEMCPY_HOST_TO_DEVICE);
  return true;
}

bool SyncDeviceToHostAndHalfToFloat(void *dst, size_t dst_size, const void *src, size_t src_size) {
  auto elem_num = src_size / FLOAT16_LEN;
  if (elem_num != (dst_size / FLOAT_LEN)) {
    MS_EXCEPTION(ArgumentError) << "HalfToFloat failed. size not match src_size[" << src_size << "], dst_size["
                                << dst_size << "]";
  }
  std::vector<float16> half_data(elem_num);
  SyncMemory(half_data.data(), src, src_size, RT_MEMCPY_DEVICE_TO_HOST);
  HalfToFloat(dst, half_data.data(), elem_num);
  return true;
}

bool SyncDeviceToHostAndFloatToFloat64(void *dst, size_t dst_size, const void *src, size_t src_size) {
  if (src_size != dst_size / 2) {
    MS_EXCEPTION(ArgumentError) << "src_size[" << src_size << "], dst_size[" << dst_size << "]";
  }
  size_t elem_num = src_size / sizeof(float);
  auto host_tmp = std::vector<float>(elem_num);
  SyncMemory(host_tmp.data(), src, src_size, RT_MEMCPY_DEVICE_TO_HOST);
  FloatToDouble(dst, host_tmp.data(), elem_num);
  return true;
}

DeviceAddressPtr AssignLaunchMemory(size_t size, const std::string &format, TypeId type) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto address_ptr = runtime_instance->AssignSingleOpLaunchMemory(size, format, type);
  return address_ptr;
}

size_t GetCommonAlignSize(size_t input_size) {
  return (input_size + kMemAlignSize + 31) / kMemAlignSize * kMemAlignSize;
}

nlohmann::json ConstructAttrs(const std::string &format) {
  nlohmann::json real_attr;
  nlohmann::json src_attr;
  nlohmann::json des_attr;
  src_attr[name] = src_format;
  src_attr[valid] = true;
  if (format == kOpFormat_FRAC_Z) {
    src_attr[value] = fractal_z;
  } else {
    src_attr[value] = format;
  }
  des_attr[name] = dst_format;
  des_attr[valid] = true;
  des_attr[value] = kOpFormat_NCHW;
  real_attr.push_back(src_attr);
  real_attr.push_back(des_attr);
  return real_attr;
}

nlohmann::json ConstructInputs(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape,
                               const std::string &format, mindspore::TypeId type) {
  nlohmann::json input;
  nlohmann::json input_json;
  nlohmann::json real_input;
  real_input[dtype] = type_id_name_map.at(type);
  if (format == kOpFormat_FRAC_Z) {
    real_input[format_str] = fractal_z;
  } else {
    real_input[format_str] = format;
  }
  real_input[name] = src;
  real_input[ori_format] = kOpFormat_NCHW;
  for (auto shape : output_shape) {
    real_input[ori_shape].push_back(shape);
  }
  real_input[param_type] = param_type_required;
  // obtain inputs shape
  for (auto shape : input_shape) {
    real_input[shape_str].push_back(shape);
  }
  real_input[valid] = true;
  input_json.push_back(real_input);
  input.push_back(input_json);
  return input;
}

nlohmann::json ConstructOutputs(const std::vector<size_t> &output_shape, mindspore::TypeId type) {
  nlohmann::json output;
  nlohmann::json output_json;
  nlohmann::json real_output;
  real_output[dtype] = type_id_name_map.at(type);
  real_output[format_str] = kOpFormat_NCHW;
  real_output[name] = dst;
  real_output[ori_format] = kOpFormat_NCHW;
  for (auto shape : output_shape) {
    real_output[ori_shape].push_back(shape);
  }
  real_output[param_type] = param_type_required;
  // obtain outputs shape
  for (auto shape : output_shape) {
    real_output[shape_str].push_back(shape);
  }
  real_output[valid] = true;
  output_json.push_back(real_output);
  output.push_back(output_json);
  return output;
}

nlohmann::json ConstructTransDataKernelJson(const std::vector<size_t> &host_shape,
                                            const std::vector<size_t> &device_shape, const std::string &format,
                                            mindspore::TypeId type) {
  // generate kernel json
  nlohmann::json kernel_json;
  kernel_json[gen_model_str] = gen_model_single;
  kernel_json[impl_path_str] = "";
  // construct op_info
  nlohmann::json op_info;
  op_info[attrs_str] = ConstructAttrs(format);
  op_info[inputs_str] = ConstructInputs(device_shape, host_shape, format, type);
  op_info[kernel_name_str] = "";
  op_info[name] = trans_data;
  op_info[outputs_str] = ConstructOutputs(host_shape, type);
  // construct soc_info
  nlohmann::json soc_info;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto tune_mode = ms_context->get_param<std::string>(MS_CTX_TUNE_MODE);
  soc_info["autoTilingMode"] = tune_mode;
  kernel_json["SocInfo"] = soc_info;
  kernel_json[op_info_str] = op_info;
  kernel_json[platform_str] = platform_tbe;
  std::string json_str = kernel_json[op_info_str].dump();
  size_t hash_id = std::hash<std::string>()(json_str);
  const std::string op_name = op_info[name];
  const std::string json_name = op_name + "_" + std::to_string(hash_id);
  kernel_json[op_info_str][kernel_name_str] = json_name;
  return kernel_json;
}

void AscendDeviceAddress::SyncStream() const {
  MS_LOG(INFO) << "Start!";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      !ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
    MS_LOG(INFO) << "Finish!";
    return;
  }
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret = runtime_instance->SyncStream();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync stream error!";
  }
  MS_LOG(INFO) << "Finish!";
}

bool AscendDeviceAddress::SyncDeviceToHost(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           void *host_ptr) const {
  MS_LOG(INFO) << "SyncDeviceToHost, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  SyncStream();
  bool sync_ok = false;
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), LongToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NCDHW) {
    if (type_id_ == type) {
      SyncMemory(host_ptr, ptr_, size, RT_MEMCPY_DEVICE_TO_HOST);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = SyncDeviceToHostAndFloatToFloat64(host_ptr, size, ptr_, size_);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      auto host = std::vector<uint8_t>(size_);
      SyncMemory(host.data(), ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
      const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size_};
      sync_ok = trans::TransDataType(type_args, host_ptr);
      if (!sync_ok) {
        MS_LOG(ERROR) << "trans data type failed.";
        return false;
      }
    }
  } else {
    auto iter = kOpNeedTransFormat.find(format_);
    if (iter != kOpNeedTransFormat.end()) {
      sync_ok = SyncDeviceToHostAndConvertFormat(shape, size, type, host_ptr);
    } else {
      MS_LOG(INFO) << "Can not find format transfer for :" << format_;
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Not support to trans, dev_format:" << format_ << ", dev_type:" << TypeIdLabel(type_id_)
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

void AscendDeviceAddress::LaunchTransData(kernel::KernelModPtr kernel_mod_ptr, void *output_address_ptr,
                                          size_t output_size, const std::vector<size_t> &workspace_size_list) const {
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  auto input_address = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(input_address);
  input_address->addr = ptr_;
  input_address->size = size_;
  auto output_address = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(output_address);
  output_address->addr = output_address_ptr;
  output_address->size = output_size;
  AddressPtrList kernel_inputs = {input_address};
  AddressPtrList kernel_outputs = {output_address};
  AddressPtrList kernel_workspaces;
  std::vector<DeviceAddressPtr> workspace_address_ptr(workspace_size_list.size());
  if (!workspace_size_list.empty()) {
    for (size_t i = 0; i < workspace_size_list.size(); ++i) {
      auto workspace_size = GetCommonAlignSize(workspace_size_list[i]);
      workspace_address_ptr[i] = AssignLaunchMemory(workspace_size, "", kTypeUnknown);
      MS_EXCEPTION_IF_NULL(workspace_address_ptr[i]);
      auto workspace_address = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace_address);
      workspace_address->addr = workspace_address_ptr[i]->GetMutablePtr();
      workspace_address->size = workspace_address_ptr[i]->GetSize();
      kernel_workspaces.push_back(workspace_address);
    }
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  auto ret =
    runtime_instance->LaunchTaskBasedOnSingleKernel(kernel_mod_ptr, kernel_inputs, kernel_outputs, kernel_workspaces);
  if (!ret) {
    MS_LOG(ERROR) << "Launch kernel failed.";
  }
  SyncStream();
}

kernel::KernelModPtr AscendDeviceAddress::CompileTransDataAndObtainKernelMod(const nlohmann::json &kernel_json) const {
  static std::set<std::string> constructed_kernel;
  auto build_manager = std::make_shared<kernel::ParallelBuildManager>();
  MS_EXCEPTION_IF_NULL(build_manager);
  std::string processor = process_aicore;
  // get size
  std::vector<size_t> input_size_list;
  std::vector<size_t> output_size_list;
  (void)kernel::TbeKernelBuild::GetIOSize(kernel_json, &input_size_list, &output_size_list, nullptr);
  std::string json_name = kernel_json[op_info_str][kernel_name_str];
  // op build
  if (constructed_kernel.find(json_name) == constructed_kernel.end()) {
    auto task_id = build_manager->StartCompileOp(kernel_json);
    build_manager->SaveTaskInfo(task_id, nullptr, json_name, input_size_list, output_size_list);
  }
  while (!build_manager->IsAllTaskFinish()) {
    int task_id = -1;
    std::string task_result;
    std::string build_result;
    auto ret = build_manager->WaitOne(&task_id, &task_result, &build_result);
    if (!ret) {
      MS_EXCEPTION(ArgumentError) << "Build Failed. wait one ret:" << ret << ", task id:" << task_id;
    }
    if (task_result != "Success") {
      MS_EXCEPTION(ArgumentError) << "task compile Failed, task id:" << task_id << ", cause:" << task_result;
    }
    (void)build_manager->TaskFinishProcess(task_id, build_result, false);
  }
  constructed_kernel.insert(json_name);
  // search cache
  auto cached_kernel_pack = TbeUtils::SearchCache(json_name, processor);
  MS_EXCEPTION_IF_NULL(cached_kernel_pack);
  auto kernel_mod_ptr =
    build_manager->GenKernelMod(json_name, processor, input_size_list, output_size_list, cached_kernel_pack);
  return kernel_mod_ptr;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormatBasedOnTransData(const std::vector<size_t> &host_shape,
                                                                           const std::vector<size_t> &device_shape,
                                                                           size_t size, mindspore::TypeId type,
                                                                           void *host_ptr) const {
  bool sync_ok = true;
  // construct trans data kernel json
  nlohmann::json kernel_json = ConstructTransDataKernelJson(host_shape, device_shape, format_, type_id_);
  MS_LOG(INFO) << "Construct trans_data kernel json: " << kernel_json.dump();
  auto kernel_mod_ptr = CompileTransDataAndObtainKernelMod(kernel_json);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  auto host_size = size;
  if (type_id_ != type) {
    auto device_dtype_size = abstract::TypeIdSize(type_id_);
    if (device_dtype_size < 1) {
      MS_LOG(ERROR) << "Illegal dtype.";
    }
    auto shape_size = abstract::ShapeSize(host_shape);
    size = device_dtype_size * shape_size;
  }
  size = GetCommonAlignSize(size);
  auto output_address = AssignLaunchMemory(size, kOpFormat_NCHW, type_id_);
  MS_EXCEPTION_IF_NULL(output_address);
  auto workspace_size_list = GetWorkspaceSizeList(kernel_json);
  // launch
  LaunchTransData(kernel_mod_ptr, output_address->GetMutablePtr(), output_address->GetSize(), workspace_size_list);
  if (type_id_ == type) {
    SyncMemory(host_ptr, output_address->GetPtr(), host_size, RT_MEMCPY_DEVICE_TO_HOST);
  } else {
    auto host = std::vector<uint8_t>(size);
    SyncMemory(host.data(), output_address->GetPtr(), size, RT_MEMCPY_DEVICE_TO_HOST);
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, host_size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  }
  return sync_ok;
}

std::vector<size_t> AscendDeviceAddress::GetWorkspaceSizeList(const nlohmann::json &kernel_json) const {
  std::string json_name = kernel_json[op_info_str][kernel_name_str];
  std::string processor = process_aicore;
  auto cached_kernel_pack = TbeUtils::SearchCache(json_name, processor);
  MS_EXCEPTION_IF_NULL(cached_kernel_pack);
  auto kernel_json_info = cached_kernel_pack->kernel_json_info();
  return kernel_json_info.workspaces;
}

std::vector<size_t> AscendDeviceAddress::GetDeviceShape(std::vector<size_t> *host_shape) const {
  std::vector<size_t> device_shape;
  if (format_ == kOpFormat_FRAC_NZ || format_ == kOpFormat_NCDHW) {
    device_shape = trans::TransShapeToDevice(*host_shape, format_);
  } else {
    if (host_shape_.empty()) {
      *host_shape = trans::PaddingShape(*host_shape, format_);
    } else {
      host_shape->clear();
      (void)std::transform(host_shape_.begin(), host_shape_.end(), std::back_inserter(*host_shape), LongToSize);
    }
    device_shape = trans::TransShapeToDevice(*host_shape, format_);
  }
  return device_shape;
}

bool AscendDeviceAddress::SyncDeviceToHostAndConvertFormat(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, void *host_ptr) const {
  MS_LOG(INFO) << "SyncDeviceToHostAndConvertFormat, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  bool sync_ok = false;
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), LongToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  std::vector<size_t> device_shape = GetDeviceShape(&host_shape);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kGraphMode &&
      ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      type_id_name_map.find(type_id_) != type_id_name_map.end()) {
    std::pair<std::string, std::string> type_format = std::make_pair(type_id_name_map.at(type_id_), format_);
    if (use_trans_data.find(type_format) != use_trans_data.end()) {
      sync_ok = SyncDeviceToHostAndConvertFormatBasedOnTransData(host_shape, device_shape, size, type, host_ptr);
      return sync_ok;
    }
  }
  auto host_tmp = std::vector<uint8_t>(size_);
  SyncMemory(host_tmp.data(), ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
  if (type_id_ != type) {
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    auto host = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host.data(), shape_size, type_id_, type, size};
    sync_ok = trans::TransDataType(type_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  } else {
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    sync_ok = trans::TransFormatFromDeviceToHost(format_args, host_ptr);
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
  }
  return sync_ok;
}

bool AscendDeviceAddress::SyncHostToDevice(const ShapeVector &shape, size_t size, mindspore::TypeId type,
                                           const void *host_ptr) const {
  MS_LOG(INFO) << "SyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }

  bool sync_ok = false;
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), LongToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  if (format_ == kOpFormat_NCHW || format_ == kOpFormat_DEFAULT || format_ == kOpFormat_NCDHW) {
    if (type_id_ == type) {
      SyncMemory(ptr_, host_ptr, size, RT_MEMCPY_HOST_TO_DEVICE);
      sync_ok = true;
    } else if (type_id_ == kNumberTypeFloat32 && type == kNumberTypeFloat64) {
      sync_ok = Float64ToFloatAndSyncHostToDevice(ptr_, size_, host_ptr, size);
    } else {
      auto shape_size = abstract::ShapeSize(host_shape);
      const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id_, size};
      auto host_tmp = std::vector<uint8_t>(size_);
      sync_ok = trans::TransDataType(type_args, host_tmp.data());
      if (!sync_ok) {
        MS_LOG(ERROR) << "Trans data type failed.";
        return false;
      }
      SyncMemory(ptr_, host_tmp.data(), size_, RT_MEMCPY_HOST_TO_DEVICE);
    }
  } else {
    auto iter = kOpNeedTransFormat.find(format_);
    if (iter != kOpNeedTransFormat.end()) {
      sync_ok = ConvertFormatAndSyncHostToDevice(shape, size, type, host_ptr);
    } else {
      MS_LOG(INFO) << "Can not find format transfer for :" << format_;
    }
  }
  if (!sync_ok) {
    MS_LOG(ERROR) << "Not support to trans, dev_format:" << format_ << ", dev_type:" << TypeIdLabel(type_id_)
                  << ", host_type:" << TypeIdLabel(type);
    return false;
  }
  return sync_ok;
}

bool AscendDeviceAddress::ConvertFormatAndSyncHostToDevice(const ShapeVector &shape, size_t size,
                                                           mindspore::TypeId type, const void *host_ptr) const {
  bool sync_ok = false;
  MS_LOG(INFO) << "ConvertFormatAndSyncHostToDevice, Device(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
               << ", size:" << size_ << "), Host(type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  std::vector<size_t> host_shape;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(host_shape), LongToSize);
  if (host_shape.empty()) {
    host_shape.emplace_back(1);
  }
  std::vector<size_t> device_shape;
  if (format_ == kOpFormat_FRAC_NZ) {
    device_shape = trans::TransShapeToDevice(host_shape, format_);
  } else {
    host_shape = trans::PaddingShape(host_shape, format_);
    device_shape = trans::TransShapeToDevice(host_shape, format_);
  }
  if (type_id_ != type) {
    auto shape_size = abstract::ShapeSize(host_shape);
    const trans::TypeIdArgs type_args{host_ptr, shape_size, type, type_id_, size};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransDataType(type_args, host_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans datatype failed.";
      return false;
    }
    const trans::FormatArgs format_args{host_tmp.data(), size_,        kOpFormat_NCHW, format_,
                                        host_shape,      device_shape, type_id_};
    auto dst_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, dst_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    SyncMemory(ptr_, dst_tmp.data(), size_, RT_MEMCPY_HOST_TO_DEVICE);
  } else {
    const trans::FormatArgs format_args{host_ptr, size_, kOpFormat_NCHW, format_, host_shape, device_shape, type_id_};
    auto host_tmp = std::vector<uint8_t>(size_);
    sync_ok = trans::TransFormat(format_args, host_tmp.data());
    if (!sync_ok) {
      MS_LOG(ERROR) << "Trans format failed.";
      return false;
    }
    SyncMemory(ptr_, host_tmp.data(), size_, RT_MEMCPY_HOST_TO_DEVICE);
  }
  return sync_ok;
}

void AscendDeviceAddress::ClearDeviceMemory() {
  if (ptr_ == nullptr) {
    return;
  }
  if (from_mem_pool_) {
    if (communication_ptr_ != nullptr) {
      AscendMemoryPool::GetInstance().FreeTensorMem(communication_ptr_);
      communication_ptr_ = nullptr;
    } else {
      AscendMemoryPool::GetInstance().FreeTensorMem(ptr_);
    }
    ptr_ = nullptr;
  }
}

AscendDeviceAddress::~AscendDeviceAddress() { ClearDeviceMemory(); }

bool AscendDeviceAddress::DumpMemToFile(const std::string &filepath, const std::string &host_fmt,
                                        const ShapeVector &host_shape, TypeId host_type, bool trans_flag) const {
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  std::string shape = "shape";
  if (host_shape.size()) {
    for (auto &value : host_shape) {
      shape = shape + '_' + std::to_string(value);
    }
  } else {
    shape = shape + "_0";
  }
  std::string file_extension = ".bin";
  if (trans_flag) {
    std::string path =
      filepath + '_' + shape + '_' + TypeIdToType(host_type)->ToString() + '_' + host_fmt + file_extension;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
    size_t host_size = out_tensor->data().nbytes();
    ret = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
    if (!ret) {
      MS_LOG(ERROR) << "Copy device mem to host failed";
      return ret;
    }
    ret = DumpJsonParser::DumpToFile(path, out_tensor->data_c(), host_size);
  } else {
    auto host_tmp = std::vector<uint8_t>(size_);
    auto ret_rt_memcpy = rtMemcpy(host_tmp.data(), size_, ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "SyncDeviceToHost: rtMemcpy mem size[" << size_ << "] fail, ret[" << ret_rt_memcpy << "]";
    }
    std::string path =
      filepath + '_' + shape + '_' + TypeIdToType(type_id_)->ToString() + '_' + format_ + file_extension;
    MS_LOG(INFO) << "E2E Dump path is " << path;
    ret = DumpJsonParser::DumpToFile(path, host_tmp.data(), size_);
  }

  return ret;
}

#ifdef ENABLE_DEBUGGER
bool AscendDeviceAddress::LoadMemToHost(const std::string &tensor_name, int execution_order,
                                        const std::string &host_fmt, const ShapeVector &host_shape, TypeId host_type,
                                        size_t slot, bool keep_prev) const {
  bool ret = false;
  if (Debugger::GetInstance()->TensorExistsInCurrent(tensor_name)) {
    MS_LOG(INFO) << tensor_name << " already loaded for this step so not loading it again.";
    return true;
  }
  // TensorData is freed up in AscendSession class
  auto tensor_data = std::make_shared<mindspore::TensorData>();
  tensor_data->SetName(tensor_name);
  tensor_data->SetExecutionOrder(execution_order);
  tensor_data->SetSlot(slot);
  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(type_id_, host_shape);
  size_t host_size = out_tensor->data().nbytes();
  auto ret_rt_memcpy = rtMemcpy(out_tensor->data_c(), host_size, ptr_, host_size, RT_MEMCPY_DEVICE_TO_HOST);
  if (ret_rt_memcpy != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "SyncDeviceToHost: rtMemcpy mem size[" << size_ << "] fail, ret[" << ret_rt_memcpy << "]";
  }
  MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
  tensor_data->SetTensor(out_tensor);
  ret = Debugger::GetInstance()->LoadNewTensor(tensor_data, keep_prev);
  return ret;
}
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
