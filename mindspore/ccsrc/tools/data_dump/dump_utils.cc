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
#include "tools/data_dump/dump_utils.h"

#include <dirent.h>
#ifdef ENABLE_DEBUGGER
#include <sys/stat.h>
#endif
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <stack>
#include <string>
#include <vector>

#include "mindspore/core/include/ir/tensor_new.h"
#include "tools/data_dump/dump_json_parser.h"
#include "utils/ms_context.h"
#ifdef ENABLE_DEBUGGER
#include "tools/data_dump/debugger/debugger.h"
#include "tools/tensor_data.h"
#endif
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/kernel_base/device_tensor_store.h"
#include "include/utils/anfalgo.h"
#include "include/utils/common.h"
#include "include/utils/utils.h"
#include "runtime/hardware_abstract/utils.h"
#include "utils/file_utils.h"

using mindspore::runtime::DeviceTensorStore;

#include "ir/tensor_new.h"
namespace mindspore {
static std::vector<std::string> g_overflow_operators;

bool SplitInt8ToInt4x2(const void *int4_data, size_t in_data_len, void *int8_data, size_t out_data_len) {
  if (in_data_len * 2 != out_data_len) {
    MS_LOG(ERROR) << "The input data length and output data length is not match, input data length: " << in_data_len
                  << ", output data length: " << out_data_len
                  << ". If sample_mode is set to 1, then sample_num must set to Integer multiples of 2 to save tensor "
                     "with int4 data type.";
    return false;
  }
  const int8_t *src_data = static_cast<const int8_t *>(int4_data);
  int8_t *dst_data = static_cast<int8_t *>(int8_data);
  for (size_t i = 0; i < in_data_len; ++i) {
    int8_t s = *src_data;
    int8_t t = s & 0xf;
    // keep the sign bit not change
    int8_t sign_bit = (t & 0x08) >> 3;
    if (sign_bit == 1) {
      t = t | 0xf0;
    } else if (sign_bit == 0) {
      t = t & 0x0f;
    } else {
      MS_LOG(ERROR) << "Error occur.";
      return false;
    }
    if (t < -8 || t > 7) {
      MS_LOG(ERROR) << "Error occurred when convert int4 to int8 data.";
      return false;
    }
    *dst_data = t;
    ++dst_data;
    t = s >> 4;
    sign_bit = (t & 0x08) >> 3;
    if (sign_bit == 1) {
      t = t | 0xf0;
    } else if (sign_bit == 0) {
      t = t & 0x0f;
    } else {
      MS_LOG(ERROR) << "Error occur.";
      return false;
    }
    if (t < -8 || t > 7) {
      MS_LOG(ERROR) << "Error occurred when convert int4 to int8 data.";
      return false;
    }
    *dst_data = t;
    ++dst_data;
    ++src_data;
  }
  return true;
}

std::string GenerateDumpPath(uint32_t graph_id, uint32_t rank_id, bool is_cst) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string net_name = dump_json_parser.net_name();
  std::string iterator = std::to_string(dump_json_parser.cur_dump_iter());
  std::string dump_path = dump_json_parser.path();
  if (dump_path.back() != '/') {
    dump_path += "/";
  }
  if (is_cst) {
    dump_path += ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/constants/");
  } else {
    dump_path +=
      ("rank_" + std::to_string(rank_id) + "/" + net_name + "/" + std::to_string(graph_id) + "/" + iterator + "/");
  }
  return dump_path;
}

void GetFileKernelName(NotNull<std::string *> kernel_name) {
  const std::string strsrc_to_replace[4] = {"/", "\\", ".", " "};
  const std::string strdst = "_";

  for (const auto &strsrc : strsrc_to_replace) {
    std::string::size_type pos = 0;
    std::string::size_type srclen = strsrc.size();
    std::string::size_type dstlen = strdst.size();
    while ((pos = kernel_name->find(strsrc, pos)) != std::string::npos) {
      kernel_name->replace(pos, srclen, strdst);
      pos += dstlen;
    }
  }
}

void GetDumpIntShape(const AnfNodePtr &node, size_t index, NotNull<ShapeVector *> const int_shapes, bool trans_flag) {
  if (trans_flag) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsValueNode<None>(node)) {
      return;
    }
    *int_shapes = AnfAlgo::GetRuntimePaddingShape(node, index);
  } else {
    *int_shapes = AnfAlgo::GetOutputDeviceShape(node, index);
  }
}

const kernel::KernelTensorPtr GetParameterInfo(const AnfNodePtr &node, NotNull<ShapeVector *> const int_shapes,
                                               NotNull<TypeId *> const host_type, NotNull<TypeId *> const device_type) {
  const auto &kernel_tensors = DeviceTensorStore::GetInstance().Fetch(node.get());
  if (kernel_tensors.size() < 1) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(kernel_tensors[0]);
  auto device_addr = kernel_tensors[0]->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool trans_flag = dump_json_parser.trans_flag();
  auto ref_node = device_addr->GetNodeIndex().first;
  MS_EXCEPTION_IF_NULL(ref_node);
  GetDumpIntShape(ref_node, kParameterOutputIndex, int_shapes, trans_flag);
  *host_type = common::AnfAlgo::GetOutputInferDataType(ref_node, kParameterOutputIndex);
  *device_type = AnfAlgo::GetOutputDeviceDataType(ref_node, kParameterOutputIndex);
  return kernel_tensors[0];
}

bool CPUDumpMemToFile(const kernel::KernelTensorPtr &kt, const std::string &filepath, const std::string &,
                      const ShapeVector &host_shape, TypeId host_type, bool) {
  MS_EXCEPTION_IF_NULL(kt);
  MS_EXCEPTION_IF_NULL(kt->device_address());
  const device::DeviceAddress &addr = *(kt->device_address());
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  std::string path = filepath + '.' + kernel::GetFormatFromEnumToStr(kt->format()) + "." + TypeIdToString(host_type);
  MS_LOG(DEBUG) << "E2E Dump path is " << path;
  if (addr.GetSize() == 0) {
    MS_VLOG(VL_DUMP) << "Data size is 0 for file: " << path << ", no need to dump.";
    return true;
  }
  if (addr.GetPtr() == nullptr) {
    MS_VLOG(VL_DUMP) << "Data is nullptr for file: " << path << ", skip it.";
    return true;
  }
  ret = DumpJsonParser::DumpToFile(path, addr.GetPtr(), addr.GetSize(), host_shape, host_type);
  if (!ret) {
    MS_LOG(ERROR) << "Dump to file failed";
  }
  return ret;
}

bool AscendDumpMemToFile(const kernel::KernelTensorPtr &kt, const std::string &filepath, const std::string &host_fmt,
                         const ShapeVector &host_shape, TypeId host_type, bool trans_flag) {
  MS_EXCEPTION_IF_NULL(kt);
  MS_EXCEPTION_IF_NULL(kt->device_address());
  const device::DeviceAddress &addr = *(kt->device_address());
  bool ret = false;
  if (filepath.empty()) {
    MS_LOG(ERROR) << "Dump file path is null!";
    return ret;
  }
  if (addr.GetSize() == 0) {
    MS_VLOG(VL_DUMP) << "the operator in filepath: " << filepath << ", size == 0";
    return true;
  }
  if (addr.GetPtr() == nullptr) {
    MS_VLOG(VL_DUMP) << "Data is nullptr for file: " << filepath << ", skip it.";
    return true;
  }
  device::DeviceContextKey host_key = {addr.GetDeviceType(), addr.device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->SyncAllStreams();
  if (trans_flag) {
    std::string path = filepath + '.' + host_fmt;
    MS_VLOG(VL_DUMP) << "E2E Dump path is " << path;
    if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin ||
        host_type == kNumberTypeComplex64) {
      MS_VLOG(VL_DUMP) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
      return false;
    }
    mindspore::tensor::TensorPtr out_tensor = tensor::from_spec(host_type, host_shape, device::DeviceType::kCPU);
    MS_EXCEPTION_IF_NULL(out_tensor);
    size_t host_size = LongToSize(out_tensor->DataNBytes());
    auto clone_device_address = std::make_shared<device::DeviceAddress>(addr.GetMutablePtr(), addr.GetSize(),
                                                                        addr.GetDeviceType(), addr.stream_id());
    MS_EXCEPTION_IF_NULL(out_tensor->device_address());
    // No need add device address info for same device address.
    ret = SyncCopy(out_tensor->device_address(), clone_device_address, addr.stream_id());
    if (!ret) {
      MS_LOG(ERROR) << "Copy device mem to host failed";
      return ret;
    }
    ret = DumpJsonParser::DumpToFile(path, out_tensor->data_c(), host_size, host_shape, host_type);
  } else {
    auto host_tmp = std::vector<uint8_t>(addr.GetSize());
    host_context->device_res_manager_->Copy(host_tmp.data(), addr.GetMutablePtr(), addr.GetSize(),
                                            device::CopyType::kD2H, addr.stream_id());
    std::string path = filepath + '.' + kernel::GetFormatFromEnumToStr(kt->format());
    MS_VLOG(VL_DUMP) << "E2E Dump path is " << path;
    ret = DumpJsonParser::DumpToFile(path, host_tmp.data(), addr.GetSize(), host_shape, kt->dtype_id());
  }
  return ret;
}

void DumpMemToFile(const std::string &file_path, const kernel::KernelTensorPtr &kt, const ShapeVector &int_shapes,
                   const TypeId &type, bool trans_flag) {
  MS_EXCEPTION_IF_NULL(kt);
  auto format = kOpFormat_DEFAULT;
  bool ret = false;
  if (kt->GetDeviceType() == device::DeviceType::kCPU) {
    ret = CPUDumpMemToFile(kt, file_path, format, int_shapes, type, trans_flag);
  } else if (kt->GetDeviceType() == device::DeviceType::kAscend) {
    ret = AscendDumpMemToFile(kt, file_path, format, int_shapes, type, trans_flag);
  }
  if (!ret) {
    MS_LOG(ERROR) << "DumpMemToFile Failed: flag:" << trans_flag << ", path:" << file_path << ", host_format:" << format
                  << ".!";
  }
}

#ifdef ENABLE_DEBUGGER
mindspore::tensor::TensorPtr LoadDeviceAddressToHost(kernel::KernelTensor *const kernel_tensor,
                                                     const std::string &tensor_name, const ShapeVector &host_shape,
                                                     TypeId host_type, bool trans_flag, bool async_copy) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor->device_address());
  const auto &addr = *(kernel_tensor->device_address());
  device::DeviceContextKey host_key = {kernel_tensor->GetDeviceType(), kernel_tensor->device_id()};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->SyncAllStreams();
  ShapeVector corrected_host_shape = host_shape;
  if (host_type == kNumberTypeInt4 && !corrected_host_shape.empty()) {
    constexpr int64_t kNumber2 = 2;
    corrected_host_shape.back() *= kNumber2;
  }
  mindspore::tensor::TensorPtr out_tensor =
    tensor::from_spec(host_type, corrected_host_shape, device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(out_tensor);
  size_t host_size = LongToSize(out_tensor->DataNBytes());
  if (host_size == 0) {
    MS_LOG(INFO) << "Tensor size is 0 for tensor: " << tensor_name;
    return std::make_shared<mindspore::tensor::Tensor>();
  }
  if (host_type == kNumberTypeInt4) {
    const int int4_nums_per_byte = 2;
    host_size = out_tensor->DataSize() / int4_nums_per_byte;
  }
  bool ret_sync = false;
  if (async_copy) {
    if (trans_flag) {
      MS_EXCEPTION_IF_NULL(out_tensor->device_address());
      MS_LOG(DEBUG) << "src kernel tensor:" << kernel_tensor->ToString()
                    << "dst device address:" << out_tensor->device_address()->ToString();
      ret_sync = SyncCopy(out_tensor, kernel_tensor, addr.stream_id());
    } else {
      ret_sync = host_context->device_res_manager_->Copy(out_tensor->data_c(), addr.GetMutablePtr(), host_size,
                                                         device::CopyType::kD2H, addr.stream_id());
    }
  } else {
    // copy device to host using sync mode
    auto ret = host_context->device_res_manager_->CopyDirectly(out_tensor->data_c(), host_size, addr.GetMutablePtr(),
                                                               addr.GetSize(), device::CopyType::kD2H);
    if (!ret) {
      MS_LOG(ERROR) << "SyncDeviceToHost fail, dst addr:" << out_tensor->data_c()
                    << " src addr:" << addr.GetMutablePtr() << " size:" << host_size;
      return nullptr;
    } else {
      ret_sync = true;
    }
  }
  if (!ret_sync) {
    MS_LOG(ERROR) << "Convert format or Copy device mem to host failed";
    return nullptr;
  }
  return out_tensor;
}

mindspore::tensor::TensorPtr ExtractContiguousTensor(const tensor::TensorPtr &ori_tensor, const ShapeVector &host_shape,
                                                     TypeId host_type, size_t storage_offset,
                                                     std::vector<int64_t> host_strides) {
  MS_EXCEPTION_IF_NULL(ori_tensor);
  if (host_shape.size() != host_strides.size()) {
    MS_LOG(ERROR) << "host_shape and host_strides must have the same dimension, but got " << host_shape.size() << " vs "
                  << host_strides.size();
    return nullptr;
  }
  if (host_shape.size() == 0) {
    MS_LOG(ERROR) << "host_shape is empty (dimension 0)";
    return nullptr;
  }
  int64_t dimension = static_cast<int64_t>(host_shape.size());
  mindspore::tensor::TensorPtr out_tensor = tensor::from_spec(host_type, host_shape, device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(out_tensor->device_address());
  MS_EXCEPTION_IF_NULL(ori_tensor->device_address());
  const char *src_data_ptr = static_cast<char *>(ori_tensor->data_c());
  char *dst_data_ptr = static_cast<char *>(out_tensor->data_c());
  auto element_size = ori_tensor->DataItemSize();
  size_t element_nums = SizeOf(host_shape);
  int64_t max_element_offset = static_cast<int64_t>(SizeOf(ori_tensor->shape()));
  std::vector<int64_t> indices(host_shape.size(), 0);
  for (size_t idx = 0; idx < element_nums; ++idx) {
    int64_t current_offset = static_cast<int64_t>(storage_offset);
    for (size_t dim = 0; dim < host_shape.size(); ++dim) {
      current_offset += indices[dim] * host_strides[dim];
    }
    if (current_offset >= max_element_offset) {
      MS_LOG(ERROR) << "Offset out of bounds: current_offset(" << current_offset << ") >= max_element_offset("
                    << max_element_offset << ")";
      return nullptr;
    }
    memcpy_s(dst_data_ptr + idx * element_size, element_size, src_data_ptr + current_offset * element_size,
             element_size);
    for (int dim = dimension - 1; dim >= 0; --dim) {
      indices[dim]++;
      if (indices[dim] < host_shape[dim]) {
        break;
      }
      indices[dim] = 0;
    }
  }
  return out_tensor;
}

bool LoadMemToHost(kernel::KernelTensor *const kernel_tensor, const std::string &tensor_name,
                   const std::string &host_fmt, const ShapeVector &host_shape, TypeId host_type, size_t slot,
                   bool keep_prev, uint32_t root_graph_id, bool force_update, bool trans_flag, bool async_copy) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  bool ret = false;
  if (kernel_tensor->GetSize() == 0) {
    MS_VLOG(VL_DUMP) << tensor_name << " size is 0, skip it.";
    return true;
  }
  if (kernel_tensor->device_ptr() == nullptr) {
    MS_VLOG(VL_DUMP) << tensor_name << " device address ptr is null, skip it.";
    return true;
  }
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (debugger->TensorExistsInCurrent(tensor_name) && !force_update) {
    MS_VLOG(VL_DUMP) << tensor_name << " already loaded for this step so not loading it again.";
    return true;
  }
  if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin || host_type == kNumberTypeComplex64) {
    MS_VLOG(VL_DUMP) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
    return false;
  }
  // For non-contiguous cases, dump the original tensor.
  auto correct_shape = host_shape;
  auto tensor_storage_info = kernel_tensor->tensor_storage_info();
  if (tensor_storage_info != nullptr) {
    MS_VLOG(VL_DUMP) << "Get dump value from non-contiguous Kernel Tensor:" << tensor_name;
    correct_shape = tensor_storage_info->ori_shape;
  }
  auto out_tensor =
    LoadDeviceAddressToHost(kernel_tensor, tensor_name, correct_shape, host_type, trans_flag, async_copy);
  // Convert to contiguous on host side.
  if (tensor_storage_info != nullptr) {
    MS_VLOG(VL_DUMP) << "Convert the non-contiguous Kernel Tensor:" << tensor_name << " to contiguous";
    out_tensor = ExtractContiguousTensor(out_tensor, host_shape, host_type, tensor_storage_info->storage_offset,
                                         tensor_storage_info->strides);
  }

  if (!out_tensor) {
    MS_LOG(ERROR) << tensor_name << " load mem to host failed.";
    return false;
  }
  if (!out_tensor->DataSize()) {
    MS_VLOG(VL_DUMP) << tensor_name << " datasize is 0, skip it.";
    return true;
  }
  std::string tensor_format = trans_flag ? host_fmt : kernel::GetFormatFromEnumToStr(kernel_tensor->format());
  size_t host_size = LongToSize(out_tensor->DataNBytes());
  if (host_type == kNumberTypeInt4) {
    const int int4_nums_per_byte = 2;
    host_size = out_tensor->DataSize() / int4_nums_per_byte;
  }
  auto tensor_data = std::make_shared<mindspore::TensorData>();
  MS_EXCEPTION_IF_NULL(tensor_data);
  tensor_data->SetName(tensor_name);
  tensor_data->SetSlot(slot);
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetDataPtr(static_cast<char *>(out_tensor->data_c()));
  tensor_data->SetByteSize(host_size);
  tensor_data->SetType(host_type);
  tensor_data->SetShape(out_tensor->shape());
  tensor_data->SetRootGraphId(root_graph_id);
  tensor_data->SetFormat(tensor_format);
  ret = debugger->LoadNewTensor(tensor_data, keep_prev);
  MS_VLOG(VL_DUMP) << "Load tensor '" << tensor_name << "' into debugger tensor loader successfully: format("
                   << tensor_format << ").";
  return ret;
}
#endif
}  // namespace mindspore
