/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "pybind_api/pynative/tensor/tensor_py.h"

#include <utility>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "pybind11/complex.h"
#include "include/utils/convert_utils_py.h"

#include "utils/cache_embedding_hashmap_struct.h"
#include "include/utils/python_adapter.h"
#include "tools/profiler/profiler.h"
#include "include/utils/pynative/adapter.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/pipeline/pipeline.h"
#include "pybind_api/pynative/tensor/mbuf_device_address.h"
#include "include/backend/ms_backend/runtime/graph_scheduler/base/move_to.h"
#include "utils/value_utils.h"
#include "device_address/device_address_maker.h"
#include "ir/tensor_new.h"
#include "include/pynative/utils/runtime/op_runner.h"
#include "utils/ms_utils_secure.h"
#include "utils/misc.h"
#include "utils/stream_guard.h"

namespace mindspore {
namespace tensor {
namespace {
struct TensorToNumpyRegister {
  TensorToNumpyRegister() { python_adapter::PyAdapterCallback::SetTensorToNumpyHandler(tensor::AsNumpy); }
} callback_register;

TensorPtr MakeCpuTensor(const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &device_address = tensor->device_address();
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_address->GetDeviceType(), device_address->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  if (!device_context->device_res_manager_->SyncAllStreams()) {
    MS_LOG(EXCEPTION) << "SyncStream failed in Offload.";
  }
  auto cpu_tensor = tensor::from_spec_fast(tensor->data_type(), tensor->shape_c(), device::DeviceType::kCPU);
  if (!SyncCopy(cpu_tensor, tensor, CurrentStream::id())) {
    MS_LOG(EXCEPTION) << "Offload failed. Copy data from device to host failed. Src:" << device_address->ToString()
                      << " Dst:" << cpu_tensor->device_address()->ToString();
  }
  return cpu_tensor;
}
}  // namespace

bool TensorPybind::IsPinned(const tensor::TensorPy &tensor) {
  const auto &base_tensor = tensor.GetTensor();
  if (base_tensor->device_address() == nullptr) {
    MS_LOG(INFO) << "In IsPinned function. device_address is nullptr.";
    return false;
  }
  const auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(base_tensor->device_address());
  const auto allocator = device_address->allocator();
  if (device_address->allocator() == nullptr) {
    MS_LOG(INFO) << "In IsPinned function. allocator is nullptr.";
    return false;
  }
  if (allocator->IsPinned()) {
    return true;
  }
  return false;
}

bool TensorPybind::IsShared(const tensor::TensorPy &tensor) {
  const auto &base_tensor = tensor.GetTensor();
  if (base_tensor->device_address() == nullptr) {
    MS_LOG(INFO) << "In IsShared function. device_address is nullptr.";
    return false;
  }
  const auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(base_tensor->device_address());
  if (device_address->GetDeviceType() == device::DeviceType::kAscend) {
    MS_LOG(WARNING) << "Tensor.is_shared() will always return true for Ascend tensor. However, reduce_tensor method "
                       "in mindspore.multiprocessing module does not support Ascend tensor currently, so inter-process "
                       "communication(IPC) is not available for NPU tensor.";
    return true;
  }
  if (device_address->map_allocator()) {
    return true;
  }
  return false;
}

TensorPtr TensorPybind::MakePinMemoryTensor(const tensor::TensorPy &tensor) {
  const auto &base_tensor = tensor.GetTensor();
  const auto &shape = base_tensor->shape();
  const auto &dtype = base_tensor->data_type();
  auto tensor_size = LongToSize(base_tensor->DataNBytes());

  const auto &base_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(base_tensor->device_address());
  if (device::IsAscendDeviceType(base_device_address->GetDeviceType())) {
    MS_LOG(EXCEPTION) << "Only CPU tensor can be pinned.The source tensor should be CPU tensor.";
  }

  auto device_address = DeviceAddressMaker(nullptr, dtype, shape)
                          .set_maker(GetDeviceAddressMaker(device::DeviceType::kCPU))
                          .make_device_address();
  auto ascend_device_ctx = runtime::OpRunner::GetDeviceContext(device::DeviceType::kAscend);
  if (ascend_device_ctx == nullptr || ascend_device_ctx->device_res_manager_ == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot find Ascend device context. ascend_device_ctx or device_res_manager is null.";
  }
  auto pin_memory_allocator = ascend_device_ctx->device_res_manager_->pin_mem_allocator();
  std::dynamic_pointer_cast<device::DeviceAddress>(device_address)->set_allocator(pin_memory_allocator);
  auto device_ctx = runtime::OpRunner::GetDeviceContext(device::DeviceType::kCPU);
  bool allocate_mem_ret = device_ctx->device_res_manager_->AllocateMemory(
    std::dynamic_pointer_cast<device::DeviceAddress>(device_address).get());
  if (!allocate_mem_ret) {
    MS_LOG(EXCEPTION) << "Tensor.pin_memory allocate memory failed!";
  }

  const void *pin_data_ptr = std::dynamic_pointer_cast<device::DeviceAddress>(device_address)->GetPtr();
  // H2H copy
  auto memcpy_ret = common::huge_memcpy(reinterpret_cast<uint8_t *>(const_cast<void *>(pin_data_ptr)), tensor_size,
                                        reinterpret_cast<uint8_t *>(base_tensor->data_c()), tensor_size);
  if (memcpy_ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy failed!";
  }

  TensorPtr tensorPtrRes = std::make_shared<Tensor>(dtype, shape, device_address);
  tensorPtrRes->set_need_pipeline_sync(true);
  return tensorPtrRes;
}

void TensorPybind::SetUserData(const TensorPtr &tensor, const py::str &key, const py::object &value) {
  const std::string name = key.cast<std::string>();
  const auto &primitive_data = std::make_shared<TensorPyUserData>();
  primitive_data->obj = value;
  const_cast<TensorPtr &>(tensor)->set_user_data<TensorPyUserData>(name, primitive_data);
}

py::object TensorPybind::GetUserData(const TensorPtr &tensor, const py::str &key) {
  const std::string name = key.cast<std::string>();
  const auto primitive_data = tensor->user_data<TensorPyUserData>(name);
  if (primitive_data == nullptr) {
    return py::none();
  }
  return primitive_data->obj;
}

static std::vector<ssize_t> GetStrides(const std::vector<ssize_t> &shape, ssize_t item_size) {
  std::vector<ssize_t> strides;
  strides.reserve(shape.size());
  const auto ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto stride = item_size;
    for (size_t j = i + 1; j < ndim; ++j) {
      stride *= shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

py::tuple TensorPybind::GetPyTupleShape(const Tensor &tensor) {
  auto &shape = tensor.shape();
  py::tuple dims(shape.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape[i]);
  }
  return dims;
}

py::tuple TensorPybind::GetPyTupleStrides(const Tensor &tensor) {
  std::vector<ssize_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor.DataItemSize());
  py::tuple py_strides(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    py_strides[i] = py::int_(strides[i]);
  }
  return py_strides;
}

py::int_ TensorPybind::GetPyItemSize(const Tensor &tensor) { return tensor.DataItemSize(); }

py::int_ TensorPybind::GetPyNBytes(const Tensor &tensor) { return tensor.DataNBytes(); }

template <typename T>
void MemCopyFromCacheToHost(void *hashmap_addr, void *host_addr, void *cache_addr, size_t host_max, size_t cache_max,
                            size_t hashmap_size, size_t col_size) {
  auto host_data = static_cast<char *>(host_addr);
  auto cache_data = static_cast<char *>(cache_addr);
  auto hashmap_data = static_cast<HashmapEntry<T> *>(hashmap_addr);
  // default param type float
  const size_t param_type_size = 4;
  size_t single_col_bytes = param_type_size * col_size;
  for (size_t i = 0; i < hashmap_size; ++i) {
    if (!hashmap_data[i].IsEmpty()) {
      size_t host_offset = single_col_bytes * LongToSize(hashmap_data[i].key_);
      size_t cache_offset = single_col_bytes * LongToSize(hashmap_data[i].value_);
      if (cache_offset + single_col_bytes <= cache_max) {
        auto ret =
          memcpy_s(host_data + host_offset, host_max - host_offset, cache_data + cache_offset, single_col_bytes);
        if (ret != 0) {
          MS_LOG(EXCEPTION) << "Memcpy failed.";
        }
      }
    }
  }
  MS_LOG(INFO) << "Memcpy from cache to host success!";
}

void TensorPybind::FlushFromCache(const Tensor &tensor) {
  py::gil_scoped_release gil_release;
  tensor::TensorPtr cpu_tensor = tensor.cpu();
  if (cpu_tensor->cache_enable()) {
    MS_LOG(INFO) << cpu_tensor->ToString() << " is cache enable.";
    auto hashmap_tensor_ptr = cpu_tensor->hashmap_tensor_ptr();
    auto cache_tensor_ptr = cpu_tensor->cache_tensor_ptr();
    if (hashmap_tensor_ptr != nullptr && cache_tensor_ptr != nullptr) {
      hashmap_tensor_ptr = hashmap_tensor_ptr->cpu();
      cache_tensor_ptr = cache_tensor_ptr->cpu();
      auto hashmap_size = hashmap_tensor_ptr->shape_c()[0];
      auto host_shape = cpu_tensor->shape_c();
      auto cache_shape = cache_tensor_ptr->shape_c();
      if (host_shape.size() != 2 && cache_shape.size() != 2 && host_shape[1] != cache_shape[1]) {
        MS_LOG(EXCEPTION) << "Got host shape and cache shape invalid."
                          << "host shape:" << host_shape << ", cache shape:" << cache_shape;
      }
      auto host_data_max_size = static_cast<size_t>(cpu_tensor->Size());
      auto cache_data_max_size = static_cast<size_t>(cache_tensor_ptr->Size());
      auto hashmap_data_type = hashmap_tensor_ptr->data_type();
      if (hashmap_data_type == TypeId::kNumberTypeInt32) {
        MemCopyFromCacheToHost<int32_t>(hashmap_tensor_ptr->data_c(), cpu_tensor->data_c(), cache_tensor_ptr->data_c(),
                                        host_data_max_size, cache_data_max_size, hashmap_size, host_shape[1]);
      } else if (hashmap_data_type == TypeId::kNumberTypeInt64) {
        MemCopyFromCacheToHost<int32_t>(hashmap_tensor_ptr->data_c(), cpu_tensor->data_c(), cache_tensor_ptr->data_c(),
                                        host_data_max_size, cache_data_max_size, hashmap_size, host_shape[1]);
      } else {
        MS_LOG(ERROR) << "Hashmap dtype only suppotr int32, in64.";
      }
    }
  }
}

py::bytes TensorPybind::GetBytes(const Tensor &tensor) {
  py::gil_scoped_acquire acquire;
  if (tensor.get_copy_done_flag()) {
    const_cast<Tensor &>(tensor).set_copy_done_flag(false);
    return py::bytes(static_cast<const char *>(tensor.data_c()), tensor.Size());
  }
  auto cpu_tensor = tensor.cpu();
  return py::bytes(static_cast<const char *>(cpu_tensor->data_c()), cpu_tensor->Size());
}

void CopyFromBuffer(char *dst, size_t dst_size, const char *src, size_t src_size, TypeId data_type) {
  bool fp16_in_fp32 = (data_type == TypeId::kNumberTypeBFloat16) && (dst_size * 2 == src_size);
  if (fp16_in_fp32) {
    int elem_num = static_cast<int>(src_size / sizeof(float));
    for (int i = 0; i < elem_num; ++i) {
      auto dst_ptr = static_cast<char *>(dst + i * sizeof(bfloat16));
      auto src_ptr = static_cast<const char *>(src + sizeof(bfloat16) + i * sizeof(float));
      errno_t ret = memcpy_s(dst_ptr, sizeof(bfloat16), src_ptr, sizeof(bfloat16));
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy the memory to new tensor:" << ret;
      }
    }
  } else {
    size_t remain_size = src_size;
    auto dst_ptr = dst;
    auto src_ptr = src;
    while (remain_size > SECUREC_MEM_MAX_LEN) {
      auto ret = memcpy_s(dst_ptr, SECUREC_MEM_MAX_LEN, src_ptr, SECUREC_MEM_MAX_LEN);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy the memory to new tensor" << ret;
      }
      remain_size -= SECUREC_MEM_MAX_LEN;
      dst_ptr += SECUREC_MEM_MAX_LEN;
      src_ptr += SECUREC_MEM_MAX_LEN;
    }
    if (remain_size != 0U) {
      auto ret = memcpy_s(dst_ptr, remain_size, src_ptr, remain_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy the memory to new tensor" << ret;
      }
    }
  }
}

TensorPtr TensorPybind::ConvertBytesToTensor(const py::bytes &bytes_obj, const py::tuple &dims,
                                             const TypePtr &type_ptr) {
  ShapeVector shape;
  for (size_t i = 0; i < dims.size(); ++i) {
    shape.push_back(dims[i].cast<int>());
  }
  TypeId data_type = type_ptr ? type_ptr->type_id() : TypeId::kTypeUnknown;
  tensor::TensorPtr tensor = tensor::from_spec(data_type, shape, device::DeviceType::kCPU);
  const char *tensor_buf = PYBIND11_BYTES_AS_STRING(bytes_obj.ptr());
  char *tensor_data_buf = reinterpret_cast<char *>(tensor->data_c());
  CopyFromBuffer(tensor_data_buf, tensor->Size(), tensor_buf, PYBIND11_BYTES_SIZE(bytes_obj.ptr()), data_type);
  return tensor;
}

py::object GetItemForToList(void *data, const TypeId &data_type, const int &index) {
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return py::int_(py::cast(*(static_cast<int8_t *>(data) + index)));
    case TypeId::kNumberTypeUInt8:
      return py::int_(py::cast(*(static_cast<uint8_t *>(data) + index)));
    case TypeId::kNumberTypeInt16:
      return py::int_(py::cast(*(static_cast<int16_t *>(data) + index)));
    case TypeId::kNumberTypeUInt16:
      return py::int_(py::cast(*(static_cast<uint16_t *>(data) + index)));
    case TypeId::kNumberTypeInt:
    case TypeId::kNumberTypeInt32:
      return py::int_(py::cast(*(static_cast<int *>(data) + index)));
    case TypeId::kNumberTypeUInt32:
      return py::int_(py::cast(*(static_cast<uint32_t *>(data) + index)));
    case TypeId::kNumberTypeInt64:
      return py::int_(py::cast(*(static_cast<int64_t *>(data) + index)));
    case TypeId::kNumberTypeUInt64:
      return py::int_(py::cast(*(static_cast<uint64_t *>(data) + index)));
    case TypeId::kNumberTypeFloat16:
      return py::float_(py::cast(*(static_cast<float16 *>(data) + index)));
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32:
      return py::float_(py::cast(*(static_cast<float *>(data) + index)));
    case TypeId::kNumberTypeDouble:
    case TypeId::kNumberTypeFloat64:
      return py::float_(py::cast(*(static_cast<double *>(data) + index)));
    case TypeId::kNumberTypeBFloat16:
      return py::float_(py::cast(*(static_cast<bfloat16 *>(data) + index)));
    case TypeId::kNumberTypeBool:
      return py::bool_(py::cast(*(static_cast<bool *>(data) + index)));
    case TypeId::kNumberTypeComplex64:
    case TypeId::kNumberTypeComplex:
      return py::cast(
        std::complex<double>{*(static_cast<float *>(data) + index * 2), *(static_cast<float *>(data) + 1 + index * 2)});
    case TypeId::kNumberTypeComplex128:
      return py::cast(std::complex<long double>{*(static_cast<double *>(data) + index * 2),
                                                *(static_cast<double *>(data) + 1 + index * 2)});
    default:
      MS_EXCEPTION(TypeError) << "Not support tensor data type: " << data_type << ".";
      break;
  }
  return py::none();
}

py::object RecursiveToList(void *data, const std::vector<int64_t> &shape, const TypeId &data_type, int *index,
                           int cur_dim) {
  int ndim = static_cast<int>(shape.size());
  py::list res_list;
  if (cur_dim == ndim) {
    auto elem = GetItemForToList(data, data_type, *index);
    (*index) += 1;
    return elem;
  }
  for (int i = 0; i < shape[cur_dim]; ++i) {
    py::object obj = RecursiveToList(data, shape, data_type, index, cur_dim + 1);
    res_list.append(obj);
  }
  return res_list;
}

py::object TensorPybind::ToList(const TensorPtr &tensor) {
  tensor::TensorPtr cpu_tensor = tensor->cpu();
  auto tensor_shape = cpu_tensor->shape();
  auto data = cpu_tensor->data_c();
  auto data_type = cpu_tensor->data_type();
  int index = 0;

  return RecursiveToList(data, tensor_shape, data_type, &index, 0);
}

py::object TensorPybind::Item(const TensorPtr &tensor) {
  auto data_type = tensor->data_type();
  switch (data_type) {
    case TypeId::kNumberTypeInt8:
      return py::int_(py::cast(TensorItem<int8_t>(tensor)));
    case TypeId::kNumberTypeUInt8:
      return py::int_(py::cast(TensorItem<uint8_t>(tensor)));
    case TypeId::kNumberTypeInt16:
      return py::int_(py::cast(TensorItem<int16_t>(tensor)));
    case TypeId::kNumberTypeUInt16:
      return py::int_(py::cast(TensorItem<uint16_t>(tensor)));
    case TypeId::kNumberTypeInt:
    case TypeId::kNumberTypeInt32:
      return py::int_(py::cast(TensorItem<int32_t>(tensor)));
    case TypeId::kNumberTypeUInt32:
      return py::int_(py::cast(TensorItem<uint32_t>(tensor)));
    case TypeId::kNumberTypeInt64:
      return py::int_(py::cast(TensorItem<int64_t>(tensor)));
    case TypeId::kNumberTypeUInt64:
      return py::int_(py::cast(TensorItem<uint64_t>(tensor)));
    case TypeId::kNumberTypeFloat16:
      return py::float_(py::cast(TensorItem<float16>(tensor)));
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32:
      return py::float_(py::cast(TensorItem<float>(tensor)));
    case TypeId::kNumberTypeDouble:
    case TypeId::kNumberTypeFloat64:
      return py::float_(py::cast(TensorItem<double>(tensor)));
    case TypeId::kNumberTypeBFloat16:
      return py::float_(py::cast(TensorItem<bfloat16>(tensor)));
    case TypeId::kNumberTypeBool:
      return py::bool_(py::cast(TensorItem<bool>(tensor)));
    case TypeId::kNumberTypeComplex64:
    case TypeId::kNumberTypeComplex:
      return py::cast(std::complex<double>{TensorItem<std::complex<float>>(tensor)});
    case TypeId::kNumberTypeComplex128:
      return py::cast(std::complex<long double>{TensorItem<std::complex<double>>(tensor)});
    default:
      MS_EXCEPTION(TypeError) << "Not support tensor data type: " << data_type << ".";
      break;
  }
  return py::none();
}

py::array TensorPybind::SyncAsNumpy(const Tensor &tensor) {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kAsnumpy);
  // Asnumpy should be a read-only operation and should not modify the original Tensor.
  if (tensor.need_pipeline_sync()) {
    runtime::Pipeline::Get().WaitAll();
  }
  TensorPtr tensor_for_copy = std::make_shared<Tensor>(tensor);
  {
    py::gil_scoped_release gil_release;

    // BFloat16 may not be supported in numpy, detail numpy version info is notified in IsCustomNumpyTypeValid.
    if (tensor.data_type() == kNumberTypeBFloat16 && !IsCustomNumpyTypeValid(true)) {
      MS_EXCEPTION(TypeError) << "The Numpy bfloat16 data type is not supported now, please ensure that the current "
                              << "Numpy version is not less than the version when the mindspore is compiled, "
                              << "and the major versions are same.";
    }

    // To be deleted
    if (!tensor.get_copy_done_flag()) {
      tensor_for_copy = tensor_for_copy->cpu();
    }
    const_cast<Tensor &>(tensor).set_copy_done_flag(false);
  }
  return AsNumpy(*tensor_for_copy);
}

TensorPtr TensorPybind::FromDLPack(const py::object &dlpack_capsule) {
  DLManagedTensor *dlpack = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(dlpack_capsule.ptr(), "dltensor"));
  if (dlpack == nullptr) {
    MS_LOG(EXCEPTION) << "from_dlpack received an invalid capsule. Note that DLTensor capsules can be consumed only "
                         "once, so you might have already constructed a tensor from it once.";
  }
  PyCapsule_SetName(dlpack_capsule.ptr(), "used_dltensor");
  return DLPackUtils::FromDLPack(dlpack);
}

static void DLPackDestructor(PyObject *data) {
  if (MS_LIKELY(!PyCapsule_IsValid(data, "dltensor"))) {
    return;
  }
  DLManagedTensor *dlMTensor = reinterpret_cast<DLManagedTensor *>(PyCapsule_GetPointer(data, "dltensor"));
  dlMTensor->deleter(dlMTensor);
}

py::object TensorPybind::ToDLPack(const py::object &src) {
  TensorPtr tensor = tensor::ConvertToTensor(src);
  DLManagedTensor *dlpack = DLPackUtils::ToDLPack(tensor);
  return py::reinterpret_steal<py::object>(PyCapsule_New(dlpack, "dltensor", DLPackDestructor));
}

void TensorPybind::Offload(const TensorPtr &tensor, bool release) {
  py::gil_scoped_release gil_release;

  const auto &device_sync = tensor->device_address();
  if (device_sync == nullptr) {
    MS_LOG(WARNING) << "Tensor without DeviceSync can not be offloaded.";
    return;
  }
  const auto &device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  if (device_address == nullptr) {
    MS_LOG(WARNING) << "Tensor without DeviceAddress can not be loaded.";
    return;
  }
  if (device_address->GetDeviceType() == device::DeviceType::kCPU) {
    MS_LOG(WARNING) << "Tensor with CPUDeviceAddress can not be offloaded.";
    return;
  }
  if (device_address->GetPtr() == nullptr) {
    MS_LOG(WARNING) << "For Offload, this tensor's device_ptr is nullptr, it may have been offloaded or released by"
                    << " the framework.";
    return;
  }
  MS_LOG(INFO) << "Tensor Offload start, the tensor's device_address is : " << device_address.get()
               << ", the tensor's size is : " << device_address->GetSize();

  auto cpu_tensor = MakeCpuTensor(tensor);
  tensor->set_device_address(cpu_tensor->device_address());
  tensor->set_format(cpu_tensor->format());
  tensor->set_storage_info(cpu_tensor->storage_info());
  if (release) {
    device_address->ClearDeviceMemory();
  }
}

void TensorPybind::Load(const Tensor &tensor) {
  py::gil_scoped_release gil_release;
  const auto &device_address = tensor.device_address();
  if (device_address == nullptr) {
    MS_LOG(WARNING) << "Tensor without DeviceAddress can not be loaded.";
    return;
  }

  if (device_address->GetDeviceType() != device::DeviceType::kCPU) {
    MS_LOG(DEBUG) << "No need to load, because the data is already on device:"
                  << device::GetDeviceNameByType(device_address->GetDeviceType());
    return;
  }

  auto ms_context = MsContext::GetInstance();
  const auto &device_target = device::GetDeviceTypeByName(ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_target, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  // make sure op execute end before data copy
  runtime::Pipeline::Get().WaitForward();
  auto new_device_address = MakeDeviceAddress(tensor.data_type(), tensor.shape(), false, device_target);

  device_ctx->Initialize();
  device_ctx->device_res_manager_->AllocateMemory(new_device_address.get());
  MS_LOG(INFO) << "Tensor Load start, the tensor's device_address is : " << new_device_address.get()
               << ", the tensor's size is : " << new_device_address->GetSize();
  // Same device address do not need to set metadata.
  if (!device_ctx->device_res_manager_->SyncAllStreams() ||
      !SyncCopy(new_device_address, device_address, new_device_address->stream_id())) {
    MS_LOG(EXCEPTION) << "Failed to sync copy for tensor pybind load:" << tensor.ToString();
  }
  const_cast<tensor::Tensor &>(tensor).set_device_address(new_device_address);
  const_cast<tensor::Tensor &>(tensor).set_format(Format::DEFAULT_FORMAT);
  const_cast<tensor::Tensor &>(tensor).set_storage_info(nullptr);
}

bool TensorPybind::SharedMemory(const TensorPtr &tensor) {
  runtime::Pipeline::Get().WaitForward();
  MS_LOG(INFO) << "Tensor shared memory allocation start";

  // get tensor info
  auto data_type = tensor->data_type();
  auto shape = tensor->shape_c();
  const auto &device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());

  // get shared mem allocator
  std::string device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name), device_id});
  auto shared_mem_allocator = device_context->device_res_manager_->shared_mem_allocator();
  if (shared_mem_allocator == nullptr) {
    MS_LOG(ERROR) << "Shared memory allocator is null!";
    return false;
  }

  if (!device_address) {
    MS_LOG(WARNING) << "Tensor does not have a device address, creation failed";
    return false;
  }

  auto device_type = device_address->GetDeviceType();
  MS_LOG(INFO) << "Src tensor ptr: " << device_address->GetPtr();

  if (device_type == device::DeviceType::kCPU) {
    // create device address
    auto device_address_ = DeviceAddressMaker(nullptr, data_type, shape)
                             .set_maker(GetDeviceAddressMaker(device::DeviceType::kAscend))
                             .make_device_address();

    MS_EXCEPTION_IF_NULL(device_address_);
    tensor->set_device_address(device_address_);
    tensor->set_format(Format::DEFAULT_FORMAT);
    tensor->set_storage_info(nullptr);
    const auto &device_address_new = std::dynamic_pointer_cast<device::DeviceAddress>(device_address_);
    device_address_new->set_allocator(shared_mem_allocator);

    if (device_address->GetPtr()) {
      if (!device_context->device_res_manager_->AllocateMemory(device_address_new.get())) {
        MS_LOG(WARNING) << "Tensor shared memory allocation error";
        return false;
      }

      auto new_host_ptr = shared_mem_allocator->GetHostPtrByDevicePtr(const_cast<void *>(device_address_new->GetPtr()));
      MS_LOG(INFO) << "Target tensor device ptr: " << device_address_new->GetPtr();
      MS_LOG(INFO) << "Target tensor host ptr: " << new_host_ptr;

      auto ret =
        memcpy_s(new_host_ptr, device_address_new->GetSize(), device_address->GetPtr(), device_address->GetSize());
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "memcpy_s errno: " << ret;
        return false;
      }
    }
  } else if (device_address->GetDeviceType() == device::DeviceType::kAscend) {
    MS_LOG(WARNING) << "Tensor is on Ascend device, creation failed";
    return false;
  }
  MS_LOG(INFO) << "Tensor shared memory allocation end";
  return true;
}

void TensorPybind::SetDeviceAddress(const TensorPtr &tensor, uintptr_t addr, const ShapeVector &shape,
                                    const TypePtr type_ptr) {
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    MS_LOG(EXCEPTION) << "set_device_address now only support Ascend backend!";
  }

  if (type_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Dtype to be set is nullptr.";
  }

  TypeId data_type = type_ptr->type_id();
  if (data_type != tensor->data_type()) {
    MS_LOG(EXCEPTION) << "Dtype to be set is not equal with the tensor's, then tensor's dtype is"
                      << tensor->data_type();
  }

  if (shape != tensor->shape()) {
    MS_LOG(EXCEPTION) << "Shape to be set is not equal with the tensor's, then tensor's shape is" << tensor->shape();
  }

  void *data = reinterpret_cast<void *>(addr);
  size_t elem_num = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    elem_num *= shape[i];
  }
  auto data_size = elem_num * GetDataTypeSize(data_type);
  auto device_sync_ = tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_sync_);
  if (device_sync_->GetDeviceType() != device::DeviceType::kAscend) {
    auto device_address = std::make_shared<device::MbufDeviceAddress>(data, data_size, shape, data_type, kAscendDevice);
    const_cast<TensorPtr &>(tensor)->set_device_address(device_address);
    const_cast<TensorPtr &>(tensor)->set_format(Format::DEFAULT_FORMAT);
    const_cast<TensorPtr &>(tensor)->set_storage_info(nullptr);
  } else {
    device_sync_->set_ptr(data);
  }
}

uintptr_t TensorPybind::DataPtr(const TensorPtr &tensor) {
  runtime::Pipeline::Get().WaitForward();
  const auto device_address = tensor->device_address();
  if (device_address == nullptr) {
    MS_LOG(INFO) << "Tensor device address is null";
    return reinterpret_cast<uintptr_t>(nullptr);
  }
  auto *data_ptr = device_address->GetMutablePtr();
  MS_LOG(DEBUG) << "Get Tensor data ptr " << data_ptr;
  auto base_addr = reinterpret_cast<uintptr_t>(data_ptr);
  auto offset = static_cast<uintptr_t>(tensor->storage_offset() * abstract::TypeIdSize(tensor->data_type()));
  return base_addr + offset;
}

std::string TensorPybind::GetDevice(const TensorPtr &tensor) {
  runtime::Pipeline::Get().WaitBackend();
  const auto &device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  if (device_address == nullptr) {
    return "CPU";
  }
  if (device_address->GetDeviceType() == device::DeviceType::kCPU) {
    return "CPU";
  }
  return device::GetDeviceNameByType(device_address->GetDeviceType()) + ":" +
         std::to_string(device_address->device_id());
}

TensorPtr TensorPybind::MoveTo(const Tensor &self, const std::string &to, bool blocking) {
  py::gil_scoped_release gil_release;
  // make sure op execute end before data copy
  runtime::Pipeline::Get().WaitForward();
  MS_LOG(INFO) << "Try move tensor to " << to;
  const auto &device = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (to != "CPU" && to != device) {
    MS_LOG(EXCEPTION) << "The value of arg 'to' of method 'move_to' should be same with device target, bug got to:"
                      << to << ", device target: " << device;
  }
  auto target_tensor = tensor::from_spec(self.data_type(), self.shape(), device::GetDeviceTypeByName(to));
  bool return_self = false;
  device::MoveTo(std::make_shared<tensor::Tensor>(self), target_tensor, to, blocking, &return_self);
  if (return_self) {
    return std::make_shared<tensor::Tensor>(self);
  }
  return target_tensor;
}

py::object TensorPyImpl::GetInitializerFromPython(const py::dict &input) {
  if (!input.contains("init") || py::isinstance<py::none>(input["init"])) {
    return py::none();
  }
  return input["init"];
}

bool TensorPyImpl::GetConstArgFromPython(const py::dict &input) {
  if (!input.contains("const_arg") || py::isinstance<py::none>(input["const_arg"])) {
    return false;
  }
  py::object obj = input["const_arg"];
  if (!PyBool_Check(obj.ptr())) {
    MS_EXCEPTION(TypeError) << "For 'Tensor', the type of 'const_arg' should be 'bool', but got type '"
                            << obj.get_type() << "'.";
  }
  return obj.cast<bool>();
}

std::string TensorPyImpl::GetDeviceFromPython(const py::dict &input) {
  if (!input.contains("device") || py::isinstance<py::none>(input["device"])) {
    return "";
  }
  py::object obj = input["device"];
  if (!py::isinstance<py::str>(obj)) {
    MS_EXCEPTION(TypeError) << "For 'Tensor', the device should be string, but got '" << obj.get_type() << "'.";
  }
  std::string device = py::str(obj);
  if (std::strncmp(device.c_str(), "CPU", std::strlen("CPU")) != 0) {
    MS_EXCEPTION(ValueError) << "Only 'CPU' is supported for device, but got '" << device << "'.";
  }
  return device;
}

py::object TensorPyImpl::GetSymbolicShapeFromPython(const py::dict &input) {
  if (!input.contains("symbolic_shape")) {
    return py::none();
  }
  py::object obj = input["symbolic_shape"];
  if (py::isinstance<py::none>(obj) || !py::isinstance<py::list>(obj)) {
    return py::none();
  }
  py::list obj_list = py::cast<py::list>(obj);
  if (obj_list.empty()) {
    return py::none();
  }
  return obj;
}

const TypePtr TensorPyImpl::GetDtypeFromPython(const py::dict &input) {
  if (!input.contains("dtype")) {
    return nullptr;
  }
  py::object obj = input["dtype"];
  if (py::isinstance<py::none>(obj)) {
    return nullptr;
  }
  if (!py::isinstance<Type>(obj)) {
    MS_EXCEPTION(TypeError)
      << "For 'Tensor', the 'dtype' should be one of [mindspore.int8, mindspore.int16, mindspore.int32, "
      << "mindspore.int64, mindspore.uint8, mindspore.uint16, mindspore.uint32, mindspore.uint64, mindspore.float16, "
      << "mindspore.float32, mindspore.float64, mindspore.bfloat16, mindspore.complex64, mindspore.complex128, "
      << "mindspore.int4, mindspore.bool_, mindspore.string_], but got '" << obj.get_type() << "'.";
  }
  return obj.cast<TypePtr>();
}

const ShapeVector TensorPyImpl::GetShapeFromPython(const py::dict &input) {
  ShapeVector shape;
  if (!input.contains("shape")) {
    return shape;
  }
  py::object obj = input["shape"];
  if (!py::isinstance<ShapeVector>(obj)) {
    MS_EXCEPTION(TypeError) << "For 'Tensor', the 'shape' should be one of [list, tuple], but got '" << obj.get_type()
                            << "'.";
  }
  shape = obj.cast<ShapeVector>();
  return shape;
}

TensorPtr TensorPyImpl::InitTensorByInputDta(const py::dict &input, const TypePtr &dtype) {
  py::object input_obj = input["input_data"];
  if (IsTensorPy(input_obj)) {
    TensorPtr obj = ConvertToTensor(input_obj);
    TypeId data_type = dtype != nullptr ? dtype->type_id() : kTypeUnknown;
    if (data_type == kTypeUnknown || obj->data_type() == data_type) {
      return std::make_shared<Tensor>(*obj);
    }
    return std::make_shared<Tensor>(*obj, data_type);
  }

  if (py::isinstance<py::float_>(input_obj) || py::isinstance<py::int_>(input_obj) ||
      py::isinstance<py::list>(input_obj) || py::isinstance<py::tuple>(input_obj) ||
      PyComplex_CheckExact(input_obj.ptr()) || py::isinstance<py::bytes>(input_obj)) {
    return tensor::MakeTensor(py::array(input_obj), dtype);
  }

  if (py::isinstance<py::array>(input_obj)) {
    return tensor::MakeTensor(input_obj, dtype);
  }

  return nullptr;
}

TensorPtr TensorPyImpl::InitTensorByShape(const py::dict &input, const TypePtr &dtype) {
  if (input.contains("shape") &&
      (py::isinstance<py::list>(input["shape"]) || py::isinstance<py::tuple>(input["shape"]))) {
    TypeId data_type = dtype != nullptr ? dtype->type_id() : TypeId::kNumberTypeFloat64;
    if (input.contains("init") && !py::isinstance<py::none>(input["init"])) {
      return tensor::from_spec(data_type, GetShapeFromTuple(input["shape"]), device::DeviceType::kNone);
    } else {
      return tensor::from_spec(data_type, GetShapeFromTuple(input["shape"]), device::DeviceType::kCPU);
    }
  }
  ShapeVector shape = GetShapeFromPython(input);
  TypeId data_type = dtype != nullptr ? dtype->type_id() : kTypeUnknown;
  return tensor::from_spec(data_type, shape, device::DeviceType::kCPU);
}

TensorPtr TensorPyImpl::InitTensor(const py::dict &input) {
  TypePtr dtype = GetDtypeFromPython(input);
  TensorPtr output = nullptr;
  if (input.contains("input_data") && (!py::isinstance<py::none>(input["input_data"]))) {
    output = InitTensorByInputDta(input, dtype);
  } else {
    output = InitTensorByShape(input, dtype);
  }
  MS_EXCEPTION_IF_NULL(output);
  return output;
}

const TensorPyPtr TensorPyImpl::InitTensorPy(const py::dict &input) {
  TensorPtr tensor = InitTensor(input);
  TensorPyPtr tensorpy = std::make_shared<TensorPy>(tensor);
  tensorpy->SetInitializer(GetInitializerFromPython(input));
  tensorpy->SetConstArg(GetConstArgFromPython(input));
  tensorpy->SetDevice(GetDeviceFromPython(input));
  tensorpy->SetSymbolicShape(GetSymbolicShapeFromPython(input));
  tensorpy->SetInitFinished(true);
  return tensorpy;
}

TensorPyPtr TensorPyImpl::ConvertBytesToTensor(const py::bytes &bytes_obj, const py::tuple &dims,
                                               const TypePtr &type_ptr) {
  auto tensor = TensorPybind::ConvertBytesToTensor(bytes_obj, dims, type_ptr);
  MS_EXCEPTION_IF_NULL(tensor);
  return std::make_shared<TensorPy>(tensor);
}

TensorPyPtr TensorPyImpl::FromDLPack(const py::object &dlpack_capsule) {
  auto tensor = TensorPybind::FromDLPack(dlpack_capsule);
  MS_EXCEPTION_IF_NULL(tensor);
  return std::make_shared<TensorPy>(tensor);
}

py::object TensorPyImpl::ToDLPack(const py::object &tensor) { return TensorPybind::ToDLPack(tensor); }

py::object TensorPyImpl::Item(const TensorPyPtr &tensorpy) {
  auto tensor = tensorpy->GetTensor();
  return TensorPybind::Item(tensor);
}

void TensorPyImpl::RemoveTensorBackwardHook(uint64_t handle_id) {
  pynative::HookAdapter::RemoveTensorBackwardHook(handle_id);
}

ShapeVector TensorPyImpl::GetShapeFromTuple(const py::tuple &tuple) {
  ShapeVector shape;
  const size_t size = tuple.size();
  shape.reserve(tuple.size());
  for (size_t i = 0; i < size; ++i) {
    shape.push_back(py::int_(tuple[i]));
  }
  return shape;
}

template <typename T>
py::tuple GetSparseTensorShape(const T &sparse_tensor) {
  auto &shape = sparse_tensor.shape();
  py::tuple dims(shape.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape[i]);
  }
  return dims;
}

py::tuple CSRTensorPy::GetPyTupleShape(const CSRTensor &csr_tensor) { return GetSparseTensorShape(csr_tensor); }

py::object CSRTensorPy::GetIndptr(const CSRTensorPtr &csr_tensor) {
  return PackTensorToPyObject(csr_tensor->GetIndptr());
}

py::object CSRTensorPy::GetIndices(const CSRTensorPtr &csr_tensor) {
  return PackTensorToPyObject(csr_tensor->GetIndices());
}

py::object CSRTensorPy::GetValues(const CSRTensorPtr &csr_tensor) {
  return PackTensorToPyObject(csr_tensor->GetValues());
}

py::tuple COOTensorPy::GetPyTupleShape(const COOTensor &coo_tensor) { return GetSparseTensorShape(coo_tensor); }

py::object COOTensorPy::GetIndices(const COOTensorPtr &coo_tensor) {
  return PackTensorToPyObject(coo_tensor->GetIndices());
}

py::object COOTensorPy::GetValues(const COOTensorPtr &coo_tensor) {
  return PackTensorToPyObject(coo_tensor->GetValues());
}

py::tuple RowTensorPy::GetPyTupleShape(const RowTensor &row_tensor) { return GetSparseTensorShape(row_tensor); }

py::object RowTensorPy::GetIndices(const RowTensorPtr &row_tensor) {
  return PackTensorToPyObject(row_tensor->GetIndices());
}

py::object RowTensorPy::GetValues(const RowTensorPtr &row_tensor) {
  return PackTensorToPyObject(row_tensor->GetValues());
}
}  // namespace tensor
}  // namespace mindspore
