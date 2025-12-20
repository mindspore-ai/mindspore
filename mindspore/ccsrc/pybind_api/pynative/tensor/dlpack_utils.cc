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

#include "pybind_api/pynative/tensor/dlpack_utils.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pybind_api/pynative/tensor/device_type_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "device_address/device_type.h"
#include "ir/tensor_storage_info.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/backend/common/device_address_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace tensor {
namespace {
constexpr int kBits8 = 8;
constexpr int kBits16 = 16;
constexpr int kBits32 = 32;
constexpr int kBits64 = 64;
constexpr int kLanes1 = 1;
constexpr int kSizes2 = 2;

bool IsContiguous(const ShapeVector &shape, const std::vector<int64_t> &strides) {
  if (shape.size() == 0) {
    return true;
  }
  if (shape.size() != strides.size()) {
    MS_LOG(EXCEPTION) << "shape.size() != strides.size()";
  }

  int64_t z = 1;
  for (int64_t i = SizeToLong(shape.size() - 1); i >= 0; --i) {
    const auto &shape_i = shape[i];
    if (shape_i != 1) {
      if (strides[i] == z) {
        z *= shape_i;
      } else {
        return false;
      }
    }
  }
  return true;
}

std::pair<ShapeVector, std::vector<int64_t>> GetOriShapeAndStrides(const ShapeVector &shape,
                                                                   const std::vector<int64_t> &strides) {
  MS_EXCEPTION_IF_CHECK_FAIL(shape.size() == strides.size(), "shape size should be equal to strides size");
  ShapeVector ori_shape;
  int64_t max_loc = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    auto strided_size = strides[i] * (shape[i] - 1);
    max_loc += strided_size;
  }
  ori_shape.push_back(max_loc);
  std::vector<int64_t> ori_strides;
  ori_strides.push_back(1);
  return std::make_pair(ori_shape, ori_strides);
}
}  // namespace
DLDataType DLPackUtils::GetDLDataType(const TypeId &type_id) {
  DLDataType dtype;
  switch (type_id) {
    case kNumberTypeFloat32:
      dtype.code = kDLFloat;
      dtype.bits = kBits32;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeFloat64:
      dtype.code = kDLFloat;
      dtype.bits = kBits64;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeFloat16:
      dtype.code = kDLFloat;
      dtype.bits = kBits16;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeBFloat16:
      dtype.code = kDLBfloat;
      dtype.bits = kBits16;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeInt8:
      dtype.code = kDLInt;
      dtype.bits = kBits8;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeInt16:
      dtype.code = kDLInt;
      dtype.bits = kBits16;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeInt32:
      dtype.code = kDLInt;
      dtype.bits = kBits32;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeInt64:
      dtype.code = kDLInt;
      dtype.bits = kBits64;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeUInt8:
      dtype.code = kDLUInt;
      dtype.bits = kBits8;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeUInt16:
      dtype.code = kDLUInt;
      dtype.bits = kBits16;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeUInt32:
      dtype.code = kDLUInt;
      dtype.bits = kBits32;
      dtype.lanes = kLanes1;
      break;
    case kNumberTypeUInt64:
      dtype.code = kDLUInt;
      dtype.bits = kBits64;
      dtype.lanes = kLanes1;
      break;
    default:
      MS_LOG(EXCEPTION) << "Unsupported data type: " << type_id;
  }
  return dtype;
}

DLDevice DLPackUtils::GetDLDevice(size_t device_id, device::DeviceType device_type) {
  DLDevice ctx;
  if (device_type == device::DeviceType::kCPU) {
    ctx.device_id = static_cast<int32_t>(0);
  } else if (device_type == device::DeviceType::kAscend) {
    ctx.device_id = static_cast<int32_t>(device_id);
  }

  ctx.device_type = DeviceTypeUtils::MsDeviceTargetToDLDeviceType(device_type);
  return ctx;
}

TypeId DLPackUtils::GetTypeId(const DLDataType &dtype) {
  TypeId type_id = kTypeUnknown;
  if (dtype.code == kDLFloat) {
    if (dtype.bits == kBits32) {
      type_id = kNumberTypeFloat32;
    } else if (dtype.bits == kBits64) {
      type_id = kNumberTypeFloat64;
    } else if (dtype.bits == kBits16) {
      type_id = kNumberTypeFloat16;
    } else {
      MS_LOG(EXCEPTION) << "Unsupported float bits: " << dtype.bits;
    }
  } else if (dtype.code == kDLBfloat) {
    type_id = kNumberTypeBFloat16;
  } else if (dtype.code == kDLInt) {
    if (dtype.bits == kBits8) {
      type_id = kNumberTypeInt8;
    } else if (dtype.bits == kBits16) {
      type_id = kNumberTypeInt16;
    } else if (dtype.bits == kBits32) {
      type_id = kNumberTypeInt32;
    } else if (dtype.bits == kBits64) {
      type_id = kNumberTypeInt64;
    } else {
      MS_LOG(EXCEPTION) << "Unsupported int bits: " << dtype.bits;
    }
  } else if (dtype.code == kDLUInt) {
    if (dtype.bits == kBits8) {
      type_id = kNumberTypeUInt8;
    } else if (dtype.bits == kBits16) {
      type_id = kNumberTypeUInt16;
    } else if (dtype.bits == kBits32) {
      type_id = kNumberTypeUInt32;
    } else if (dtype.bits == kBits64) {
      type_id = kNumberTypeUInt64;
    } else {
      MS_LOG(EXCEPTION) << "Unsupported uint bits: " << dtype.bits;
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported data type code: " << dtype.code;
  }
  return type_id;
}

TensorPtr DLPackUtils::FromDLPack(DLManagedTensor *dlpack) {
  if (dlpack == nullptr) {
    MS_LOG(EXCEPTION) << "Input dlpack is nullptr";
  }
  // This tensor is treated as a leaf node in the computation graph.
  // As a result, gradient propagation (backpropagation) to other frameworks is not supported.
  auto type_id = GetTypeId(dlpack->dl_tensor.dtype);
  auto shape = dlpack->dl_tensor.shape;
  size_t ndim = static_cast<size_t>(dlpack->dl_tensor.ndim);
  ShapeVector shape_vec;
  for (size_t i = 0; i < ndim; i++) {
    shape_vec.push_back(shape[i]);
  }
  auto strides = dlpack->dl_tensor.strides;
  std::vector<int64_t> strides_vec;
  for (size_t i = 0; i < ndim; i++) {
    strides_vec.push_back(strides[i]);
  }
  auto offset = dlpack->dl_tensor.byte_offset;
  if (offset != 0) {
    MS_LOG(EXCEPTION) << "Unsupported dlpack byte offset: " << offset;
  }
  auto [ori_shape, ori_strides] = GetOriShapeAndStrides(shape_vec, strides_vec);
  auto storage_info = std::make_shared<TensorStorageInfo>(shape_vec, strides_vec, offset, ori_shape, ori_strides,
                                                          IsContiguous(shape_vec, strides_vec));
  auto tensor = std::make_shared<Tensor>(type_id, shape_vec);
  tensor->set_need_pipeline_sync(true);
  tensor->set_storage_info(storage_info);

  auto dldevice = dlpack->dl_tensor.device;
  auto device_type = dldevice.device_type;

  // Get current device type from MindSpore context
  auto ms_context = MsContext::GetInstance();
  auto ms_device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey host_key = {DeviceTypeUtils::DLDeviceTypeToMsDeviceTarget(device_type), ms_device_id};
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  auto stream_id = device_context->device_res_manager_->GetCurrentStreamId();
  auto address_size = GetTypeByte(TypeIdToType(type_id)) * SizeOf(ori_shape);
  auto device_address =
    std::make_shared<device::DeviceAddress>(nullptr, address_size, device_context->GetDeviceType(), stream_id);

  device_address->set_tensor_storage_info(storage_info);
  tensor->set_device_address(device_address);
  tensor->set_contiguous_callback([device_context](const TensorPtr &self) -> DeviceAddressPtr {
    MS_EXCEPTION_IF_NULL(self);
    // as_numpy sync promise contiguous run_sync
    return runtime::DeviceAddressUtils::ConvertContiguousDeviceAddress(device_context, self);
  });

  // set data
  auto data = dlpack->dl_tensor.data;
  device_address->set_ptr(data);
  device_address->set_from_mem_pool(false);

  // update deleter
  auto ref_cnt = device_address->device_pointer();
  ref_cnt->set_deleter([dlpack = dlpack](void *, bool) {
    if (dlpack == nullptr) {
      return;
    }
    if (dlpack->deleter) {
      dlpack->deleter(dlpack);
    }
  });
  return tensor;
}

namespace {
struct DLMTensor {
  TensorPtr handle;
  DLManagedTensor tensor{};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};
}  // namespace

static void deleter(DLManagedTensor *arg) { delete static_cast<DLMTensor *>(arg->manager_ctx); }

DLManagedTensor *DLPackUtils::ToDLPack(const TensorPtr &src) {
  MS_EXCEPTION_IF_NULL(src);
  DLMTensor *dlm_tensor = new DLMTensor();
  dlm_tensor->shape = src->shape();
  dlm_tensor->strides = src->stride();
  // normalized strides
  for (size_t i = 0; i < dlm_tensor->shape.size(); i++) {
    if (dlm_tensor->shape[i] < kSizes2) {
      dlm_tensor->strides[i] = 1;
    }
  }
  runtime::Pipeline::Get().WaitFrontend();
  const auto device_type = src->device_address()->GetDeviceType();
  kernel::pyboost::OpRunStatus::Get().set_run_info(kernel::pyboost::OpStatus(true, device_type));
  auto view = kernel::pyboost::as_strided(src, dlm_tensor->shape, dlm_tensor->strides, src->storage_offset());
  runtime::Pipeline::Get().WaitForward();
  dlm_tensor->tensor.manager_ctx = dlm_tensor;
  dlm_tensor->tensor.deleter = &deleter;
  auto view_address = std::dynamic_pointer_cast<device::DeviceAddress>(view->device_address());
  if (view_address == nullptr) {
    MS_LOG(EXCEPTION) << "Device address is nullptr";
  }
  dlm_tensor->handle = view;
  const auto &ms_device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  // Set DLPack device via unified conversion
  dlm_tensor->tensor.dl_tensor.device = GetDLDevice(ms_device_id, device_type);
  dlm_tensor->tensor.dl_tensor.ndim = static_cast<int32_t>(dlm_tensor->shape.size());
  dlm_tensor->tensor.dl_tensor.dtype = GetDLDataType(src->data_type());
  dlm_tensor->tensor.dl_tensor.shape = view->storage_info()->shape.data();
  dlm_tensor->tensor.dl_tensor.strides = view->storage_info()->strides.data();
  dlm_tensor->tensor.dl_tensor.byte_offset = 0;
  auto offset = mindspore::abstract::TypeIdSize(view->data_type()) * static_cast<size_t>(view->storage_offset());
  dlm_tensor->tensor.dl_tensor.data = static_cast<char *>(const_cast<void *>(view_address->GetPtr())) + offset;
  if (dlm_tensor->tensor.dl_tensor.data == nullptr) {
    MS_LOG(EXCEPTION) << "Data is nullptr";
  }
  return &(dlm_tensor->tensor);
}
}  // namespace tensor
}  // namespace mindspore
