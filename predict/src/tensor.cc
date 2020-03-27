/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "include/tensor.h"
#include "common/mslog.h"
#include "src/op_common.h"
#include "include/errorcode.h"
#include "securec/include/securec.h"
#include "common/common.h"
#include "src/runtime/allocator.h"

namespace mindspore {
namespace predict {
Tensor *Tensor::CopyFromTensorDef(const TensorDef &tensorDef) {
  std::vector<int64_t> dims;

  if (tensorDef.dims() == nullptr) {
    MS_LOGD("tensorDef->dims is nullptr");
  } else {
    MS_ASSERT(tensorDef.dims()->data() != nullptr);
    for (uint32_t j = 0; j < tensorDef.dims()->size(); j++) {
      dims.push_back(tensorDef.dims()->data()[j]);
    }
  }
  auto tensor =
    std::unique_ptr<Tensor>(new (std::nothrow) Tensor(tensorDef.dataType(), dims, tensorDef.format(), nullptr));
  if (tensor == nullptr) {
    MS_LOGE("new Tensor failed");
    return nullptr;
  }

  if (tensorDef.refCount() == MSConst_WEIGHT_REFCOUNT && tensorDef.data() != nullptr && tensorDef.data()->size() > 0) {
    if (dims.size() < 1) {
      tensor->SetDims({1});
    }
    auto ret = tensor->MallocData();
    if (ret != RET_OK) {
      MS_LOGE("malloc data fail,datasize %zu", tensor->GetDataSize());
      return nullptr;
    }
    auto tensorData = tensorDef.data()->data();
    ret = memcpy_sp(tensor->GetData(), tensor->GetDataSize(), tensorData, tensorDef.data()->size());
    if (ret != RET_OK) {
      MS_LOGE("copy data fail,dst size %zu, src size %u", tensor->GetDataSize(), tensorDef.data()->size());
      return nullptr;
    }
  }
  tensor->refCount = tensorDef.refCount();
  return tensor.release();
}

Tensor::Tensor(const Tensor &tensor, bool copyData) {
  format = tensor.format;
  dlTensor.data = nullptr;
  dlTensor.ctx.device_type = tensor.dlTensor.ctx.device_type;
  dlTensor.ctx.device_id = tensor.dlTensor.ctx.device_id;
  dlTensor.strides = nullptr;
  dlTensor.byte_offset = tensor.dlTensor.byte_offset;
  dlTensor.dtype.code = tensor.dlTensor.dtype.code;
  dlTensor.dtype.bits = tensor.dlTensor.dtype.bits;
  dlTensor.dtype.lanes = tensor.dlTensor.dtype.lanes;

  dlTensor.ndim = tensor.dlTensor.ndim;
  if (dlTensor.ndim > 0) {
    dlTensor.shape = new (std::nothrow) int64_t[dlTensor.ndim];
    if (dlTensor.shape != nullptr) {
      for (int i = 0; i < dlTensor.ndim; i++) {
        dlTensor.shape[i] = tensor.dlTensor.shape[i];
      }
    } else {
      MS_LOGW("new shape fail,ndim %d", dlTensor.ndim);
    }
  } else {
    dlTensor.shape = nullptr;
  }
  if (copyData) {
    allocator = tensor.allocator;
    refCount = tensor.refCount;
    auto ret = MallocData();
    if (ret != RET_OK) {
      return;
    }
    size_t datasize = GetDataSize();
    ret = memcpy_sp(dlTensor.data, datasize, tensor.dlTensor.data, datasize);
    if (ret != RET_OK) {
      return;
    }
  }
}

Tensor::Tensor(DataType dt, const std::vector<int64_t> &dims, Format format, void *data) {
  this->format = format;
  dlTensor.data = data;
  dlTensor.ctx.device_type = DLDeviceType::kDLCPU;
  dlTensor.ctx.device_id = 0;
  dlTensor.strides = nullptr;
  dlTensor.byte_offset = 0;

  dlTensor.ndim = static_cast<int>(dims.size());
  if (dlTensor.ndim > 0) {
    dlTensor.shape = new (std::nothrow) int64_t[dlTensor.ndim];
    if (dlTensor.shape != nullptr) {
      for (int i = 0; i < dlTensor.ndim; i++) {
        dlTensor.shape[i] = dims[i];
      }
    } else {
      MS_LOGW("new shape fail,ndim %d", dlTensor.ndim);
    }
  } else {
    dlTensor.shape = nullptr;
  }

  SetDataType(dt);
}

Tensor::~Tensor() { FreeTensor(); }

DLDataType Tensor::GetTensorDtype() const { return dlTensor.dtype; }

void *Tensor::GetData() const { return dlTensor.data; }

void Tensor::SetData(void *data) { dlTensor.data = data; }

DataType Tensor::GetDataType() const {
  DataType dataType = DataType_DT_UNDEFINED;
  switch (dlTensor.dtype.code) {
    case kDLFloat:
      if (dlTensor.dtype.bits == 32) {
        dataType = DataType_DT_FLOAT;
      } else if (dlTensor.dtype.bits == 16) {
        dataType = DataType_DT_FLOAT16;
      }
      break;
    case kDLInt:
      if (dlTensor.dtype.bits == 32) {
        dataType = DataType_DT_INT32;
      } else if (dlTensor.dtype.bits == 8) {
        dataType = DataType_DT_INT8;
      }
      break;
    case kDLUInt:
      if (dlTensor.dtype.bits == 32) {
        dataType = DataType_DT_UINT32;
      } else if (dlTensor.dtype.bits == 8) {
        dataType = DataType_DT_UINT8;
      }
      break;
    default:
      break;
  }
  return dataType;
}

void Tensor::SetDataType(DataType dt) {
  switch (dt) {
    case DataType_DT_FLOAT:
      dlTensor.dtype.code = kDLFloat;
      dlTensor.dtype.bits = 32;
      dlTensor.dtype.lanes = 1;
      break;
    case DataType_DT_FLOAT16:
      dlTensor.dtype.code = kDLFloat;
      dlTensor.dtype.bits = 16;
      dlTensor.dtype.lanes = 1;
      break;
    case DataType_DT_INT8:
      dlTensor.dtype.code = kDLInt;
      dlTensor.dtype.bits = 8;
      dlTensor.dtype.lanes = 1;
      break;
    case DataType_DT_UINT8:
      dlTensor.dtype.code = kDLUInt;
      dlTensor.dtype.bits = 8;
      dlTensor.dtype.lanes = 1;
      break;
    case DataType_DT_INT32:
      dlTensor.dtype.code = kDLInt;
      dlTensor.dtype.bits = 32;
      dlTensor.dtype.lanes = 1;
      break;
    case DataType_DT_UINT32:
      dlTensor.dtype.code = kDLUInt;
      dlTensor.dtype.bits = 32;
      dlTensor.dtype.lanes = 1;
      break;
    default:
      MS_LOGW(" DataType %d is not implemented.", dt);
      MS_LOGW(" DataType DT_FLOAT is used.");
      dlTensor.dtype.code = kDLFloat;
      dlTensor.dtype.bits = 32;
      dlTensor.dtype.lanes = 1;
      return;
  }
}

int Tensor::GetNDim() const { return dlTensor.ndim; }

std::vector<int64_t> Tensor::GetDims() const {
  std::vector<int64_t> dims;
  for (int i = 0; i < dlTensor.ndim; i++) {
    dims.push_back(dlTensor.shape[i]);
  }
  return dims;
}

size_t Tensor::GetElementSize() const {
  const int tile = 4;
  if (format == Format_NC4HW4) {
    size_t size = 1;
    for (int i = 0; i < dlTensor.ndim; i++) {
      auto var = static_cast<size_t>(dlTensor.shape[i]);
      if (i == 1) {
        var = UP_DIV(var, tile) * tile;
      }
      size *= var;
    }
    return size;
  } else {
    size_t size = 1;
    for (int i = 0; i < dlTensor.ndim; i++) {
      size *= static_cast<size_t>(dlTensor.shape[i]);
    }

    return size;
  }
}

size_t Tensor::GetDataSize() const {
  size_t size = GetElementSize();

  const int BYTES = 8;
  const int GAP = 7;
  size *= (dlTensor.dtype.bits * dlTensor.dtype.lanes + GAP) / BYTES;
  return size;
}

int Tensor::MallocData(std::shared_ptr<Allocator> allocator, int refCount) {
  if (dlTensor.data != nullptr) {
    this->refCount += refCount;
    return RET_OK;
  }
  this->refCount = refCount;

  size_t size = GetDataSize();
  if (allocator) {
    this->allocator = allocator;
    dlTensor.data = allocator->Malloc(size);
  } else {
    if (size > MAX_MALLOC_SIZE) {
      return RET_ERROR;
    }
    dlTensor.data = malloc(size);
  }
  if (dlTensor.data == nullptr) {
    return RET_ERROR;
  }
  return RET_OK;
}

void Tensor::ForceFreeData() {
  if (allocator) {
    allocator->Free(dlTensor.data);
  } else {
    free(dlTensor.data);
  }
  dlTensor.data = nullptr;
}

void Tensor::FreeData() {
  --refCount;
  if (refCount <= 0) {
    ForceFreeData();
  }
}

bool Tensor::CompareShape(const Tensor &dst) {
  if (dlTensor.ndim != dst.dlTensor.ndim || dlTensor.shape == nullptr || dst.dlTensor.shape == nullptr) {
    MS_LOGE("param error, one.ndim: %d, other.ndim: %d, one shape %p,other shape %p", dlTensor.ndim, dst.dlTensor.ndim,
            dlTensor.shape, dst.dlTensor.shape);
    return false;
  }

  for (int i = 0; i < dlTensor.ndim; i++) {
    if (dlTensor.shape[i] != dst.dlTensor.shape[i]) {
      MS_LOGE("one.shape[%d]: %ld, other.shape[%d]: %ld", i, dlTensor.shape[i], i, dst.dlTensor.shape[i]);
      return false;
    }
  }
  return true;
}

bool Tensor::CompareShape(const std::vector<int64_t> &other) {
  if (dlTensor.ndim != other.size() || dlTensor.shape == nullptr) {
    return false;
  }

  for (int i = 0; i < dlTensor.ndim; i++) {
    if (dlTensor.shape[i] != other[i]) {
      return false;
    }
  }
  return true;
}

int64_t Tensor::Height() const {
  if (dlTensor.shape == nullptr) {
    MS_LOGE("shape is null");
  }
  if (dlTensor.ndim != DIM_DEFAULT_SIZE) {
    MS_LOGE("Tensor should be 4 dimensional.");
    return -1;
  }
  switch (this->format) {
    case Format_NCHW:
    case Format_NC4HW4:
      return dlTensor.shape[NCHW_H];
    case Format_NHWC:
      return dlTensor.shape[NHWC_H];
    default:
      MS_LOGE("Unsupported format: %d", this->format);
      return -1;
  }
}

int64_t Tensor::Width() const {
  if (dlTensor.shape == nullptr) {
    MS_LOGE("shape is null");
  }
  if (dlTensor.ndim != DIM_DEFAULT_SIZE) {
    MS_LOGE("Tensor should be 4 dimensional.");
    return -1;
  }
  switch (this->format) {
    case Format_NCHW:
    case Format_NC4HW4:
      return dlTensor.shape[NCHW_W];
    case Format_NHWC:
      return dlTensor.shape[NHWC_W];
    default:
      MS_LOGE("Unsupported format: %d", this->format);
      return -1;
  }
}

int64_t Tensor::Channel() const {
  if (dlTensor.shape == nullptr) {
    MS_LOGE("shape is null");
  }
  if (dlTensor.ndim != DIM_DEFAULT_SIZE) {
    MS_LOGE("Tensor should be 4 dimensional.");
    return -1;
  }
  switch (this->format) {
    case Format_NCHW:
    case Format_NC4HW4:
      return dlTensor.shape[NCHW_C];
    case Format_NHWC:
      return dlTensor.shape[NHWC_C];
    default:
      MS_LOGE("Unsupported format: %d", this->format);
      return -1;
  }
}

int64_t Tensor::Batch() const {
  if (dlTensor.shape == nullptr) {
    MS_LOGE("shape is null");
  }
  if (dlTensor.ndim != DIM_DEFAULT_SIZE) {
    MS_LOGE("Tensor should be 4 dimensional.");
    return -1;
  }
  switch (this->format) {
    case Format_NCHW:
    case Format_NC4HW4:
    case Format_NHWC:
      return dlTensor.shape[NCHW_N];
    default:
      MS_LOGE("Unsupported format: %d", this->format);
      return -1;
  }
}

int64_t Tensor::Stride(int index) const {
  if (dlTensor.strides) {
    return dlTensor.strides[index];
  }
  if (dlTensor.shape == nullptr) {
    MS_LOGE("shape is null");
    return -1;
  }
  int64_t stride = 1;
  for (int i = index + 1; i < dlTensor.ndim; i++) {
    stride *= dlTensor.shape[i];
  }
  return stride;
}

void Tensor::SetStride() {
  if (dlTensor.strides == nullptr) {
    if (dlTensor.ndim < 1) {
      MS_LOGE("dims of dlTensor is empty.");
      return;
    }
    dlTensor.strides = new (std::nothrow) int64_t[dlTensor.ndim - 1];
    if (dlTensor.strides == nullptr) {
      MS_LOGW("new stride fail, ndim %d.", dlTensor.ndim);
      return;
    }
  }

  for (int idx = 0; idx < dlTensor.ndim - 1; idx++) {
    int64_t stride = 1;
    if (dlTensor.ndim <= idx + 1) {
      MS_LOGE("out of for loop upper limit.");
      return;
    }
    for (int i = idx + 1; i < dlTensor.ndim; i++) {
      stride *= dlTensor.shape[i];
    }
    dlTensor.strides[idx] = stride;
  }
}
void Tensor::SetScale(bool isScale) { this->isScale = isScale; }

void Tensor::SetStride(int index, int64_t stride) {
  if (index >= dlTensor.ndim) {
    return;
  }

  if (dlTensor.strides == nullptr) {
    SetStride();
  }

  dlTensor.strides[index] = stride;
  return;
}

void Tensor::SetDims(const std::vector<int64_t> &dims) {
  if (dlTensor.shape != nullptr) {
    delete[] dlTensor.shape;
  }
  dlTensor.ndim = static_cast<int>(dims.size());
  if (dlTensor.ndim > 0) {
    dlTensor.shape = new (std::nothrow) int64_t[dlTensor.ndim];
    if (dlTensor.shape != nullptr) {
      for (int i = 0; i < dlTensor.ndim; i++) {
        dlTensor.shape[i] = dims[i];
      }
    } else {
      MS_LOGW("new shape fail,ndim %d", dlTensor.ndim);
    }
  } else {
    dlTensor.shape = nullptr;
  }
}

void Tensor::FreeTensor() {
  if (dlTensor.shape != nullptr) {
    delete[] dlTensor.shape;
    dlTensor.shape = nullptr;
  }

  if (dlTensor.strides != nullptr) {
    delete[] dlTensor.strides;
    dlTensor.strides = nullptr;
  }

  dlTensor.ndim = 0;

  if (allocator != nullptr) {
    allocator->Free(dlTensor.data);
  } else {
    free(dlTensor.data);
  }
  dlTensor.data = nullptr;
}

size_t Tensor::GetNC4HW4ElementSize(bool isNhwc) {
  int alignIndex = 1;
  if (isNhwc) {
    alignIndex = 3;
  }

  size_t size = 1;
  for (int i = 0; i < dlTensor.ndim; i++) {
    auto var = static_cast<size_t>(dlTensor.shape[i]);
    if (i == alignIndex) {
      var = ALIGN_UP4(var);
    }
    size *= var;
  }
  return size;
}

size_t Tensor::GetNC4HW4DataSize(bool isNhwc) {
  size_t size = GetNC4HW4ElementSize(isNhwc);
  const int BYTES = 8;
  const int GAP = 7;
  size *= (dlTensor.dtype.bits * dlTensor.dtype.lanes + GAP) / BYTES;
  return size;
}
}  // namespace predict
}  // namespace mindspore
