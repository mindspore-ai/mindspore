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

#include "src/runtime/kernel/arm/base/carry_data.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int CarryDataKernel::MoveData(const std::vector<lite::Tensor *>::iterator &dst_begin,
                              const std::vector<lite::Tensor *>::iterator &dst_end,
                              const std::vector<lite::Tensor *>::iterator &src_begin,
                              const std::vector<lite::Tensor *>::iterator &src_limit) {
  for (auto dst_iter = dst_begin, src_iter = src_begin; dst_iter != dst_end; dst_iter++, src_iter++) {
    if (src_iter == src_limit) {
      MS_LOG(ERROR) << "out of range of input tensor";
      return RET_ERROR;
    }
    auto *dst_tensor = *dst_iter;
    auto *src_tensor = *src_iter;
    if (dst_tensor == nullptr || src_tensor == nullptr) {
      MS_LOG(ERROR) << "input tensor or output tensor of merge is nullptr";
      return RET_ERROR;
    }
    lite::STATUS ret = RET_OK;
    if (src_tensor->IsConst() || src_tensor->IsGraphInput()) {
      MS_LOG(DEBUG) << "Carry const data and graph inputs.";
      dst_tensor->set_data(src_tensor->data());
      dst_tensor->set_own_data(false);
    } else {
      if (src_tensor->data_type() == kObjectTypeTensorType && dst_tensor->data_type() == kObjectTypeTensorType) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
        MS_LOG(DEBUG) << "Carry MoveTensorListData";
        ret = MoveTensorListData(reinterpret_cast<lite::TensorList *>(dst_tensor),
                                 reinterpret_cast<lite::TensorList *>(src_tensor));
#else
        MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
        return RET_NOT_SUPPORT;
#endif
      } else {
        MS_LOG(DEBUG) << "Carry MoveTensorData";
        ret = MoveTensorData(dst_tensor, src_tensor);
      }
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Move data failed : " << ret;
      return ret;
    }
  }
  return RET_OK;
}

int CarryDataKernel::MoveTensorData(lite::Tensor *dst_tensor, lite::Tensor *src_tensor) {
  if (dst_tensor->data_type() != src_tensor->data_type() || dst_tensor->format() != src_tensor->format() ||
      !(dst_tensor->shape() == src_tensor->shape() || (dst_tensor->shape().empty() && src_tensor->shape().empty()))) {
    MS_LOG(ERROR) << "input tensor and output tensor is incompatible.";
    MS_LOG(ERROR) << "input tensor data_type: " << src_tensor->data_type() << " vs "
                  << "output tensor data_type: " << dst_tensor->data_type()
                  << "input tensor format: " << src_tensor->format() << " vs "
                  << "output tensor format: " << dst_tensor->format() << " input tensor shape: " << src_tensor->shape()
                  << " vs "
                  << "output tensor shape: " << dst_tensor->shape();
    return RET_ERROR;
  }
  if (src_tensor->allocator() == nullptr) {
    MS_LOG(ERROR) << "src_tensor allocator is nullptr.";
    return RET_ERROR;
  }

  // need replace with increase data ref count
  memcpy(dst_tensor->data(), src_tensor->data(), src_tensor->Size());
  return RET_OK;
}
#ifndef CONTROLFLOW_TENSORLIST_CLIP
int CarryDataKernel::MoveTensorListData(lite::TensorList *dst_tensorlist, lite::TensorList *src_tensorlist) {
  // shape may change, because tensors.size() can be change in RunGraph
  if (dst_tensorlist->data_type() != src_tensorlist->data_type() ||
      dst_tensorlist->format() != src_tensorlist->format()) {
    MS_LOG(ERROR) << "input tensorlist and output tensorlist data_type or format is incompatible";
    MS_LOG(ERROR) << "input tensor data_type: " << src_tensorlist->data_type() << " vs "
                  << "output tensor data_type: " << dst_tensorlist->data_type()
                  << "input tensor format: " << src_tensorlist->format() << " vs "
                  << "output tensor format: " << dst_tensorlist->format();
    return RET_ERROR;
  }
  // when tensorlist malloc is done. this need to check element_shape compatibility
  dst_tensorlist->set_element_shape(src_tensorlist->element_shape());

  auto update_data_type = kTypeUnknown;
  auto dst_tensor_data_type = dst_tensorlist->tensors_data_type();
  auto src_tensor_data_type = src_tensorlist->tensors_data_type();
  if (dst_tensor_data_type != src_tensor_data_type) {
    if (src_tensor_data_type != kTypeUnknown && dst_tensor_data_type != kTypeUnknown) {
      MS_LOG(ERROR) << "input tensorlist and output tensorlist is incompatible";
      return RET_ERROR;
    }
    update_data_type = dst_tensor_data_type != kTypeUnknown ? dst_tensor_data_type : src_tensor_data_type;
  }
  if (update_data_type != kTypeUnknown) {
    src_tensorlist->set_tensors_data_type(update_data_type);
    dst_tensorlist->set_tensors_data_type(update_data_type);
  }
  size_t src_tensorlist_tensors_size = src_tensorlist->tensors().size();
  for (size_t i = 0; i < src_tensorlist_tensors_size; ++i) {
    auto &src_tensor = src_tensorlist->tensors()[i];
    auto &dst_tensor = dst_tensorlist->tensors()[i];

    if (src_tensor->allocator() != nullptr) {
      src_tensor->allocator()->IncRefCount(src_tensor->data(), dst_tensor->ref_count());
    }
    dst_tensor->set_own_data(src_tensor->own_data());
    if (src_tensor->data() != nullptr) {
      dst_tensor->set_data(src_tensor->data());
    }
    dst_tensor->set_shape(src_tensor->shape());
  }
  return RET_OK;
}
#endif
}  // namespace mindspore::kernel
