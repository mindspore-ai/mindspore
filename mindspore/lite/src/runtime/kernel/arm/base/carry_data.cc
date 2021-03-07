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
#include "src/tensorlist.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int CarryDataKernel::MoveData(std::vector<lite::Tensor *>::iterator dst_begin,
                              std::vector<lite::Tensor *>::iterator dst_end,
                              std::vector<lite::Tensor *>::iterator src_begin,
                              std::vector<lite::Tensor *>::iterator src_limit) {
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
    lite::STATUS ret;
    if (src_tensor->data_type() == kObjectTypeTensorType && dst_tensor->data_type() == kObjectTypeTensorType) {
      ret = MoveTensorListData(reinterpret_cast<lite::TensorList *>(dst_tensor),
                               reinterpret_cast<lite::TensorList *>(src_tensor));
    } else {
      ret = MoveTensorData(dst_tensor, src_tensor);
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
  if (src_tensor->root_tensor() == nullptr) {
    if (src_tensor->IsConst() || src_tensor->IsGraphInput() || src_tensor->ref_count() > 1) {
      auto dst_data = dst_tensor->MutableData();
      if (dst_data == nullptr) {
        MS_LOG(ERROR) << "data of dst tensor is nullptr";
        return RET_ERROR;
      }
      auto src_data = src_tensor->data_c();
      MS_ASSERT(src_data != nullptr);
      memcpy(dst_data, src_data, dst_tensor->Size());
    } else {
      dst_tensor->FreeData();
      dst_tensor->set_data(src_tensor->data_c());
      src_tensor->set_data(nullptr);
    }
  } else {
    auto ret = dst_tensor->set_root_tensor(src_tensor->root_tensor());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set root tensor for tensor(" << dst_tensor->tensor_name() << ") failed";
      return ret;
    }
  }
  return RET_OK;
}

int CarryDataKernel::MoveTensorListData(lite::TensorList *dst_tensor, lite::TensorList *src_tensor) {
  // shape may change, because tensors.size() can be change in RunGraph
  if (dst_tensor->data_type() != src_tensor->data_type() || dst_tensor->format() != src_tensor->format()) {
    MS_LOG(ERROR) << "input tensorlist and output tensorlist data_type or format is incompatible";
    MS_LOG(ERROR) << "input tensor data_type: " << src_tensor->data_type() << " vs "
                  << "output tensor data_type: " << dst_tensor->data_type()
                  << "input tensor format: " << src_tensor->format() << " vs "
                  << "output tensor format: " << dst_tensor->format();
    return RET_ERROR;
  }
  // when tensorlist malloc is done. this need to check element_shape compatibility
  dst_tensor->set_element_shape(src_tensor->element_shape());

  auto update_data_type = kTypeUnknown;
  auto dst_tensor_data_type = dst_tensor->tensors_data_type();
  auto src_tensor_data_type = src_tensor->tensors_data_type();
  if (dst_tensor_data_type != src_tensor_data_type) {
    if (src_tensor_data_type != kTypeUnknown && dst_tensor_data_type != kTypeUnknown) {
      MS_LOG(ERROR) << "input tensorlist and output tensorlist is incompatible";
      return RET_ERROR;
    }
    update_data_type = dst_tensor_data_type != kTypeUnknown ? dst_tensor_data_type : src_tensor_data_type;
  }
  if (update_data_type != kTypeUnknown) {
    src_tensor->set_tensors_data_type(update_data_type);
    dst_tensor->set_tensors_data_type(update_data_type);
  }
  if (src_tensor->root_tensor() == nullptr) {
    dst_tensor->CopyTensorList(*src_tensor, false);
    src_tensor->set_tensors({});
  } else {
    dst_tensor->set_shape(src_tensor->shape());
    auto ret = dst_tensor->set_root_tensor(src_tensor->root_tensor());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set root tensor for tensor(" << dst_tensor->tensor_name() << ") failed";
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
