/**
 * Copyright 2019 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/util/slice.h"

namespace mindspore {
namespace dataset {
WritableSlice::WritableSlice(const WritableSlice &src, off64_t offset, size_t len) : ReadableSlice(src, offset, len) {
  mutable_data_ = static_cast<char *>(src.mutable_data_) + offset;
}
WritableSlice::WritableSlice(const WritableSlice &src, off64_t offset)
    : WritableSlice(src, offset, src.GetSize() - offset) {}
Status WritableSlice::Copy(WritableSlice *dest, const ReadableSlice &src) {
  RETURN_UNEXPECTED_IF_NULL(dest);
  RETURN_UNEXPECTED_IF_NULL(dest->GetMutablePointer());
  if (dest->GetSize() <= 0) {
    RETURN_STATUS_UNEXPECTED("Destination length is non-positive");
  }
  auto err = memcpy_s(dest->GetMutablePointer(), dest->GetSize(), src.GetPointer(), src.GetSize());
  if (err) {
    RETURN_STATUS_UNEXPECTED(std::to_string(err));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
