/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_CXX_API_ACL_VM_MS_TENSOR_REF_H
#define MINDSPORE_CCSRC_CXX_API_ACL_VM_MS_TENSOR_REF_H

#include <memory>
#include <string>
#include <vector>
#include "include/api/types.h"
#include "mindspore/core/base/base_ref.h"

namespace mindspore {
class MSTensorRef : public BaseRef {
 public:
  MS_DECLARE_PARENT(MSTensorRef, BaseRef);

  static VectorRef Convert(const std::vector<MSTensor> &tensors);
  static std::vector<MSTensor> Convert(const BaseRef &args);

  explicit MSTensorRef(const MSTensor &tensor) : ms_tensor_(tensor) {}
  ~MSTensorRef() override = default;

  const MSTensor &GetTensor() const { return ms_tensor_; }
  std::shared_ptr<Base> copy() const override;

  uint32_t type() const override { return tid(); }
  std::string ToString() const override { return ms_tensor_.Name(); }
  bool operator==(const BaseRef &other) const override;

 private:
  static std::vector<MSTensor> ConvertTuple(const VectorRef &args);

  MSTensor ms_tensor_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_ACL_VM_MS_TENSOR_REF_H
