/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_HELPER_INFERR_HELPERS_H_
#define MINDSPORE_LITE_SRC_COMMON_HELPER_INFERR_HELPERS_H_

#include "src/common/helper/external_tensor/helper.h"

namespace mindspore::infer::helper {
class InferHelpers {
 public:
  InferHelpers() = default;
  explicit InferHelpers(ExternalTensorHelper *external_tensor_helper)
      : external_tensor_helper_(external_tensor_helper) {}
  virtual ~InferHelpers() {
    if (external_tensor_helper_ != nullptr) {
      delete external_tensor_helper_;
      external_tensor_helper_ = nullptr;
    }
  }

  ExternalTensorHelper *GetExternalTensorHelper() { return external_tensor_helper_; }
  void SetExternalTensorHelper(ExternalTensorHelper *external_tensor_helper) {
    external_tensor_helper_ = external_tensor_helper;
  }

 private:
  ExternalTensorHelper *external_tensor_helper_ = nullptr;
};
}  // namespace mindspore::infer::helper

#endif  // MINDSPORE_LITE_SRC_COMMON_HELPER_INFERR_HELPERS_H_
