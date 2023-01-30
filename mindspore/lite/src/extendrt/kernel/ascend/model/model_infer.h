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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_INFER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_INFER_H_

#include <vector>
#include <memory>
#include <set>
#include <utility>
#include <string>
#include "extendrt/kernel/ascend/model/model_process.h"
#include "extendrt/kernel/ascend/model/acl_env_guard.h"
#include "extendrt/kernel/ascend/options/acl_model_options.h"
#include "include/api/types.h"
#include "include/errorcode.h"

namespace mindspore::kernel {
namespace acl {
using mindspore::lite::STATUS;

class ModelInfer {
 public:
  explicit ModelInfer(const AclModelOptionsPtr &options);
  ~ModelInfer() = default;

  bool Init();
  bool Finalize();
  bool Load(const void *om_data, size_t om_data_size);
  bool Inference(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs);
  std::vector<Format> GetInputFormat();
  const std::vector<ShapeVector> GetOutputShape();
  const std::vector<ShapeVector> GetInputShape();
  const std::vector<TypeId> GetInputDataType();
  const std::vector<TypeId> GetOutputDataType();
  std::vector<Format> GetOutputFormat();

  bool Resize(const std::vector<ShapeVector> &new_shapes);

 private:
  bool init_flag_;
  std::string device_type_;
  aclrtContext context_;
  AclModelOptionsPtr options_;
  ModelProcess model_process_;
  std::shared_ptr<AclEnvGuard> acl_env_;
};

using ModelInferPtr = std::shared_ptr<ModelInfer>;
}  // namespace acl
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_INFER_H_
