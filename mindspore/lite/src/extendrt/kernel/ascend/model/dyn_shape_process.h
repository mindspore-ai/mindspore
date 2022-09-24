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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_DYN_SHAPE_PROCESS_H
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_DYN_SHAPE_PROCESS_H

#include <vector>
#include <memory>
#include <string>
#include "extendrt/kernel/ascend/options/acl_model_options.h"
#include "kernel/kernel.h"
#include "include/api/types.h"

namespace mindspore::kernel {
namespace acl {
class DynShapeProcess {
 public:
  explicit DynShapeProcess(const AclModelOptionsPtr &options, size_t input_data_idx)
      : acl_options_(options), input_data_idx_(input_data_idx), batch_size_ptr_(nullptr), image_size_ptr_(nullptr) {}

  int ProcDynamicInput(std::vector<KernelTensorPtr> *const original_datas, std::vector<KernelTensorPtr> *const inputs);
  void DestroyDynamicInput(std::vector<KernelTensorPtr> *const inputs);

 private:
  int CheckBatchSize(std::vector<KernelTensorPtr> *const original_datas, std::vector<KernelTensorPtr> *const inputs);
  int CheckImageSize(std::vector<KernelTensorPtr> *const original_datas, std::vector<KernelTensorPtr> *const inputs);
  int AddBatchSizeInput(std::vector<KernelTensorPtr> *const original_datas, std::vector<KernelTensorPtr> *const inputs);
  int AddImageSizeInput(std::vector<KernelTensorPtr> *const original_datas, std::vector<KernelTensorPtr> *const inputs);
  int GetRealBatchSize(std::vector<KernelTensorPtr> *const inputs, int32_t *batch_size);
  int GetRealImageSize(std::vector<KernelTensorPtr> *const inputs, int32_t *image_size, int32_t num);

  AclModelOptionsPtr acl_options_;
  size_t input_data_idx_;
  AddressPtr batch_size_ptr_;
  AddressPtr image_size_ptr_;
};

using DynShapeProcPtr = std::shared_ptr<DynShapeProcess>;
}  // namespace acl
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_DYN_SHAPE_PROCESS_H
