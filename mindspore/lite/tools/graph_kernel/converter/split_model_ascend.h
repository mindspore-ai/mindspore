/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_MODEL_ASCEND_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_MODEL_ASCEND_H_

#include "backend/common/graph_kernel/split_model/split_model_factory.h"
namespace mindspore::graphkernel::inner {
class SplitModelAscend : public SplitModel {
 public:
  SplitModelAscend() = default;
  virtual ~SplitModelAscend() = default;

 protected:
  AreaMode GetDefaultAreaMode(const PrimOpPtr &) const override;
  void InitFusePatterns() override;
};
}  // namespace mindspore::graphkernel::inner
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_SPLIT_MODEL_ASCEND_H_
