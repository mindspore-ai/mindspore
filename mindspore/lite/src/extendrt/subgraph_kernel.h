/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_SUBGRAPH_KERNEL_H
#define MINDSPORE_LITE_EXTENDRT_SUBGRAPH_KERNEL_H
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/status.h"
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
#include "ccsrc/kernel/kernel.h"
struct KernelInfo;
class SubGraphKernel : public KernelMod {
 public:
  SubGraphKernel() = default;
  virtual ~SubGraphKernel() = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

 protected:
  std::vector<KernelInfo> kernels_;
};
#endif
