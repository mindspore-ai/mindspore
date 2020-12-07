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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AI_CPU_DYNAMIC_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AI_CPU_DYNAMIC_KERNEL_H_

#include <string>
#include <memory>
#include "runtime/device/executor/dynamic_kernel.h"
#include "ir/anf.h"
#include "runtime/device/ascend/executor/aicpu_ext_info_handle.h"

namespace mindspore {
namespace device {
namespace ascend {
class AiCpuDynamicKernel : public DynamicKernel {
 public:
  AiCpuDynamicKernel(void *stream, const CNodePtr &cnode_ptr, const std::string &args, const std::string &ext_info_data,
                     const std::string &so_name, const std::string &kernel_name)
      : DynamicKernel(stream, cnode_ptr),
        args_(args),
        ext_info_data_(ext_info_data),
        so_name_(so_name),
        kernel_name_(kernel_name),
        ext_info_handler_(nullptr),
        ext_info_addr_dev_(nullptr),
        ext_info_size_(0),
        input_num_(0),
        output_num_(0),
        unknow_type_(DEPEND_IN_SHAPE) {}

  ~AiCpuDynamicKernel() override;

  void UpdateArgs() override;
  void Execute() override;
  void Initialize() override;
  void PostExecute() override;

  // Get Compute Shape from ExtInfo
  bool UpdateOutputShapeFromExtInfo();

 private:
  std::string args_;
  std::string ext_info_data_;
  std::string so_name_;
  std::string kernel_name_;

  std::shared_ptr<AicpuExtInfoHandler> ext_info_handler_;
  void *ext_info_addr_dev_;
  size_t ext_info_size_;

  size_t input_num_;
  size_t output_num_;

  UnknowShapeOpType unknow_type_;

  bool UpdateInputOutputAddr();
  bool UpdateExtInfo();
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AI_CPU_DYNAMIC_KERNEL_H_
