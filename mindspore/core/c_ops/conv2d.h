/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_C_OPS_CONV2D_H_
#define MINDSPORE_CORE_C_OPS_CONV2D_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "c_ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
class Conv2d : public PrimitiveC {
 public:
  Conv2d();
  ~Conv2d() = default;
  void Init(int out_channel, const std::vector<int> &kernel_size, int mode = 1, const std::string &pad_mode = "valid",
            const std::vector<int> &pad = {0, 0, 0, 0}, const std::vector<int> &stride = {1, 1, 1, 1},
            const std::vector<int> &dilation = {1, 1, 1, 1}, int group = 1);
  std::vector<int> GetKernelSize() const;
  std::vector<int> GetStride() const;
  std::vector<int> GetDilation() const;
  std::string GetPadMode() const;
  std::vector<int> GetPad() const;
  int GetMode() const;
  int GetGroup() const;
  int GetOutputChannel() const;
  void SetKernelSize(const std::vector<int> &kernel_size);
  void SetStride(const std::vector<int> &stride);
  void SetDilation(const std::vector<int> &dilation);
  void SetPadMode(const std::string &pad_mode);
  void SetPad(const std::vector<int> &pad);
  void SetMode(int mode);
  void SetGroup(int group);
  void SetOutChannel(int output_channel);
  void SetPadList(const std::vector<int> &pad_list);
};
AbstractBasePtr Conv2dInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
using PrimConv2dPtr = std::shared_ptr<Conv2d>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_CONV2D_H_
