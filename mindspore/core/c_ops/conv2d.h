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

#ifndef MINDSPORE_CORE_C_OPS_CONV2D_H
#define MINDSPORE_CORE_C_OPS_CONV2D_H
#include <map>
#include <vector>
#include <string>
#include "c_ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
class Conv2d : public PrimitiveC {
 public:
  Conv2d() : PrimitiveC(kConv2DName) { InitIOName({"x", "w"}, {"output"}); }
  void Init(int out_channel, const std::vector<int> &kernel_size, int mode = 1, const std::string &pad_mode = "valid",
            const std::vector<int> &pad = {0, 0, 0, 0}, const std::vector<int> &stride = {1, 1, 1, 1},
            const std::vector<int> &dilation = {1, 1, 1, 1}, int group = 1);
  std::vector<int> GetKernelSize() const {
    auto value_ptr = this->GetAttr(kKernelSize);
    return GetValue<std::vector<int>>(value_ptr);
  }
  std::vector<int> GetStride() const {
    auto value_ptr = GetAttr(kStride);
    return GetValue<std::vector<int>>(value_ptr);
  }
  std::vector<int> GetDilation() const {
    auto value_ptr = GetAttr(kDilation);
    return GetValue<std::vector<int>>(value_ptr);
  }
  std::string GetPadMode() const {
    auto value_ptr = this->GetAttr(kPadMode);
    return GetValue<string>(value_ptr);
  }
  std::vector<int> GetPad() const {
    auto value_ptr = this->GetAttr(kPad);
    return GetValue<std::vector<int>>(value_ptr);
  }
  int GetMode() const {
    auto value_ptr = this->GetAttr(kMode);
    return GetValue<int>(value_ptr);
  }

  int GetGroup() const {
    auto value_ptr = this->GetAttr(kGroup);
    return GetValue<int>(value_ptr);
  }
  int GetOutputChannel() const {
    auto value_ptr = this->GetAttr(kOutputChannel);
    return GetValue<int>(value_ptr);
  }

  void SetKernelSize(const std::vector<int> &kernel_size) { this->AddAttr(kKernelSize, MakeValue(kernel_size)); }
  void SetStride(const std::vector<int> &stride) { this->AddAttr(kStride, MakeValue(stride)); }
  void SetDilation(const std::vector<int> &dilation) { this->AddAttr(kDilation, MakeValue(dilation)); }
  void SetPadMode(const std::string &pad_mode) { this->AddAttr(kPadMode, MakeValue(pad_mode)); }
  void SetPad(const std::vector<int> &pad) { this->AddAttr(kPad, MakeValue(pad)); }
  void SetMode(int mode) { this->AddAttr(kMode, MakeValue(mode)); }
  void SetGroup(int group) { this->AddAttr(kGroup, MakeValue(group)); }
  void SetOutChannel(int output_channel) { this->AddAttr(kOutputChannel, MakeValue(output_channel)); }
  void SetPadList(const std::vector<int> &pad_list) { this->AddAttr(kPadList, MakeValue(pad_list)); }

 private:
  inline static const string kKernelSize = "kernel_size";
  inline static const string kStride = "stride";
  inline static const string kDilation = "dilation";
  inline static const string kPadMode = "pad_mode";
  inline static const string kPad = "pad";
  inline static const string kMode = "mode";
  inline static const string kGroup = "group";
  inline static const string kOutputChannel = "output channel";
  inline static const string kPadList = "pad_list";
  inline static const string kConv2DName = "Conv2D";
};
AbstractBasePtr Conv2dInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_C_OPS_CONV2D_H
