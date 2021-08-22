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

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "src/custom_common.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const auto kFloat32 = DataType::kNumberTypeFloat32;
}
class CustomAddKernel : public Kernel {
 public:
  CustomAddKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                  const schema::Primitive *primitive, const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}
  ~CustomAddKernel() = default;

  // Prepare will be called during graph compilation
  int Prepare() override { return lite::RET_OK; }

  // Execute is called to compute.
  int Execute() override {
    if (inputs_.size() != 2) {
      return lite::RET_PARAM_INVALID;
    }
    PreProcess();
    ParseAttrData();
    const float *in0 = static_cast<const float *>(inputs_[0].Data().get());
    const float *in1 = static_cast<const float *>(inputs_[1].Data().get());
    float *out = static_cast<float *>(outputs_[0].MutableData());
    auto num = outputs_[0].ElementNum();
    for (int i = 0; i < num; ++i) {
      out[i] = in0[i] + in1[i];
    }
    return lite::RET_OK;
  }

  // Resize is used to update some parameters if current node can change along with inputs.
  int ReSize() override { return lite::RET_OK; }

 private:
  // if output shape exists value -1, need to be inferred before applying memory for output tensor.
  int PreProcess() {
    if (common::CheckOutputs(outputs_) != lite::RET_OK) {
      auto status =
        registry::RegisterKernelInterface::GetKernelInterface({}, primitive_)->Infer(&inputs_, &outputs_, primitive_);
      if (status != kSuccess) {
        std::cerr << "infer failed." << std::endl;
        return lite::RET_ERROR;
      }
      auto ret = ReSize();
      if (ret != lite::RET_OK) {
        std::cerr << "resize failed." << std::endl;
        return ret;
      }
    }
    for (auto &output : outputs_) {
      // malloc data for output tensor
      auto data = output.MutableData();
      if (data == nullptr) {
        std::cerr << "Get data failed" << std::endl;
        return lite::RET_ERROR;
      }
    }
    return lite::RET_OK;
  }

  // fetch attributes if user need.
  void ParseAttrData() {
    auto prim = primitive_->value_as_Custom();
    if (prim->attr()->size() < 1) {
      return;
    }
    for (size_t i = 0; i < prim->attr()->size(); ++i) {
      auto attr = prim->attr()->Get(0);
      auto attr_key = attr->name()->str();
      auto data_bytes = attr->data();
      auto data_size = data_bytes->size();
      char buf[100];
      for (size_t j = 0; j < data_size; ++j) {
        buf[j] = static_cast<char>(data_bytes->Get(j));
      }
      buf[data_size] = 0;
      attrs_[attr_key] = std::string(buf);
    }
  }
  std::map<std::string, std::string> attrs_;
};

std::shared_ptr<Kernel> CustomAddCreator(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                                         const schema::Primitive *primitive, const mindspore::Context *ctx) {
  return std::make_shared<CustomAddKernel>(inputs, outputs, primitive, ctx);
}
REGISTER_CUSTOM_KERNEL(CPU, Tutorial, kFloat32, Custom_Add, CustomAddCreator)
}  // namespace kernel
}  // namespace mindspore
