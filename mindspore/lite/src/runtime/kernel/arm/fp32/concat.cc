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

#include "src/runtime/kernel/arm/fp32/concat.h"
#include <vector>
#include "src/runtime/kernel/arm/nnacl/fp32/concat.h"
#include "src/kernel_registry.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {
    int ConcatCPUKernel::Init() {
      ConcatBaseCPUKernel::Init();
      schema::Format input0_format = inputs_[0]->GetFormat();
      bool need_convert_format = false;
      for (size_t i = 1; i < inputs_.size(); ++i) {
        if (inputs_[i]->GetFormat() != input0_format) {
          need_convert_format = true;
        }
      }
      if (!need_convert_format) {
        outputs_[0]->SetFormat(input0_format);
        return RET_OK;
      }
      MS_LOG(ERROR) << "All input format should be the same!";
      return RET_ERROR;
    }

    int ConcatCPUKernel::ReSize() { return RET_OK; }

    int ConcatCPUKernel::Run() {
      auto input_num = inputs_.size();
      std::vector<void *> inputs_addr(input_num, nullptr);
      std::vector<int *> inputs_output_shape(input_num + 1, nullptr);

      std::vector <std::vector<int>> shapes;
      for (size_t i = 0; i < input_num; ++i) {
        inputs_addr[i] = inputs_[i]->Data();
        shapes.push_back(inputs_[i]->shape());
        inputs_output_shape[i] = shapes[i].data();
      }
      auto output_shape = outputs_.at(0)->shape();
      inputs_output_shape[input_num] = output_shape.data();
      auto output_addr = outputs_.at(0)->Data();

      Concat(reinterpret_cast<void **>(inputs_addr.data()), input_num, axis_, inputs_output_shape.data(),
             output_shape.size(), output_addr);
      return RET_OK;
    }
}  // namespace mindspore::kernel


