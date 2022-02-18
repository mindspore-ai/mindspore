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

#include "src/proposal_infer.h"
#include <memory>
#include <vector>
#include "include/errorcode.h"
#include "src/proposal.h"
#include "include/api/format.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace proposal {
std::shared_ptr<KernelInterface> ProposalInferCreater() {
  auto infer = std::make_shared<ProposalInterface>();
  if (infer == nullptr) {
    LOGE("new custom infer is nullptr");
    return nullptr;
  }

  return infer;
}
Status ProposalInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                                const mindspore::schema::Primitive *primitive) {
  if (inputs->size() != 2) {
    LOGE("Inputs size less 2");
    return kLiteError;
  }
  if (outputs->size() == 0) {
    LOGE("Outputs size 0");
    return kLiteError;
  }
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    LOGE("Primitive type is not PrimitiveType_Custom");
    return kLiteError;
  }

  size_t id = 0;
  while (id < outputs->size()) {
    std::vector<int64_t> shape{-1, COORDI_NUM};
    (*outputs)[id].SetShape(shape);
    (*outputs)[id].SetDataType(DataType::kNumberTypeFloat32);
    (*outputs)[id].SetFormat(Format::NCHW);
    id++;
  }
  return kSuccess;
}
}  // namespace proposal
}  // namespace mindspore
namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(NNIE, Proposal, proposal::ProposalInferCreater);
}  // namespace kernel
}  // namespace mindspore
