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
#ifndef MINDSPORE_CCSRC_KERNEL_AKG_AKG_KERNEL_ATTRS_PROCESS_H
#define MINDSPORE_CCSRC_KERNEL_AKG_AKG_KERNEL_ATTRS_PROCESS_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "ir/anf.h"
#include "utils/utils.h"
#include "operator/ops.h"

namespace mindspore {
namespace kernel {
void SetAkgAttrsForFour2Five(const AnfNodePtr &anf_node);
void SetAkgAttrsForFive2Four(const AnfNodePtr &anf_node);
void SetAkgAttrsForCast(const AnfNodePtr &anf_node);
void SetAkgAttrsForBNGrad1(const AnfNodePtr &anf_node);
void SetAkgAttrsForBNGrad2(const AnfNodePtr &anf_node);
void SetAkgAttrsForBNGrad3(const AnfNodePtr &anf_node);
void SetAkgAttrsForFusedBN1(const AnfNodePtr &anf_node);
void SetAkgAttrsForFusedBN2(const AnfNodePtr &anf_node);
void SetAkgAttrsForFusedBN3(const AnfNodePtr &anf_node);
void SetAkgAttrsForConvBN1(const AnfNodePtr &anf_node);
void SetAkgAttrsForBN2AddRelu(const AnfNodePtr &anf_node);
void SetAkgAttrsForBN2Relu(const AnfNodePtr &anf_node);

const std::unordered_map<std::string, std::function<void(const AnfNodePtr &anf_node)>> kAkgKernelAttrsProcessMap = {
  {kFour2FiveOpName, SetAkgAttrsForFour2Five},
  {kFive2FourOpName, SetAkgAttrsForFive2Four},
  {"Cast", SetAkgAttrsForCast},
  {kBNGrad1OpName, SetAkgAttrsForBNGrad1},
  {kBNGrad2OpName, SetAkgAttrsForBNGrad2},
  {kBNGrad3OpName, SetAkgAttrsForBNGrad3},
  {kFusedBN1OpName, SetAkgAttrsForFusedBN1},
  {kFusedBN2OpName, SetAkgAttrsForFusedBN2},
  {kFusedBN3OpName, SetAkgAttrsForFusedBN3},
  {kConvBN1OpName, SetAkgAttrsForConvBN1},
  {kBN2AddReluOpName, SetAkgAttrsForBN2AddRelu},
  {kBN2ReLUOpName, SetAkgAttrsForBN2Relu},
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_AKG_AKG_KERNEL_ATTRS_PROCESS_H
