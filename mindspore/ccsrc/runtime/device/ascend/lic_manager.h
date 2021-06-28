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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PASS_SWITCH_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PASS_SWITCH_MANAGER_H_

#include <memory>
#include <string>
#include <map>
#include <vector>

namespace mindspore {
enum class OptPassEnum {
  MatmulBiasaddFusion,
  MulAddFusion,
  ReshapeTransposeFusion,
  SoftmaxGradExtFusion,
  SquareSumFusion,
  TransposeReshapeFusion,
  ClipByNormNoDivSquareSumFusion,
  MomentumLossscaleFusion,
  DereluFusion,
  FusedBatchNormFusion,
  MatmulEltwiseFusionPass,
  BatchMatmulFusedMulAddFusionPass,
  EltwiseFusionPass,
  MultiOutputFusionPass,
  BnupdateEltwiseEltwiseFusionPass,
  BnupdateEltwiseFusionPass,
  Conv2DBackpropEltwiseFusionPass,
  ConvBnReduceFusionPass,
};

class LicManager {
 public:
  static LicManager &GetInstance();
  bool GetPassSwitch(OptPassEnum pass);

 private:
  void ParseSwitch();
  void ParseFeSwitch(const std::string &options_str);

  bool init_flag = false;
  std::map<OptPassEnum, bool> pass_switch_ = {};
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_PASS_SWITCH_MANAGER_H_
