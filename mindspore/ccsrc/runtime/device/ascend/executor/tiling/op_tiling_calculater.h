/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_TILING_OP_TILING_CALCULATE_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_TILING_OP_TILING_CALCULATE_H_

#include <map>
#include <memory>
#include <string>
#include "utils/ms_utils.h"
#include "utils/contract.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "register/op_tiling.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace device {
namespace ascend {
class OpTilingCalculater {
 public:
  static OpTilingCalculater &GetInstance() {
    static OpTilingCalculater instance;
    return instance;
  }

  void Init();
  void CalculateTiling(const NotNull<CNodePtr> &cnode, const optiling::OpCompileInfo &op_compile_info,
                       const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
                       NotNull<optiling::OpRunInfo *> op_run_info);

 private:
  OpTilingCalculater() = default;
  ~OpTilingCalculater() = default;
  DISABLE_COPY_AND_ASSIGN(OpTilingCalculater);

  std::map<std::string, optiling::OpTilingFunc> tiling_func_map_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_TILING_OP_TILING_CALCULATE_H_
