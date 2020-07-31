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

#include <memory>
#include "minddata/dataset/engine/opt/optional/tensor_op_fusion_pass.h"
#include "minddata/dataset/kernels/image/decode_op.h"
#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/kernels/image/random_crop_decode_resize_op.h"

namespace mindspore {
namespace dataset {

Status TensorOpFusionPass::RunOnNode(std::shared_ptr<MapOp> node, bool *modified) {
  // Most primitive pattern: DecodeOp immediately followed by RandomCropAndResizeOp
  // Abstract into a more general member function that can find any pattern, expressed
  // by regular expressions, for instance.
  // Add a list of optimisation policies. For now, just this lambda
  auto FindPattern = [](auto &tfuncs) {
    auto it =
      std::find_if(tfuncs.begin(), tfuncs.end(), [](const auto &tf) -> bool { return tf->Name() == kDecodeOp; });
    auto next = it + 1;
    if (it != tfuncs.end() && next != tfuncs.end() && (*next)->Name() == kRandomCropAndResizeOp) {
      return it;
    } else {
      return tfuncs.end();
    }
  };

  auto &tfuncs = node->TFuncs();
  auto it = FindPattern(tfuncs);
  if (it != tfuncs.end()) {
    auto next = it + 1;
    auto op = static_cast<RandomCropAndResizeOp *>(next->get());
    *it = std::static_pointer_cast<TensorOp>(std::make_shared<RandomCropDecodeResizeOp>(*op));
    tfuncs.erase(next);
  }
  if (modified != nullptr) {
    *modified = true;
  } else {
    RETURN_STATUS_UNEXPECTED("modified is nullptr");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
