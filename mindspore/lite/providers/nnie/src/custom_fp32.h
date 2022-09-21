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

#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_SRC_CUSTOM_FP32_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_SRC_CUSTOM_FP32_H_

#include <vector>
#include <string>
#include <memory>
#include "include/schema/model_generated.h"
#include "include/api/kernel.h"
#include "src/custom_infer.h"
#include "include/hi_type.h"
#include "src/nnie_cfg_parser.h"
#include "src/nnie_manager.h"
#include "src/nnie_print.h"
#include "src/custom_allocator.h"

using mindspore::MSTensor;
using mindspore::kernel::Kernel;
namespace mindspore {
namespace nnie {
class CustomCPUKernel : public Kernel {
 public:
  CustomCPUKernel(nnie::NNIEManager *manager, int seg_id, bool forward_bbox, const std::vector<MSTensor> &inputs,
                  const std::vector<MSTensor> &outputs, const mindspore::schema::Primitive *primitive,
                  const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx), manager_(manager), seg_id_(seg_id), forward_bbox_(forward_bbox) {
    if ((manager_) == nullptr) {
      LOGE("manager_ is nullptr.");
    } else {
      manager_->SetMaxSegId(seg_id);
    }
  }

  ~CustomCPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Execute() override;

  int seg_id(void) const { return seg_id_; }

  void set_seg_id(int id) { seg_id_ = id; }

  int forward_bbox(void) const { return forward_bbox_; }

  void set_forward_bbox(bool flag) { forward_bbox_ = flag; }

 private:
  nnie::NNIEManager *manager_ = nullptr;
  int seg_id_ = 0;
  bool forward_bbox_ = false;
  std::vector<std::vector<int64_t>> outputs_shapes_;
};
}  // namespace nnie
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_NNIE_SRC_CUSTOM_FP32_H_
