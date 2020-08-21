/**
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

#ifndef LITE_MINDSPORE_LITE_C_OPS_POOLING_GRAD_H_
#define LITE_MINDSPORE_LITE_C_OPS_POOLING_GRAD_H_

#include <vector>
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class PoolingGrad : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  explicit PoolingGrad(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
#endif
  explicit PoolingGrad(schema::Primitive *primitive) : PrimitiveC(primitive) {}

  int GetFormat() const;
  int GetPoolingMode() const;
  bool GetGlobal() const;
  int GetWindowW() const;
  int GetWindowH() const;
  int GetStrideW() const;
  int GetStrideH() const;
  int GetPadMode() const;
  int GetPadUp() const;
  int GetPadDown() const;
  int GetPadLeft() const;
  int GetPadRight() const;
  int GetRoundMode() const;
  void SetFormat(int format);
  void SetPoolingMode(int pooling_mode);
  void SetGlobal(bool global);
  void SetWindowW(int window_w);
  void SetWindowH(int window_h);
  void SetStrideW(int stride_w);
  void SetStrideH(int stride_h);
  void SetPadMode(int pad_mode);
  void SetPadUp(int pad_up);
  void SetPadDown(int pad_down);
  void SetPadLeft(int pad_left);
  void SetPadRight(int pad_right);
  void SetRoundMode(int round_mode);
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_POOLING_GRAD_H_
