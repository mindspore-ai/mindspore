/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_QUERY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_QUERY_H_

#include <vector>
#include "extendrt/delegate/ascend_native/ascend_native_impl/encoder.h"

namespace mindspore::ascend_native {

class AscendNativeQuery : public AscendNativeEncoder {
 public:
  AscendNativeQuery() : AscendNativeEncoder(false) {}
  virtual ~AscendNativeQuery() {}
  void QEmbedding(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q);
  void Attn(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q) override;
  void HeadPangu(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q);
  size_t GetWorkspaceSize(const EncoderParams &p) override;
  void Forward(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q) override;
};

}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_QUERY_H_
