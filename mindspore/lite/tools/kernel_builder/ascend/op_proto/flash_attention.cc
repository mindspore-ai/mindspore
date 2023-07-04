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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "./flash_attention.h"
namespace ge {
IMPLEMT_COMMON_INFERFUNC(FlashAttentionInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(op.GetInputDescByName("q").GetShape());
  output_desc.SetDataType(op.GetInputDescByName("q").GetDataType());
  output_desc.SetFormat(op.GetInputDescByName("q").GetFormat());
  auto ret = op.UpdateOutputDesc("y", output_desc);
  if (ret != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(FlashAttention, FlashAttentionVerify) { return GRAPH_SUCCESS; }

COMMON_INFER_FUNC_REG(FlashAttention, FlashAttentionInferShape);
VERIFY_FUNC_REG(FlashAttention, FlashAttentionVerify);

}  // namespace ge
