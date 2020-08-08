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

#include "tools/converter/legacy_optimizer/const_fold/mul_const_fold_pass.h"
#include "utils/log_adapter.h"
#include "converter/common/tensor_util.h"
#include "converter/common/converter_op_utils.h"
#include "src/operator/cpu/creator/mul.h"

namespace mindspore {
namespace lite {
STATUS MulConstFoldPass::Run(GraphNode *graphNode) { return ConstFoldPass::Run(graphNode); }

STATUS MulConstFoldPass::CreateOp(SubGraphDefT *subGraph, OpDefT *node) {
  InnerContext ctx;
  OpDesc desc{};
  desc.type = OpT_Mul;
  desc.arch = kCPU;
  MS_ASSERT(inputs.size() == kArithOpInputNum);
  auto inTensor0 = inputs.at(kArithOpInputTensorIndex0);
  auto inTensor1 = inputs.at(kArithOpInputTensorIndex1);
  MS_ASSERT(inTensor0 != nullptr);
  MS_ASSERT(inTensor1 != nullptr);
  DataType dataType;
  if (inTensor0->GetNDim() > 1) {
    dataType = inTensor0->GetDataType();
  } else {
    dataType = inTensor1->GetDataType();
  }
  op = nullptr;
  switch (dataType) {
    case DataType_DT_UINT8: {
      op = new (std::nothrow) OpMul<uint8_t>(inputs, outputs, *PackOpDefT(node), &ctx, desc);
    } break;
    case DataType_DT_INT32: {
      op = new (std::nothrow) OpMul<int32_t>(inputs, outputs, *PackOpDefT(node), &ctx, desc);
    } break;
    case DataType_DT_FLOAT: {
      op = new (std::nothrow) OpMul<float>(inputs, outputs, *PackOpDefT(node), &ctx, desc);
    } break;
    case DataType_DT_INT8: {
      op = new (std::nothrow) OpMul<int8_t>(inputs, outputs, *PackOpDefT(node), &ctx, desc);
    } break;
    case DataType_DT_UINT32: {
      op = new (std::nothrow) OpMul<uint32_t>(inputs, outputs, *PackOpDefT(node), &ctx, desc);
    } break;
    default: {
      MS_LOGE("Unsupported dataType: %d", dataType);
      return RET_ERROR;
    }
  }
  if (op == nullptr) {
    MS_LOGE("new OpMul return nullptr");
    return RET_ERROR;
  }
  auto ret = op->InferShape(inputs, outputs);
  if (ret != RET_OK) {
    MS_LOGE("OpMul InferShape Failed");
    return RET_ERROR;
  }
  ret = op->Init(inputs, outputs);
  if (ret != RET_OK) {
    MS_LOGE("OpMul Init Failed");
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MulConstFoldPass::DoFold(SubGraphDefT *subGraph, OpDefT *node) {
  MS_ASSERT(op != nullptr);
  auto ret = op->Execute(inputs, outputs);
  if (ret != RET_OK) {
    MS_LOGE("OpMul Execute Failed");
    return RET_ERROR;
  }

  if (node->outputIndex.size() != kArithOpOutputNum) {
    MS_LOGE("The number of output for mul must be %u, nodeName: %s", kArithOpOutputNum, node->name.c_str());
    return RET_ERROR;
  }
  this->outputTensor = subGraph->allTensors.at(node->outputIndex.front()).get();
  CopyTensor2TensorDefT(outputs.front(), this->outputTensor);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

