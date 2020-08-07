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

#ifndef MINDSPORE_PREDICT_CONCAT_V2_CONST_FOLD_PASS_H
#define MINDSPORE_PREDICT_CONCAT_V2_CONST_FOLD_PASS_H

#include <vector>
#include "converter/optimizer/const_fold/const_fold_pass.h"
#include "converter/common/tensor_util.h"
#include "utils/log_adapter.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
class ConcatV2ConstFoldPass : public ConstFoldPass {
 public:
  ConcatV2ConstFoldPass() : ConstFoldPass(OpT_Concat) {}

  ~ConcatV2ConstFoldPass() override = default;

  STATUS Run(GraphNode *graphNode) override;

  STATUS CreateOp(SubGraphDefT *subGraph, OpDefT *node) override;

  STATUS DoFold(SubGraphDefT *subGraph, OpDefT *node) override;

 private:
  template <typename T>
  STATUS DoConcat(SubGraphDefT *subGraph, const std::vector<uint32_t> &inTensorIdxes, int axis) {
    MS_ASSERT(this->outputTensor != nullptr);
    std::vector<TensorDefT *> inTensors;
    std::vector<T *> inDatas;
    for (size_t i = 0; i < inTensorIdxes.size(); i++) {
      auto &inTensor = subGraph->allTensors.at(inTensorIdxes.at(i));
      MS_ASSERT(inTensor != nullptr);
      inTensors.emplace_back(inTensor.get());
      void *inData = inTensor->data.data();
      MS_ASSERT(inData != nullptr);
      T *castedInData = static_cast<T *>(inData);
      MS_ASSERT(castedInData != nullptr);
      inDatas.emplace_back(castedInData);
    }
    auto &inShape = subGraph->allTensors.at(inTensorIdxes.at(0))->dims;
    std::vector<int32_t> outputDims;
    for (size_t i = 0; i < inShape.size(); i++) {
      if (i == axis) {
        int32_t axisDim = 0;
        for (size_t j = 0; j < inTensors.size(); j++) {
          axisDim += inTensors.at(j)->dims.at(i);
        }
        outputDims.push_back(axisDim);
        continue;
      }
      outputDims.push_back(inShape.at(i));
    }

    size_t outShapeSize = 1;
    for (auto dim : outputDims) {
      outShapeSize *= dim;
    }
    size_t elementSize = GetElementSize(subGraph->allTensors.at(inTensorIdxes.at(0))->dataType);

    this->outputTensor->dims = outputDims;
    this->outputTensor->data.clear();
    this->outputTensor->data.resize(outShapeSize * elementSize);

    void *outData = this->outputTensor->data.data();
    MS_ASSERT(outData != nullptr);
    T *castedOutData = static_cast<T *>(outData);

    size_t copyBlockTile = 1;
    for (int i = axis + 1; i < inShape.size(); i++) {
      copyBlockTile *= inShape[i];
    }
    std::vector<size_t> inCopyBlocks;
    size_t outCopyBlock = 0;
    for (size_t i = 0; i < inTensors.size(); i++) {
      inCopyBlocks.emplace_back(copyBlockTile * (inTensors.at(i)->dims.at(axis)));
      outCopyBlock += inCopyBlocks.back();
    }

    size_t outIndex = 0;
    while (outIndex < outShapeSize) {
      for (size_t i = 0; i < inDatas.size(); i++) {
        ::memcpy_s(castedOutData + outIndex, inCopyBlocks.at(i), inDatas.at(i), inCopyBlocks.at(i));
        outIndex += inCopyBlocks.at(i);
        inDatas.at(i) += inCopyBlocks.at(i);
      }
    }

    return RET_OK;
  }
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_CONCAT_V2_CONST_FOLD_PASS_H
