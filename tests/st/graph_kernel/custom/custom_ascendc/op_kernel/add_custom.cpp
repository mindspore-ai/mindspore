/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

/*
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend platform.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue

class KernelAdd {
 public:
  __aicore__ inline KernelAdd() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    this->blockLength = totalLength / GetBlockNum();
    this->tileNum = tileNum;
    ASSERT(tileNum != 0 && "tile num can not be zero!");
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

    xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), this->blockLength);
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), this->blockLength);
    zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * GetBlockIdx(), this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
  }
  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress) {
    LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
    DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }
  __aicore__ inline void Compute(int32_t progress) {
    LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
    LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
    Add(zLocal, xLocal, yLocal, this->tileLength);
    outQueueZ.EnQue<DTYPE_Z>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
    DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
  GlobalTensor<DTYPE_X> xGm;
  GlobalTensor<DTYPE_Y> yGm;
  GlobalTensor<DTYPE_Z> zGm;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  KernelAdd op;
  op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
  op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void add_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z,
                   uint8_t *workspace, uint8_t *tiling) {
  add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif