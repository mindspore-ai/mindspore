#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t formerNum,
            uint32_t tailNum, uint32_t formerLength, uint32_t tailLength, uint32_t alignNum)
        {
            // 由于不同的核有不同的数据量，所以不同的核有不同的offset
            if (GetBlockIdx() < formerNum) {
                this->tileLength = formerLength;
                xGm.SetGlobalBuffer((__gm__ half *)x + formerLength * GetBlockIdx(), formerLength);
                yGm.SetGlobalBuffer((__gm__ half *)y + formerLength * GetBlockIdx(), formerLength);
                zGm.SetGlobalBuffer((__gm__ half *)z + formerLength * GetBlockIdx(), formerLength);
            } else {
                this->tileLength = tailLength;
                xGm.SetGlobalBuffer(
                    (__gm__ half *)x + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
                yGm.SetGlobalBuffer(
                    (__gm__ half *)y + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
                zGm.SetGlobalBuffer(
                    (__gm__ half *)z + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
            }
            ASSERT(alignNum != 0 && "align num can not be zero!");
            // 切分后有些数量不满足32B对齐，所以需要对length向上对齐到32B的数据量
            pipe.InitBuffer(inQueueX, BUFFER_NUM,
                (((this->tileLength + alignNum - 1) / alignNum) * alignNum) * sizeof(half));
            pipe.InitBuffer(inQueueY, BUFFER_NUM,
                (((this->tileLength + alignNum - 1) / alignNum) * alignNum) * sizeof(half));
            pipe.InitBuffer(outQueueZ, BUFFER_NUM,
                (((this->tileLength + alignNum - 1) / alignNum) * alignNum) * sizeof(half));
    }
    __aicore__ inline void Process()
    {
  
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm, this->tileLength);
        DataCopy(yLocal, yGm, this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Sub(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm, zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};


extern "C" __global__ __aicore__ void sub_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelSub op;
    op.Init(x, y, z, tilingData.formerNum, tilingData.tailNum, tilingData.formerLength,
        tilingData.tailLength, tilingData.alignNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}