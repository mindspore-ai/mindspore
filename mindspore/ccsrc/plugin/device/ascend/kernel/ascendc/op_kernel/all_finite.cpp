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
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t OUT_MIN_LEN = 16;

template <typename IN_TYPE>
class KernelAllFinite {
 public:
  __aicore__ explicit KernelAllFinite() {}
  __aicore__ inline void setArgs(GM_ADDR in, GM_ADDR out) {
    gm_x = reinterpret_cast<__gm__ IN_TYPE *>(in);
    gm_y = reinterpret_cast<__gm__ half *>(out);
    core_idx = get_block_idx();
    core_num = get_block_num();
  }
  __aicore__ inline void setTiling(uint32_t avg_block_count_in, uint32_t avg_block_ub_num_in,
                                   uint32_t avg_block_ub_tail_in, uint32_t avg_block_ub_loop_in,
                                   uint32_t avg_block_ub_real_in, uint32_t avg_block_ub_pad_in,
                                   uint32_t tail_block_count_in, uint32_t tail_block_ub_num_in,
                                   uint32_t tail_block_ub_tail_in, uint32_t tail_block_ub_loop_in,
                                   uint32_t tail_block_ub_real_in, uint32_t tail_block_ub_pad_in,
                                   uint32_t buffer_num_in, uint32_t in_dtype_in) {
    avg_block_count = avg_block_count_in;
    avg_block_ub_num = avg_block_ub_num_in;
    avg_block_ub_tail = avg_block_ub_tail_in;
    avg_block_ub_loop = avg_block_ub_loop_in;
    avg_block_ub_real = avg_block_ub_real_in;
    avg_block_ub_pad = avg_block_ub_pad_in;

    tail_block_count = tail_block_count_in;
    tail_block_ub_num = tail_block_ub_num_in;
    tail_block_ub_tail = tail_block_ub_tail_in;
    tail_block_ub_loop = tail_block_ub_loop_in;
    tail_block_ub_real = tail_block_ub_real_in;
    tail_block_ub_pad = tail_block_ub_pad_in;

    buffer_num = buffer_num_in;
    in_dtype = in_dtype_in;
  }
  __aicore__ inline void setShift(uint32_t left, uint32_t right) {
    left_shift = left;
    right_shift = right;
  }

  __aicore__ inline void Process() {
    if (core_idx >= core_num) {
      return;
    }

    uint32_t ub_count = avg_block_ub_num;
    uint32_t ub_loop = avg_block_ub_loop;
    uint32_t ub_tail = avg_block_ub_tail;
    uint32_t ub_real = avg_block_ub_real;
    uint32_t ub_pad = avg_block_ub_pad;

    if (core_idx == core_num - 1) {
      ub_count = tail_block_ub_num;
      ub_loop = tail_block_ub_loop;
      ub_tail = tail_block_ub_tail;
      ub_real = tail_block_ub_real;
      ub_pad = tail_block_ub_pad;
    }

    Init(ub_count);
    if (in_dtype == 1 || in_dtype == 27) {
      ProcessHalf(ub_count, ub_tail, ub_loop, ub_real, ub_pad);
    } else if (in_dtype == 0) {
      ProcessFp32(ub_count, ub_tail, ub_loop, ub_real, ub_pad);
    }
  }

 private:
  __aicore__ inline void ProcessHalf(uint32_t ub_count, uint32_t ub_tail, uint32_t ub_loop, uint32_t ub_real,
                                     uint32_t ub_pad) {
    AscendC::LocalTensor<uint8_t> comp_t = compQue.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint16_t> tmp_t = tmpQue.AllocTensor<uint16_t>();
    AscendC::LocalTensor<uint16_t> mask_t = maskQue.AllocTensor<uint16_t>();
    if (right_shift == 11) {                          //  half
      Duplicate(mask_t, (uint16_t)0x001F, ub_count);  //  0 00000 00000 11111
    } else {                                          //  8 bf16
      Duplicate(mask_t, (uint16_t)0x00FF, ub_count);  //  0 000 0000 1111 1111
    }

    uint32_t loop = 0;
    for (; loop < ub_loop - 1; loop++) {
      CopyIn(loop, ub_count, ub_count);
      ComputeHalf(ub_count, tmp_t, mask_t, comp_t, &loop);
    }

    /* for ub tail */
    if (ub_tail == 0 || loop >= ub_loop) {
      return;
    }
    CopyInPad(loop, ub_count, ub_tail, ub_real, ub_pad);
    ComputeHalf(ub_tail, tmp_t, mask_t, comp_t, &loop);

    /* free tmp local tensor */
    tmpQue.FreeTensor(tmp_t);
    maskQue.FreeTensor(mask_t);
    compQue.FreeTensor(comp_t);
  }

  __aicore__ inline void ProcessFp32(uint32_t ub_count, uint32_t ub_tail, uint32_t ub_loop, uint32_t ub_real,
                                     uint32_t ub_pad) {
    AscendC::LocalTensor<uint8_t> comp_t = compQue.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint32_t> tmp_t = tmpQue.AllocTensor<uint32_t>();
    AscendC::LocalTensor<uint32_t> mask_t = maskQue.AllocTensor<uint32_t>();
    Duplicate(mask_t, (uint32_t)0x00FF, ub_count);  //  0 00000000 000 0000 0000 0000 1111 1111

    uint32_t loop = 0;
    for (; loop < ub_loop - 1; loop++) {
      CopyIn(loop, ub_count, ub_count);
      ComputeFp32(ub_count, tmp_t, mask_t, comp_t, &loop);
    }

    /* for ub tail */
    if (ub_tail == 0 || loop >= ub_loop) {
      return;
    }
    CopyInPad(loop, ub_count, ub_tail, ub_real, ub_pad);
    ComputeFp32(ub_tail, tmp_t, mask_t, comp_t, &loop);

    /* free tmp local tensor */
    tmpQue.FreeTensor(tmp_t);
    maskQue.FreeTensor(mask_t);
    compQue.FreeTensor(comp_t);
  }

  __aicore__ inline void Init(uint32_t count) {
    xGm.SetGlobalBuffer(gm_x + core_idx * avg_block_count);
    yGm.SetGlobalBuffer(gm_y);
    pipe.InitBuffer(xQue, buffer_num, count * sizeof(IN_TYPE));
    pipe.InitBuffer(tmpQue, buffer_num, count * sizeof(IN_TYPE));
    pipe.InitBuffer(maskQue, buffer_num, count * sizeof(IN_TYPE));
    pipe.InitBuffer(compQue, buffer_num, count / 8 * sizeof(uint8_t));
  }

  __aicore__ inline void CopyIn(uint32_t idx, uint32_t stride, uint32_t count) {
    AscendC::LocalTensor<IN_TYPE> x = xQue.AllocTensor<IN_TYPE>();
    DataCopy(x, xGm[idx * stride], count);
    xQue.EnQue(x);
  }

  __aicore__ inline void CopyInPad(uint32_t idx, uint32_t stride, uint32_t count, uint32_t real, uint32_t pad) {
    uint32_t real_cp_size = real * sizeof(IN_TYPE);
    AscendC::DataCopyExtParams copy_params{1, real_cp_size, 0, 0, 0};
    uint8_t pad_ele_count = pad;
    AscendC::DataCopyPadExtParams<IN_TYPE> pad_param{true, 0, pad_ele_count, 0};
    AscendC::LocalTensor<IN_TYPE> x = xQue.AllocTensor<IN_TYPE>();
    Duplicate(x, (IN_TYPE)0x0, count);
    DataCopyPad(x, xGm[idx * stride], copy_params, pad_param);
    xQue.EnQue(x);
  }

  __aicore__ inline void CheckValidHalf(uint32_t count, AscendC::LocalTensor<uint16_t> shift_t,
                                        AscendC::LocalTensor<uint16_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t) {
    AscendC::LocalTensor<uint16_t> in_t = xQue.DeQue<uint16_t>();

    AscendC::ShiftLeft<uint16_t>(shift_t, in_t, left_shift, count);
    pipe_barrier(PIPE_ALL);
    AscendC::ShiftRight<uint16_t>(shift_t, shift_t, right_shift, count);
    pipe_barrier(PIPE_ALL);

    xQue.FreeTensor(in_t);

    AscendC::LocalTensor<half> shift_half_t = shift_t.ReinterpretCast<half>();
    AscendC::LocalTensor<half> mask_half_t = mask_t.ReinterpretCast<half>();

    Compare(comp_t, shift_half_t, mask_half_t, AscendC::CMPMODE::EQ, count);
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void CheckValidFp32(uint32_t count, AscendC::LocalTensor<uint32_t> shift_t,
                                        AscendC::LocalTensor<uint32_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t) {
    AscendC::LocalTensor<uint32_t> in_t = xQue.DeQue<uint32_t>();

    AscendC::ShiftLeft<uint32_t>(shift_t, in_t, 1, count);
    pipe_barrier(PIPE_ALL);
    AscendC::ShiftRight<uint32_t>(shift_t, shift_t, 24, count);
    pipe_barrier(PIPE_ALL);

    xQue.FreeTensor(in_t);

    AscendC::LocalTensor<float> shift_fp32_t = shift_t.ReinterpretCast<float>();
    AscendC::LocalTensor<float> mask_fp32_t = mask_t.ReinterpretCast<float>();

    Compare(comp_t, shift_fp32_t, mask_fp32_t, AscendC::CMPMODE::EQ, count);
    pipe_barrier(PIPE_ALL);
  }
  __aicore__ inline void CombRes(uint32_t count, uint32_t *loop, AscendC::LocalTensor<uint8_t> comp_t,
                                 AscendC::LocalTensor<uint16_t> ui16_t) {
    const int mask = 128;
    int total_count = count / 8;
    int repeat = (total_count + 127) / mask;

    AscendC::LocalTensor<half> half_comp_t = ui16_t.ReinterpretCast<half>();
    Duplicate(half_comp_t, (half)0x0, count);
    Cast(half_comp_t, comp_t, AscendC::RoundMode::CAST_NONE, total_count);
    pipe_barrier(PIPE_ALL);

    while (repeat > 1) {
      WholeReduceSum(half_comp_t, half_comp_t, mask, repeat, 1, 1, 8);
      repeat = (repeat + 127) / mask;
      total_count = (total_count + 127) / mask;
      pipe_barrier(PIPE_ALL);
    }

    WholeReduceSum(half_comp_t, half_comp_t, total_count, 1, 1, 1, 8);
    pipe_barrier(PIPE_ALL);

    float result = half_comp_t.GetValue(0);
    if (result != 0) {
      ui16_t.SetValue(0, 1);
      DataCopy(yGm[0], half_comp_t, OUT_MIN_LEN);
      *loop = count;
    }
  }

  __aicore__ inline void ComputeHalf(uint32_t count, AscendC::LocalTensor<uint16_t> tmp_t,
                                     AscendC::LocalTensor<uint16_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t,
                                     uint32_t *loop) {
    CheckValidHalf(count, tmp_t, mask_t, comp_t);
    CombRes(count, loop, comp_t, tmp_t);
  }

  __aicore__ inline void ComputeFp32(uint32_t count, AscendC::LocalTensor<uint32_t> tmp_t,
                                     AscendC::LocalTensor<uint32_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t,
                                     uint32_t *loop) {
    CheckValidFp32(count, tmp_t, mask_t, comp_t);
    CombRes(count, loop, comp_t, tmp_t.ReinterpretCast<uint16_t>());
  }

  AscendC::TPipe pipe;

  AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQue;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> tmpQue, maskQue, compQue;

  AscendC::GlobalTensor<IN_TYPE> xGm;
  AscendC::GlobalTensor<half> yGm;

  __gm__ IN_TYPE *__restrict__ gm_x{nullptr};
  __gm__ half *__restrict__ gm_y{nullptr};

  uint32_t left_shift{0};
  uint32_t right_shift{0};

  uint32_t core_idx{0};
  uint32_t core_num{0};

  uint32_t buffer_num{0};
  uint32_t in_dtype{0};

  uint32_t avg_block_count{0};
  uint32_t avg_block_ub_num{0};
  uint32_t avg_block_ub_tail{0};
  uint32_t avg_block_ub_loop{0};
  uint32_t avg_block_ub_real{0};
  uint32_t avg_block_ub_pad{0};

  uint32_t tail_block_count{0};
  uint32_t tail_block_ub_num{0};
  uint32_t tail_block_ub_tail{0};
  uint32_t tail_block_ub_loop{0};
  uint32_t tail_block_ub_real{0};
  uint32_t tail_block_ub_pad{0};
};

extern "C" __global__ __aicore__ void all_finite(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
  uint32_t avg_block_count = (uint32_t)(*((__gm__ uint32_t *)tiling + 0));
  uint32_t avg_block_ub_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 1));
  uint32_t avg_block_ub_tail = (uint32_t)(*((__gm__ uint32_t *)tiling + 2));
  uint32_t avg_block_ub_loop = (uint32_t)(*((__gm__ uint32_t *)tiling + 3));
  uint32_t avg_block_ub_real = (uint32_t)(*((__gm__ uint32_t *)tiling + 4));
  uint32_t avg_block_ub_pad = (uint32_t)(*((__gm__ uint32_t *)tiling + 5));

  uint32_t tail_block_count = (uint32_t)(*((__gm__ uint32_t *)tiling + 6));
  uint32_t tail_block_ub_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 7));
  uint32_t tail_block_ub_tail = (uint32_t)(*((__gm__ uint32_t *)tiling + 8));
  uint32_t tail_block_ub_loop = (uint32_t)(*((__gm__ uint32_t *)tiling + 9));
  uint32_t tail_block_ub_real = (uint32_t)(*((__gm__ uint32_t *)tiling + 10));
  uint32_t tail_block_ub_pad = (uint32_t)(*((__gm__ uint32_t *)tiling + 11));

  uint32_t buffer_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 12));
  uint32_t in_dtype = (uint32_t)(*((__gm__ uint32_t *)tiling + 13));

  if (in_dtype == 0) {
    KernelAllFinite<uint32_t> op;
    op.setArgs(x, z);
    op.setTiling(avg_block_count, avg_block_ub_num, avg_block_ub_tail, avg_block_ub_loop, avg_block_ub_real,
                 avg_block_ub_pad, tail_block_count, tail_block_ub_num, tail_block_ub_tail, tail_block_ub_loop,
                 tail_block_ub_real, tail_block_ub_pad, buffer_num, in_dtype);
    op.Process();
  } else if (in_dtype == 1) {
    KernelAllFinite<uint16_t> op;
    op.setArgs(x, z);
    op.setShift(1, 11);
    op.setTiling(avg_block_count, avg_block_ub_num, avg_block_ub_tail, avg_block_ub_loop, avg_block_ub_real,
                 avg_block_ub_pad, tail_block_count, tail_block_ub_num, tail_block_ub_tail, tail_block_ub_loop,
                 tail_block_ub_real, tail_block_ub_pad, buffer_num, in_dtype);
    op.Process();
  } else if (in_dtype == 27) {
    KernelAllFinite<uint16_t> op;  // bf16
    op.setArgs(x, z);
    op.setShift(1, 8);
    op.setTiling(avg_block_count, avg_block_ub_num, avg_block_ub_tail, avg_block_ub_loop, avg_block_ub_real,
                 avg_block_ub_pad, tail_block_count, tail_block_ub_num, tail_block_ub_tail, tail_block_ub_loop,
                 tail_block_ub_real, tail_block_ub_pad, buffer_num, in_dtype);
    op.Process();
  }
}
