/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "drop_out_gen_mask_kernels.h"
#include <cfloat>
#include <ctime>
#include <random>
#include <memory.h>

#include "aicpu_sharder/aicpu_sharder.h"
#include "common/kernel_errcode.h"
#include "common/kernel_log.h"

#include "Eigen/Core"

namespace aicpu {
namespace {
std::random_device e;
const size_t kIndexOutput = 4;
const size_t kOffsetIndex2 = 2;
const size_t kOffsetIndex3 = 3;
const size_t kAlign = 127;
}  // namespace

#if (defined __ARM_ARCH) || (defined PLATFORM_AARCH64)  // compiled on arm arch
#define CONFIG_ENABLE_PERIOD_64BIT
static void OffsetAdd(uint64_t number, const uint64_t *baseOffset, uint64_t *offset) {
  uint64_t tmpBaseOffset0 = baseOffset[0];
  uint64_t tmpBaseOffset1 = baseOffset[1];
  offset[0] = tmpBaseOffset0 + number;
  offset[1] = tmpBaseOffset1;
  if (offset[0] < tmpBaseOffset0) {
    offset[1]++;
  }
}

static void ARMDropOutGenMaskKernel(const uint64_t count, const float prob, const uint8_t *offset, const uint8_t *key,
                                    uint8_t *out) {
  const uint16_t threshold = static_cast<uint16_t>(UINT16_MAX * prob);
  const uint8_t in_offset[16] = {0x01, 0, 0, 0, 0, 0, 0, 0, 0x01};
  const uint8_t inc_step[16] = {0x02};

  // a const key. reference paper: https://dl.acm.org/citation.cfm?id=206340
  const uint8_t key_const[16] = {0xBB, 0x67, 0xAE, 0x85, 0x84, 0xCA, 0xA7, 0x3B,
                                 0x9E, 0x37, 0x79, 0xB9, 0x7F, 0x4A, 0x7C, 0x15};

  const uint8_t *key_const_ptr = &(key_const[0]);
  const uint8_t *inc_step_ptr = &(inc_step[0]);

  // Each iteration generates 4-bit * 8 elements (in vector reg) * 4 (repeated code blocks)
  const uint64_t loop_time = count / 4 / 8 / 4;
  __asm volatile(
    ".arch  armv8-a+crypto \n"

    "ldr x0, %[loop_time] \n"

    "ldr x16, %[key_const_ptr] \n"
    "ld1 {v2.16b}, [x16] \n"

    // generate in1
    "ldr x1, %[offset] \n"
    "ld1 {v0.16b}, [x1] \n"  // tmp input

    "ldr x2, %[key] \n"
    "ld1 {v1.16b}, [x2] \n"       // first round key
    "add v5.2d, v1.2d, v2.2d \n"  // second round key

    // generate in2
    "ldp x10, x11, %[in_offset] \n"
    "ldp x12, x13, [x1] \n"
    "adds x14, x12, x10 \n"
    "adc x15, x13, x11 \n"
    "mov v10.d[0], x14 \n"
    "mov v10.d[1], x15 \n"

    // generate in3 = in1
    "mov v3.16b, v0.16b \n"
    // generate in4 = in2
    "mov v13.16b, v10.16b \n"

  // load input inc step
#ifdef CONFIG_ENABLE_PERIOD_64BIT
    "ldr x17, %[inc_step_ptr] \n"
    "ld1 {v4.16b}, [x17] \n"
#else
    "ldp x10, x11, %[inc_step] \n"
#endif

    "ldr w7, %[threshold] \n"
    "dup v20.8h, w7 \n"

    // Generate 16 bitmasks to 16 regs
    "mov w7, #0x8000 \n"
    "dup v21.8h, w7 \n"
    "mov w7, #0x4000 \n"
    "dup v12.8h, w7 \n"
    "mov w7, #0x2000 \n"
    "dup v2.8h, w7 \n"
    "mov w7, #0x1000 \n"
    "dup v6.8h, w7 \n"

    "mov w7, #0x800 \n"
    "dup v7.8h, w7 \n"
    "mov w7, #0x400 \n"
    "dup v8.8h, w7 \n"
    "mov w7, #0x200 \n"
    "dup v14.8h, w7 \n"
    "mov w7, #0x100 \n"
    "dup v15.8h, w7 \n"

    "mov w7, #0x80 \n"
    "dup v26.8h, w7 \n"
    "mov w7, #0x40 \n"
    "dup v27.8h, w7 \n"
    "mov w7, #0x20 \n"
    "dup v9.8h, w7 \n"
    "mov w7, #0x10 \n"
    "dup v11.8h, w7 \n"

    "mov w7, #0x8 \n"
    "dup v16.8h, w7 \n"
    "mov w7, #0x4 \n"
    "dup v17.8h, w7 \n"
    "mov w7, #0x2 \n"
    "dup v18.8h, w7 \n"
    "mov w7, #0x1 \n"
    "dup v19.8h, w7 \n"

    // load out pointer addr to register
    "ldr x5, %[out] \n"

    // Iteration begins
    ".ARS: \n"

    /* Mix v0 with v1 */
    "aese   v0.16b, v1.16b \n"
    "aesmc  v0.16b, v0.16b \n"

    /* Mix v10 with v5 */
    "aese   v10.16b, v5.16b \n"
    "aesmc  v10.16b, v10.16b \n"

    /* Compare the random number v0 against threshold */
    "cmhs v22.8h, v20.8h, v0.8h \n"
    /* Update the output register with v0 */
    "bit v29.16b, v22.16b, v21.16b \n"

    /* Mix v13 with v1 */
    "aese   v13.16b, v1.16b \n"
    "aesmc  v13.16b, v13.16b \n"

    /* Compare the random number v10 against threshold */
    "cmhs v23.8h, v20.8h, v10.8h \n"
    /* Update the output register with v10 */
    "bit v29.16b, v23.16b, v12.16b \n"

    /* Mix v3 with v5 */
    "aese   v3.16b, v5.16b \n"
    "aesmc  v3.16b, v3.16b \n"

    /* Compare the random number v13 against threshold */
    "cmhs v25.8h, v20.8h, v13.8h \n"
    /* Update the output register with v13 */
    "bit v29.16b, v25.16b, v6.16b \n"

    /* Mix v0 with v1 */
    "aese   v0.16b, v1.16b \n"
    "aesmc  v0.16b, v0.16b \n"

    /* Compare the random number v3 against threshold */
    "cmhs v24.8h, v20.8h, v3.8h \n"
    /* Update the output register with v3 */
    "bit v29.16b, v24.16b, v2.16b \n"

    /* Mix v10 with v5 */
    "aese   v10.16b, v5.16b \n"
    "aesmc  v10.16b, v10.16b \n"

    /* Compare the random number v0 against threshold */
    "cmhs v22.8h, v20.8h, v0.8h \n"
    /* Update the output register with v0 */
    "bit v29.16b, v22.16b, v7.16b \n"

    /* Mix v13 with v1 */
    "aese   v13.16b, v1.16b \n"
    "aesmc  v13.16b, v13.16b \n"

    /* Compare the random number v10 against threshold */
    "cmhs v23.8h, v20.8h, v10.8h \n"
    /* Update the output register with v10 */
    "bit v29.16b, v23.16b, v8.16b \n"

    /* Mix v3 with v5 */
    "aese   v3.16b, v5.16b \n"
    "aesmc  v3.16b, v3.16b \n"

    /* Compare the random number v13 against threshold */
    "cmhs v25.8h, v20.8h, v13.8h \n"
    /* Update the output register with v13 */
    "bit v29.16b, v25.16b, v14.16b \n"

    /* Mix v0 with v1 */
    "aese   v0.16b, v1.16b \n"
    "aesmc  v0.16b, v0.16b \n"

    /* Compare the random number v3 against threshold */
    "cmhs v24.8h, v20.8h, v3.8h \n"
    /* Update the output register with v3 */
    "bit v29.16b, v24.16b, v15.16b \n"

    /* Mix v10 with v5 */
    "aese   v10.16b, v5.16b \n"
    "aesmc  v10.16b, v10.16b \n"

    /* Compare the random number v0 against threshold */
    "cmhs v22.8h, v20.8h, v0.8h \n"
    /* Update the output register with v0 */
    "bit v29.16b, v22.16b, v26.16b \n"

    /* Mix v13 with v1 */
    "aese   v13.16b, v1.16b \n"
    "aesmc  v13.16b, v13.16b \n"

    /* Compare the random number v10 against threshold */
    "cmhs v23.8h, v20.8h, v10.8h \n"
    /* Update the output register with v10 */
    "bit v29.16b, v23.16b, v27.16b \n"

    /* Mix v3 with v5 */
    "aese   v3.16b, v5.16b \n"
    "aesmc  v3.16b, v3.16b \n"

    /* Compare the random number v13 against threshold */
    "cmhs v25.8h, v20.8h, v13.8h \n"
    /* Update the output register with v13 */
    "bit v29.16b, v25.16b, v9.16b \n"

    /* Mix v0 with v1 */
    "aese   v0.16b, v1.16b \n"
    "aesmc  v0.16b, v0.16b \n"

    /* Compare the random number v3 against threshold */
    "cmhs v24.8h, v20.8h, v3.8h \n"
    /* Update the output register with v3 */
    "bit v29.16b, v24.16b, v11.16b \n"

    /* Mix v10 with v5 */
    "aese   v10.16b, v5.16b \n"
    "aesmc  v10.16b, v10.16b \n"

    /* Compare the random number v0 against threshold */
    "cmhs v22.8h, v20.8h, v0.8h \n"
    /* Update the output register with v0 */
    "bit v29.16b, v22.16b, v16.16b \n"

    /* Mix v13 with v1 */
    "aese   v13.16b, v1.16b \n"
    "aesmc  v13.16b, v13.16b \n"

    /* Compare the random number v10 against threshold */
    "cmhs v23.8h, v20.8h, v10.8h \n"
    /* Update the output register with v10 */
    "bit v29.16b, v23.16b, v17.16b \n"

    /* Mix v3 with v5 */
    "aese   v3.16b, v5.16b \n"
    "aesmc  v3.16b, v3.16b \n"

    /* Compare the random number v13 against threshold */
    "cmhs v25.8h, v20.8h, v13.8h \n"
    /* Update the output register with v13 */
    "bit v29.16b, v25.16b, v18.16b \n"

  // Update the key
#ifdef CONFIG_ENABLE_PERIOD_64BIT
    "add v1.2d, v1.2d, v4.2d \n"
    "add v5.2d, v5.2d, v4.2d \n"
#else
    "mov x12, v1.d[0] \n"
    "mov x13, v1.d[1] \n"
    "adds x14, x12, x10 \n"
    "adc x15, x13, x11 \n"
    "mov v1.d[0], x14 \n"
    "mov v1.d[1], x15 \n"

    "mov x12, v5.d[0] \n"
    "mov x13, v5.d[1] \n"
    "adds x14, x12, x10 \n"
    "adc x15, x13, x11 \n"
    "mov v5.d[0], x14 \n"
    "mov v5.d[1], x15 \n"
#endif

    /* Compare the random number v3 against threshold */
    "cmhs v24.8h, v20.8h, v3.8h \n"
    /* Update the output register with v3 */
    "bit v29.16b, v24.16b, v19.16b \n"

    // Store the output register to memory
    "st1 {v29.16b}, [x5] \n"
    "add x5, x5, 16 \n"

    // Next iteration
    "subs   x0, x0, 1 \n"
    "bne    .ARS \n"
    :
    : [ offset ] "m"(offset), [ out ] "m"(out), [ in_offset ] "m"(in_offset), [ key ] "m"(key),
      [ key_const ] "m"(key_const), [ inc_step ] "m"(inc_step), [ loop_time ] "m"(loop_time),
      [ threshold ] "m"(threshold), [ key_const_ptr ] "m"(key_const_ptr), [ inc_step_ptr ] "m"(inc_step_ptr)
    : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "w7", "x7", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17",
      "x18", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
      "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v29");
}

uint32_t DropOutGenMaskKernel::DoCompute() {
  float prob = keep_prob_;

  uint64_t bit_count = static_cast<uint64_t>(count_);
  // align to 128 and around up
  bit_count = (bit_count + kAlign) & (~kAlign);
  // transfer bit count to byte count
  uint64_t byte_count = bit_count >> 3;

  // if prob is 0, set all bits to 0
  if (prob <= FLT_EPSILON) {
    memset_s(reinterpret_cast<void *>(io_addrs_[kIndexOutput]), byte_count, 0x00, byte_count);
    return kAicpuKernelStateSucess;
  }
  // if prob is 1, set all bits to 1
  if (abs(prob - 1.0f) <= FLT_EPSILON) {
    memset_s(reinterpret_cast<void *>(io_addrs_[kIndexOutput]), byte_count, 0xff, byte_count);
    return kAicpuKernelStateSucess;
  }

  uint8_t *outBuff = reinterpret_cast<uint8_t *>(io_addrs_[kIndexOutput]);
  // cal actual bit count due to align to 128
  uint64_t key[2] = {g_key[0], g_key[1]};
  uint64_t baseOffset[2] = {g_offset[0], g_offset[1]};
  auto shards = [prob, baseOffset, key, outBuff](int64_t start, int64_t limit) {
    uint64_t bitCount = (static_cast<uint64_t>(limit - start)) << 7;  // transfer 128bits to bit
    uint8_t *pOut = outBuff;
    pOut += ((static_cast<uint64_t>(start)) << 4);  // calculate skip bytes
    uint64_t offset[2] = {0, 0};
    OffsetAdd(start << 7, baseOffset, offset);
    ARMDropOutGenMaskKernel(bitCount, prob, reinterpret_cast<const uint8_t *>(&offset),
                            reinterpret_cast<const uint8_t *>(&key), pOut);
  };
  const int64_t total_unit = static_cast<int64_t>(byte_count >> 4);
  const int64_t perUnitSize = 1;  // shard unit size
  ParallelFor(total_unit, perUnitSize, shards);
  const int64_t margin = 1021;  // the margin of offset
  OffsetAdd(bit_count + margin, g_offset, g_offset);
  auto offset0 = reinterpret_cast<uint64_t *>(io_addrs_[2]);
  auto offset1 = reinterpret_cast<uint64_t *>(io_addrs_[3]);
  offset0[0] = g_offset[0];
  offset1[0] = g_offset[1];
  outBuff = nullptr;
  return kAicpuKernelStateSucess;
}

#else  // compiled on x86 arch

uint32_t DropOutGenMaskKernel::DoCompute() {
  std::default_random_engine te(time(0));
  std::bernoulli_distribution b(keep_prob_);
  const uint8_t mask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
  uint64_t byteCount = count_ >> 3;
  out_ = reinterpret_cast<uint8_t *>(io_addrs_[kIndexOutput]);
  for (uint64_t i = 0; i < byteCount; ++i) {
    out_[i] = 0x00;
    for (const auto &m : mask) {
      if (b(te)) {
        out_[i] = static_cast<uint8_t>(out_[i] | m);
      }
    }
  }
  out_ = nullptr;
  return kAicpuKernelStateSucess;
}

#endif

uint32_t DropOutGenMaskKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> nodedef_map = node_def_.attrs();
  AICPU_LOGEVENT("InputNum=[%zu], OutputNum=[%zu], ioAddrNum=[%zu], seed exist: %d, seed2 exist: %d.",
                 node_def_.inputs_size(), node_def_.outputs_size(), io_addrs_.size(), nodedef_map.contains("seed"),
                 nodedef_map.contains("seed2"));

  aicpuops::AttrValue seed0 = nodedef_map["seed"];
  aicpuops::AttrValue seed1 = nodedef_map["seed2"];
  seed0_ = static_cast<uint64_t>(seed0.i());
  seed1_ = static_cast<uint64_t>(seed1.i());
  if (seed0_ == 0 && seed1_ == 0) {
    seed0_ = e();
    seed1_ = e();
  }
  g_key[0] = static_cast<uint64_t>(seed1_);
  g_key[1] = static_cast<uint64_t>(seed0_);
  g_offset[0] = *reinterpret_cast<uint64_t *>(io_addrs_[kOffsetIndex2]);
  g_offset[1] = *reinterpret_cast<uint64_t *>(io_addrs_[kOffsetIndex3]);

  uint64_t tmp_count = 1;
  aicpuops::Tensor shape_tensor = node_def_.inputs(0);
  aicpuops::TensorShape input_shape = shape_tensor.tensor_shape();
  aicpuops::DataType shape_dt = static_cast<::aicpuops::DataType>(shape_tensor.tensor_type());
  for (int j = 0; j < input_shape.dim_size(); j++) {
    tmp_count *= static_cast<uint64_t>(input_shape.dim(j).size());
  }
  if (shape_dt == aicpuops::MS_INT32) {
    auto input0 = reinterpret_cast<int32_t *>(io_addrs_[0]);
    count_ = 1;
    for (uint64_t index = 0; index < tmp_count; index++) {
      count_ *= static_cast<uint64_t>(input0[index]);
    }
  } else {
    auto input0 = reinterpret_cast<int64_t *>(io_addrs_[0]);
    count_ = 1;
    for (uint64_t index = 0; index < tmp_count; index++) {
      count_ *= static_cast<uint64_t>(input0[index]);
    }
  }

  aicpuops::Tensor prob_tensor = node_def_.inputs(1);
  aicpuops::DataType dt = static_cast<::aicpuops::DataType>(prob_tensor.tensor_type());
  if (dt == aicpuops::MS_FLOAT16) {
#if (defined __ARM_ARCH) || (defined PLATFORM_AARCH64)  // compiled on arm arch
    keep_prob_ = *reinterpret_cast<float16_t *>(io_addrs_[1]);
#else
    keep_prob_ = *reinterpret_cast<float *>(io_addrs_[1]);
#endif
  } else {
    keep_prob_ = *reinterpret_cast<float *>(io_addrs_[1]);
  }
  if ((keep_prob_ < 0.0f) || (keep_prob_ > 1.0f)) {
    AICPU_LOGE("The prob must be in [0,1] range, got prob info %f.", keep_prob_);
    return kAicpuKernelStateInvalid;
  }
  AICPU_LOGI("DropoutGenMask mask count and pro: %lu %f", count_, keep_prob_);
  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t DropoutGenMask(void *param) {
  aicpu::DropOutGenMaskKernel dropoutGenMaskKernel;
  return dropoutGenMaskKernel.Compute(param);
}
}
