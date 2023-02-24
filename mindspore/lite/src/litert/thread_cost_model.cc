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

#include "src/litert/thread_cost_model.h"
#include <map>
#include "src/common/log_util.h"
#include "src/litert/inner_context.h"
#include "thread/threadpool.h"

namespace mindspore::lite {
const std::map<int32_t, float> kernel_compute_cost_map_ = {
  {TC_TYPE(schema::PrimitiveType_Activation, schema::ActivationType_RELU), 1.806f},        // dataNum about 100k
  {TC_TYPE(schema::PrimitiveType_Activation, schema::ActivationType_RELU6), 1.806f},       // dataNum about 100k
  {TC_TYPE(schema::PrimitiveType_Activation, schema::ActivationType_LEAKY_RELU), 1.806f},  // dataNum about 100k
  {TC_TYPE(schema::PrimitiveType_Activation, schema::ActivationType_TANH), 41.65625},      // dataNum about 5k
  {TC_TYPE(schema::PrimitiveType_Activation, schema::ActivationType_SIGMOID), 59.65625f},  // dataNum about 3.5k
  {TC_TYPE(schema::PrimitiveType_Activation, schema::ActivationType_GELU), 83.65625f},     // dataNum about 2.5k

  {TC_TYPE(schema::PrimitiveType_Sqrt, 0), 1.806f},    // dataNum about 100k
  {TC_TYPE(schema::PrimitiveType_Split, 0), 21.573f},  // dataNum about 8k
  {TC_TYPE(schema::PrimitiveType_Stack, 0), 9.286},    // dataNum about 12k
  {TC_TYPE(schema::PrimitiveType_Softmax, 0), 521.0},  // dataNum about 0.5k

  {TC_TYPE(schema::PrimitiveType_MulFusion, schema::ActivationType_RELU), 2.288f},           // dataNum about 80k
  {TC_TYPE(schema::PrimitiveType_MulFusion, schema::ActivationType_RELU6), 2.288f},          // dataNum about 80k
  {TC_TYPE(schema::PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION), 1.806f},  // dataNum about 100k

  {TC_TYPE(schema::PrimitiveType_AddFusion, schema::ActivationType_RELU), 2.288f},           // dataNum about 80k
  {TC_TYPE(schema::PrimitiveType_AddFusion, schema::ActivationType_RELU6), 2.288f},          // dataNum about 80k
  {TC_TYPE(schema::PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION), 1.806f},  // dataNum about 100k

  {TC_TYPE(schema::PrimitiveType_SubFusion, schema::ActivationType_RELU), 2.288f},           // dataNum about 80k
  {TC_TYPE(schema::PrimitiveType_SubFusion, schema::ActivationType_RELU6), 2.288f},          // dataNum about 80k
  {TC_TYPE(schema::PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION), 1.806f},  // dataNum about 100k

  {TC_TYPE(schema::PrimitiveType_DivFusion, schema::ActivationType_RELU), 2.288f},           // dataNum about 15k
  {TC_TYPE(schema::PrimitiveType_DivFusion, schema::ActivationType_RELU6), 2.288f},          // dataNum about 15k
  {TC_TYPE(schema::PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION), 1.806f},  // dataNum about 30k

  {TC_TYPE(schema::PrimitiveType_RealDiv, schema::ActivationType_RELU), 13.65625f},          // dataNum about 15k
  {TC_TYPE(schema::PrimitiveType_RealDiv, schema::ActivationType_RELU6), 13.65625f},         // dataNum about 15k
  {TC_TYPE(schema::PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION), 6.65625f},  // dataNum about 30k

  {TC_TYPE(schema::PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION), 6.65625f},  // dataNum about 30k
  {TC_TYPE(schema::PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION), 6.65625f},  // dataNum about 30k

  {TC_TYPE(schema::PrimitiveType_GreaterEqual, schema::ActivationType_NO_ACTIVATION), 4.90625f},  // dataNum about 40k
  {TC_TYPE(schema::PrimitiveType_LessEqual, schema::ActivationType_NO_ACTIVATION), 4.90625f},     // dataNum about 40k

  {TC_TYPE(schema::PrimitiveType_StridedSlice, 0), 38.027f},  // type 0 : parallel on outer tile, dataNum about 5.2k
  {TC_TYPE(schema::PrimitiveType_StridedSlice, 1), 42.042f},  // type 1 : parallel on split axis, dataNum about 4.5k

  {TC_TYPE(schema::PrimitiveType_BiasAdd, 0), 2.723f},  // dataNum about 65k
  {TC_TYPE(schema::PrimitiveType_Gather, 0), 11.438f},  // dataNum about 16k

  {TC_TYPE(schema::PrimitiveType_Fill, 0), 0.181f},  // dataNum about 260k(float/int IO : load 0, store 1)
  {TC_TYPE(schema::PrimitiveType_Cast, 0), 0.181f},  // dataNum about 100k(float/int IO : load 1, store 1)

  {TC_TYPE(schema::PrimitiveType_LayerNormFusion, 0), 507.812f},  // dataNum about 0.5k
  {TC_TYPE(schema::PrimitiveType_OneHot, 0), 136.562f},           // dataNum about 1.5k

  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceAll), 66.5625f},         // dataNum about 3k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceASum), 206.5625f},       // dataNum about 1k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceL2), 259.0625f},         // dataNum about 0.8k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceMax), 66.5625f},         // dataNum about 3k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceMean), 259.0625f},       // dataNum about 0.8k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceMin), 66.5625f},         // dataNum about 3k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceProd), 206.5625f},       // dataNum about 1k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceSum), 206.5625f},        // dataNum about 1k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, schema::ReduceMode_ReduceSumSquare), 206.5625f},  // dataNum about 1k
  {TC_TYPE(schema::PrimitiveType_ReduceFusion, (schema::ReduceMode_MAX + 1)), 259.0625f},        // dataNum about 0.8k
};

float ThreadCostModel::per_unit_load_cost_ = 1.0 / 64 * 11;   // 64: L2 cache size, 11 : L2 cache latency on Haswell
float ThreadCostModel::per_unit_store_cost_ = 1.0 / 64 * 11;  // 64: L2 cache size, 11 : L2 cache latency on Haswell
int64_t ThreadCostModel::per_unit_compute_num_ = 1;           // 1 : per unit compute num

float ThreadCostModel::thread_startup_cost_ = 100000.0f;  // 100000 : thread startup inherent cost
float ThreadCostModel::single_thread_cost_ = 100000.0f;   // 100000 : Minimum cost of single-threaded
float ThreadCostModel::parallel_thread_cost_ = 40000.0f;  // 40000 : Minimum cost of per thread in parallel-thread

int ThreadCostModel::GetOptimalThreadNum(const ThreadCostContext *thread_cost_context, const int thread_num) {
  const int64_t max_oversharding_factor = 4;

  int64_t block_size = MSVALID(max_oversharding_factor * thread_num, ThreadBlockSize(thread_cost_context),
                               thread_cost_context->total_unit_num_);
  int64_t block_count = UP_DIV(thread_cost_context->total_unit_num_, block_size);
  // the maximum block size should be 2 times of the regular block size.
  int64_t max_block_size = MSMIN(thread_cost_context->total_unit_num_, 2 * block_size);
  double max_efficiency = static_cast<double>(block_count) / (UP_DIV(block_count, thread_num) * thread_num);
  for (int64_t prev_block_count = block_count; max_efficiency < 1.0 && prev_block_count > 1;) {
    int64_t cur_block_size = UP_DIV(thread_cost_context->total_unit_num_, prev_block_count - 1);
    if (cur_block_size > max_block_size) {
      break;
    }
    const int64_t cur_block_count = UP_DIV(thread_cost_context->total_unit_num_, cur_block_size);
    MS_ASSERT(cur_block_count < prev_block_count);
    prev_block_count = cur_block_count;
    const double cur_efficiency =
      static_cast<double>(cur_block_count) / (UP_DIV(cur_block_count, thread_num) * thread_num);
    if (cur_efficiency + 0.01 >= max_efficiency) {  // update threshold : 0.01
      block_size = cur_block_size;
      block_count = cur_block_count;
      if (max_efficiency < cur_efficiency) {
        max_efficiency = cur_efficiency;
      }
    }
  }
  return block_count;
}

int ThreadNumUpdateStrategy(const ThreadCostContext *thread_cost_context, int task_num) {
  if (task_num <= 1) {
    return task_num;
  }

  if (thread_cost_context != nullptr) {
    if (ThreadCostModel::ThreadNum(thread_cost_context) <= 1) {
      return 1;
    }
    int opt_thread = static_cast<int>(ThreadCostModel::ParallelDegree(thread_cost_context));
    task_num = MSVALID(1, opt_thread, task_num);
    task_num = MSMIN(task_num, thread_cost_context->total_unit_num_);
  }
  return task_num;
}

int UpdateThreadNum(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num, int64_t unit_num,
                    int thread_num) {
  if (kernel_compute_cost_map_.count(kernel_type) > 0) {
    lite::ThreadCostContext thread_cost_context;
    thread_cost_context.per_unit_compute_cost_ = kernel_compute_cost_map_.at(kernel_type);
    thread_cost_context.per_unit_load_num_ = per_unit_load_num;
    thread_cost_context.per_unit_store_num_ = per_unit_store_num;
    thread_cost_context.total_unit_num_ = unit_num;
    return ThreadNumUpdateStrategy(&thread_cost_context, thread_num);
  }
  return thread_num;
}
}  // namespace mindspore::lite
