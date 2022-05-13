/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_PARALLEL_CONFIG_H_
#define MINDSPORE_LITE_SRC_PARALLEL_CONFIG_H_

// whether to enable parallel_executor or not
#define PARALLEL 1

// cut graph in case of split, concat, add, eltwise, or branches. Used in scheduler.cc
#define SUB_GRAPH 1

// whether to omit PReLU in converter or not
#define REMOVE_PRELU 1

// whether to enable count time and data printer in parallel_executor.cc or not
#define PROFILE 0

#define CPU16SUB_INSERT_CAST 0
#endif  // MINDSPORE_LITE_SRC_PARALLEL_CONFIG_H_
