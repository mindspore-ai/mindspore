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

#include "ir/dtype/number.h"
#include "ops/test_ops.h"

namespace mindspore::ops {
auto EltwiseDynShapeTestCases =
  testing::Values(EltwiseOpShapeParams{{10}, {10}},                         /* 1 dims */
                  EltwiseOpShapeParams{{20, 30}, {20, 30}},                 /* 2 dims */
                  EltwiseOpShapeParams{{6, 7, 8}, {6, 7, 8}},               /* 3 dims */
                  EltwiseOpShapeParams{{2, 3, 4, 5}, {2, 3, 4, 5}},         /* 4 dims */
                  EltwiseOpShapeParams{{-1, -1}, {-1, -1}},                 /* dynamic shape */
                  EltwiseOpShapeParams{{-1, -1, -1}, {-1, -1, -1}},         /* dynamic shape */
                  EltwiseOpShapeParams{{2, -1}, {2, -1}},                   /* dynamic shape */
                  EltwiseOpShapeParams{{2, -1, 4, 5, 6}, {2, -1, 4, 5, 6}}, /* dynamic shape */
                  EltwiseOpShapeParams{{-1, -1, 2, -1}, {-1, -1, 2, -1}},   /* dynamic shape */
                  EltwiseOpShapeParams{{-2}, {-2}}                          /* dynamic rank */
  );

auto EltwiseGradDynShapeTestCases =
  testing::Values(EltwiseGradOpShapeParams{{10}, {10}, {10}},                                     /* 1 dims */
                  EltwiseGradOpShapeParams{{20, 30}, {20, 30}, {20, 30}},                         /* 2 dims */
                  EltwiseGradOpShapeParams{{6, 7, 8}, {6, 7, 8}, {6, 7, 8}},                      /* 3 dims */
                  EltwiseGradOpShapeParams{{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},             /* 4 dims */
                  EltwiseGradOpShapeParams{{-1, -1}, {-1, -1}, {-1, -1}},                         /* dynamic shape */
                  EltwiseGradOpShapeParams{{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},             /* dynamic shape */
                  EltwiseGradOpShapeParams{{2, -1}, {2, -1}, {2, -1}},                            /* dynamic shape */
                  EltwiseGradOpShapeParams{{2, -1, 4, 5, 6}, {2, -1, 4, 5, 6}, {2, -1, 4, 5, 6}}, /* dynamic shape */
                  EltwiseGradOpShapeParams{{-1, -1, 2, -1}, {-1, -1, 2, -1}, {-1, -1, 2, -1}},    /* dynamic shape */
                  EltwiseGradOpShapeParams{{-2}, {-2}, {-2}}                                      /* dynamic rank */
  );

auto BroadcastOpShapeScalarTensorCases = testing::Values(
  /* y is number */
  BroadcastOpShapeParams{{10}, {}, {10}}, BroadcastOpShapeParams{{10, 1, 2}, {}, {10, 1, 2}},
  BroadcastOpShapeParams{{10, 4, 2}, {}, {10, 4, 2}}, BroadcastOpShapeParams{{10, 1, -1}, {}, {10, 1, -1}},
  BroadcastOpShapeParams{{-2}, {}, {-2}},
  /* x is number */
  BroadcastOpShapeParams{{}, {10}, {10}}, BroadcastOpShapeParams{{}, {10, 1, 2}, {10, 1, 2}},
  BroadcastOpShapeParams{{}, {10, 4, 2}, {10, 4, 2}}, BroadcastOpShapeParams{{}, {10, 1, -1}, {10, 1, -1}},
  BroadcastOpShapeParams{{}, {-2}, {-2}});

auto BroadcastOpShapeTensorTensorCases = testing::Values(
  /* x and y both tensor */
  BroadcastOpShapeParams{{4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
  BroadcastOpShapeParams{{2, 1, 4, 5, 6, 9}, {9}, {2, 1, 4, 5, 6, 9}},
  BroadcastOpShapeParams{{2, 3, 4, -1}, {2, 3, 4, 5}, {2, 3, 4, 5}},
  BroadcastOpShapeParams{{2, 3, 4, -1}, {-1, -1, 4, 5}, {2, 3, 4, 5}},
  BroadcastOpShapeParams{{2, 1, 4, -1}, {-1, -1, 4, 5}, {2, -1, 4, 5}},
  BroadcastOpShapeParams{{2, 1, 4, 5, 6, 9}, {-2}, {-2}}, BroadcastOpShapeParams{{2, 1, 4, 5, -1, 9}, {-2}, {-2}},
  BroadcastOpShapeParams{{-2}, {2, 1, 4, 5, 6, 9}, {-2}}, BroadcastOpShapeParams{{-2}, {2, 1, 4, 5, -1, 9}, {-2}},
  BroadcastOpShapeParams{{-2}, {-2}, {-2}});
}  // namespace mindspore::ops
