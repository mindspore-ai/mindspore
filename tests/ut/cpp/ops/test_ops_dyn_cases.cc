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
auto EltwiseDynTestCase_Float16 =
  testing::Values(EltwiseOpParams{{10}, kFloat16, {10}, kFloat16},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kFloat16, {20, 30}, kFloat16},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kFloat16, {6, 7, 8}, kFloat16},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kFloat16, {2, 3, 4, 5}, kFloat16},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kFloat16, {-1, -1}, kFloat16},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kFloat16, {-1, -1, -1}, kFloat16},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kFloat16, {2, -1}, kFloat16},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kFloat16, {-1, -1, 2, -1}, kFloat16}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kFloat16, {-2}, kFloat16}                        /* dynamic rank */
  );

auto EltwiseDynTestCase_Float32 =
  testing::Values(EltwiseOpParams{{10}, kFloat32, {10}, kFloat32},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kFloat32, {20, 30}, kFloat32},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kFloat32, {6, 7, 8}, kFloat32},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kFloat32, {2, 3, 4, 5}, kFloat32},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kFloat32, {-1, -1}, kFloat32},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kFloat32, {-1, -1, -1}, kFloat32},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kFloat32, {2, -1}, kFloat32},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kFloat32, {-1, -1, 2, -1}, kFloat32}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kFloat32, {-2}, kFloat32}                        /* dynamic rank */
  );

auto EltwiseDynTestCase_Float64 =
  testing::Values(EltwiseOpParams{{10}, kFloat64, {10}, kFloat64},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kFloat64, {20, 30}, kFloat64},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kFloat64, {6, 7, 8}, kFloat64},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kFloat64, {2, 3, 4, 5}, kFloat64},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kFloat64, {-1, -1}, kFloat64},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kFloat64, {-1, -1, -1}, kFloat64},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kFloat64, {2, -1}, kFloat64},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kFloat64, {-1, -1, 2, -1}, kFloat64}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kFloat64, {-2}, kFloat64}                        /* dynamic rank */
  );

auto EltwiseDynTestCase_Int8 =
  testing::Values(EltwiseOpParams{{10}, kInt8, {10}, kInt8},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kInt8, {20, 30}, kInt8},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kInt8, {6, 7, 8}, kInt8},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kInt8, {2, 3, 4, 5}, kInt8},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kInt8, {-1, -1}, kInt8},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kInt8, {-1, -1, -1}, kInt8},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kInt8, {2, -1}, kInt8},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kInt8, {-1, -1, 2, -1}, kInt8}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kInt8, {-2}, kInt8}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_Int16 =
  testing::Values(EltwiseOpParams{{10}, kInt16, {10}, kInt16},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kInt16, {20, 30}, kInt16},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kInt16, {6, 7, 8}, kInt16},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kInt16, {2, 3, 4, 5}, kInt16},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kInt16, {-1, -1}, kInt16},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kInt16, {-1, -1, -1}, kInt16},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kInt16, {2, -1}, kInt16},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kInt16, {-1, -1, 2, -1}, kInt16}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kInt16, {-2}, kInt16}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_Int32 =
  testing::Values(EltwiseOpParams{{10}, kInt32, {10}, kInt32},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kInt32, {20, 30}, kInt32},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kInt32, {6, 7, 8}, kInt32},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kInt32, {2, 3, 4, 5}, kInt32},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kInt32, {-1, -1}, kInt32},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kInt32, {-1, -1, -1}, kInt32},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kInt32, {2, -1}, kInt32},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kInt32, {-1, -1, 2, -1}, kInt32}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kInt32, {-2}, kInt32}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_Int64 =
  testing::Values(EltwiseOpParams{{10}, kInt64, {10}, kInt64},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kInt64, {20, 30}, kInt64},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kInt64, {6, 7, 8}, kInt64},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kInt64, {2, 3, 4, 5}, kInt64},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kInt64, {-1, -1}, kInt64},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kInt64, {-1, -1, -1}, kInt64},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kInt64, {2, -1}, kInt64},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kInt64, {-1, -1, 2, -1}, kInt64}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kInt64, {-2}, kInt64}                        /* dynamic rank */
  );

auto EltwiseDynTestCase_UInt8 =
  testing::Values(EltwiseOpParams{{10}, kUInt8, {10}, kUInt8},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kUInt8, {20, 30}, kUInt8},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kUInt8, {6, 7, 8}, kUInt8},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kUInt8, {2, 3, 4, 5}, kUInt8},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kUInt8, {-1, -1}, kUInt8},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kUInt8, {-1, -1, -1}, kUInt8},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kUInt8, {2, -1}, kUInt8},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kUInt8, {-1, -1, 2, -1}, kUInt8}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kUInt8, {-2}, kUInt8}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_UInt16 =
  testing::Values(EltwiseOpParams{{10}, kUInt16, {10}, kUInt16},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kUInt16, {20, 30}, kUInt16},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kUInt16, {6, 7, 8}, kUInt16},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kUInt16, {2, 3, 4, 5}, kUInt16},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kUInt16, {-1, -1}, kUInt16},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kUInt16, {-1, -1, -1}, kUInt16},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kUInt16, {2, -1}, kUInt16},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kUInt16, {-1, -1, 2, -1}, kUInt16}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kUInt16, {-2}, kUInt16}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_UInt32 =
  testing::Values(EltwiseOpParams{{10}, kUInt32, {10}, kUInt32},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kUInt32, {20, 30}, kUInt32},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kUInt32, {6, 7, 8}, kUInt32},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kUInt32, {2, 3, 4, 5}, kUInt32},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kUInt32, {-1, -1}, kUInt32},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kUInt32, {-1, -1, -1}, kUInt32},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kUInt32, {2, -1}, kUInt32},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kUInt32, {-1, -1, 2, -1}, kUInt32}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kUInt32, {-2}, kUInt32}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_UInt64 =
  testing::Values(EltwiseOpParams{{10}, kUInt64, {10}, kUInt64},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kUInt64, {20, 30}, kUInt64},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kUInt64, {6, 7, 8}, kUInt64},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kUInt64, {2, 3, 4, 5}, kUInt64},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kUInt64, {-1, -1}, kUInt64},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kUInt64, {-1, -1, -1}, kUInt64},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kUInt64, {2, -1}, kUInt64},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kUInt64, {-1, -1, 2, -1}, kUInt64}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kUInt64, {-2}, kUInt64}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_Complex64 =
  testing::Values(EltwiseOpParams{{10}, kComplex64, {10}, kComplex64},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kComplex64, {20, 30}, kComplex64},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kComplex64, {6, 7, 8}, kComplex64},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kComplex64, {2, 3, 4, 5}, kComplex64},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kComplex64, {-1, -1}, kComplex64},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kComplex64, {-1, -1, -1}, kComplex64},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kComplex64, {2, -1}, kComplex64},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kComplex64, {-1, -1, 2, -1}, kComplex64}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kComplex64, {-2}, kComplex64}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_Complex128 =
  testing::Values(EltwiseOpParams{{10}, kComplex128, {10}, kComplex128},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kComplex128, {20, 30}, kComplex128},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kComplex128, {6, 7, 8}, kComplex128},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kComplex128, {2, 3, 4, 5}, kComplex128},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kComplex128, {-1, -1}, kComplex128},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kComplex128, {-1, -1, -1}, kComplex128},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kComplex128, {2, -1}, kComplex128},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kComplex128, {-1, -1, 2, -1}, kComplex128}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kComplex128, {-2}, kComplex128}                        /* dynamic rank */
  );
auto EltwiseDynTestCase_Bool =
  testing::Values(EltwiseOpParams{{10}, kBool, {10}, kBool},                       /* 1 dims */
                  EltwiseOpParams{{20, 30}, kBool, {20, 30}, kBool},               /* 2 dims */
                  EltwiseOpParams{{6, 7, 8}, kBool, {6, 7, 8}, kBool},             /* 3 dims */
                  EltwiseOpParams{{2, 3, 4, 5}, kBool, {2, 3, 4, 5}, kBool},       /* 4 dims */
                  EltwiseOpParams{{-1, -1}, kBool, {-1, -1}, kBool},               /* dynamic shape */
                  EltwiseOpParams{{-1, -1, -1}, kBool, {-1, -1, -1}, kBool},       /* dynamic shape */
                  EltwiseOpParams{{2, -1}, kBool, {2, -1}, kBool},                 /* dynamic shape */
                  EltwiseOpParams{{-1, -1, 2, -1}, kBool, {-1, -1, 2, -1}, kBool}, /* dynamic shape */
                  EltwiseOpParams{{-2}, kBool, {-2}, kBool}                        /* dynamic rank */
  );
}  // namespace mindspore::ops
