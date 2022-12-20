#ifndef GE_OP_MATMULTIK_H
#define GE_OP_MATMULTIK_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(MatmulTik)
  .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OP_END_FACTORY_REG(MatmulTik)
}

#endif  // GE_OP_MATMULTIK_H