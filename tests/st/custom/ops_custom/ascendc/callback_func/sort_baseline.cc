#include <vector>
#include "acl/acl_base.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

extern "C" int SortGetWorkSpaceSize(void *func_ptr, std::vector<void *> inputs, std::vector<void *> outputs,
                                    uint64_t *workspace_size, aclOpExecutor **executor) {
  using FuncType = int (*)(aclTensor *, int64_t, bool, bool, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
  auto func = reinterpret_cast<FuncType>(func_ptr);
  aclTensor *input0 = static_cast<aclTensor *>(inputs[0]);
  int64_t *input1 = static_cast<int64_t *>(inputs[1]);
  bool *input2 = static_cast<bool *>(inputs[2]);
  bool *input3 = static_cast<bool *>(inputs[3]);
  aclTensor *output0 = static_cast<aclTensor *>(outputs[0]);
  aclTensor *output1 = static_cast<aclTensor *>(outputs[1]);
  return func(input0, *input1, *input2, *input3, output0, output1, workspace_size, executor);
}
