
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SubCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, formerNum); // 添加tiling字段，分配到较多数据量的核心数，即大块
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);   // 添加tiling字段，分配到较少数据量的核心数，即小块
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);  // 添加tiling字段，大块的长度
  TILING_DATA_FIELD_DEF(uint32_t, tailLength); // 添加tiling字段，小块的长度
  TILING_DATA_FIELD_DEF(uint32_t, alignNum); // 添加tiling字段，需要对齐到的最小数据量
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SubCustom, SubCustomTilingData)
}
