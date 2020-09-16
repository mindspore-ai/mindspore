#!/bin/bash
echo '{
  "strategy": "additive",
  "featureFilters": {
    "normalization": "include"
  }
}' > filter.json
CFLAGS="-fstack-protector -Wl,-z,now -D_FORTIFY_SOURCE=2 -O2" ./icu4c/source/runConfigureICU Linux --enable-rpath --disable-tests --disable-samples --disable-icuio --disable-extras ICU_DATA_FILTER_FILE=filter.json "$@"
