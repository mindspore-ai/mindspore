#### classify all .h .c .cc files to FILE_SET
set(CODER_SRC
        ${MICRO_DIR}/coder/coder.cc
        ${MICRO_DIR}/coder/context.cc
        ${MICRO_DIR}/coder/graph.cc
        ${MICRO_DIR}/coder/session.cc
        ${MICRO_DIR}/coder/shape_info_container.cc
        ${MICRO_DIR}/coder/dynamic_mem_manager.cc
        ${MICRO_DIR}/coder/utils/coder_utils.cc
        ${MICRO_DIR}/coder/utils/dir_utils.cc
        ${MICRO_DIR}/coder/utils/train_utils.cc
        ${MICRO_DIR}/coder/utils/type_cast.cc
        )

set(CODER_SRC ${CODER_SRC}
        ${MICRO_DIR}/coder/train/train_session.cc
        ${MICRO_DIR}/coder/train/train_generator.cc
        )

set(CODER_ALLOCATOR_SRC
        ${MICRO_DIR}/coder/allocator/allocator.cc
        ${MICRO_DIR}/coder/allocator/memory_manager.cc
        )

file(GLOB CODER_GENERATOR_SRC
        ${MICRO_DIR}/coder/generator/*.cc
        ${MICRO_DIR}/coder/generator/inference/*.cc
        ${MICRO_DIR}/coder/generator/component/*.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/*.cc
        )

file(GLOB CODER_OPCODERS_SRC
        ${MICRO_DIR}/coder/wrapper/int8/*.c
        ${MICRO_DIR}/coder/opcoders/*.cc
        #### serializer
        ${MICRO_DIR}/coder/opcoders/serializers/nnacl_serializer/*.cc
        #### base coder
        ${MICRO_DIR}/coder/opcoders/base/*.cc
        #### cmsis int8 coder
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/*.cc
        #### nnacl fp16 coder
        ${MICRO_DIR}/coder/opcoders/nnacl/fp16/*.cc
        #### nnacl fp32 coder
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/*.cc
        #### nnacl fp32_grad coder
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32_grad/*.cc
        #### nnacl int8 coder
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/*.cc
        #### nnacl dequant coder
        ${MICRO_DIR}/coder/opcoders/nnacl/dequant/*.cc
        #### custom
        ${MICRO_DIR}/coder/opcoders/custom/*.cc
        )

set(REGISTRY_SRC
        ${MICRO_DIR}/coder/opcoders/kernel_registry.cc
        )

list(APPEND FILE_SET ${CODER_SRC} ${CODER_OPCODERS_SRC} ${CODER_GENERATOR_SRC}
        ${CODER_ALLOCATOR_SRC} ${LITE_SRC} ${REGISTRY_SRC})
