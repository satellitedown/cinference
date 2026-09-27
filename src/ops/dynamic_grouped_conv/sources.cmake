# Modified by satellitedown for Cinference: build the shared finish and the Q4 projection route.
# See NOTICE and upstream-provenance.json for upstream attribution.

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv_add_finish.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/q4_dynamic_grouped_conv_add.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_dynamic_grouped_conv_add_materialized.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_dynamic_grouped_conv_add_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/dynamic_grouped_conv.cpp"
)
