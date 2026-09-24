target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_rmsnorm_quantize.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/rmsnorm_swiglu_ffn.cpp"
)
