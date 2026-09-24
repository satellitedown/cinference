target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/qk_rmsnorm_rope.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/qk_rmsnorm_rope.cpp"
)
