# Fallback find module for MLX when CONFIG mode fails.
# Typically not needed when MLX is installed via Homebrew.

find_path(MLX_INCLUDE_DIRS mlx/mlx.h
  HINTS
    /opt/homebrew/include
    /usr/local/include
    $ENV{MLX_DIR}/include
)

find_library(MLX_LIBRARY mlx
  HINTS
    /opt/homebrew/lib
    /usr/local/lib
    $ENV{MLX_DIR}/lib
)

if(MLX_INCLUDE_DIRS AND MLX_LIBRARY)
  set(MLX_FOUND TRUE)
  if(NOT TARGET mlx)
    add_library(mlx SHARED IMPORTED)
    set_target_properties(mlx PROPERTIES
      IMPORTED_LOCATION ${MLX_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${MLX_INCLUDE_DIRS}
    )
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLX DEFAULT_MSG MLX_LIBRARY MLX_INCLUDE_DIRS)
