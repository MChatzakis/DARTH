#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "faiss" for configuration ""
set_property(TARGET faiss APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(faiss PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libfaiss.so"
  IMPORTED_SONAME_NOCONFIG "libfaiss.so"
  )

list(APPEND _cmake_import_check_targets faiss )
list(APPEND _cmake_import_check_files_for_faiss "${_IMPORT_PREFIX}/lib/libfaiss.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
