#ifndef GRID_VECTOR_TYPES_NEW
#define GRID_VECTOR_TYPES_NEW

#include "Grid_vector_types_base.h"

#ifdef __clang__
    #include "Grid_vector_types_arith_clang.h"
#else
    #include "Grid_vector_types_arith_gen.h"
#endif

#endif
