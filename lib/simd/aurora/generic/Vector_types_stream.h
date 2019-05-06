#ifndef GRID_VECTOR_TYPES_STREAM
#define GRID_VECTOR_TYPES_STREAM

#include <cassert>
#include "../Vector_types_base.h" 

namespace Grid 
{
    namespace Opt = Grid::Optimization;
    
    template <class num_t, class float_t>
    inline void vstream(Grid_simd<num_t, Opt::vec<float_t>> &out, const Grid_simd<num_t, Opt::vec<float_t>> &in) 
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            out.v.v[i] = in.v.v[i];
        }
    }
}

#endif
