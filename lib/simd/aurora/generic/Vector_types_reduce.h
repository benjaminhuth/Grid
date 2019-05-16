#ifndef GRID_VECTOR_TYPES_REDUCE
#define GRID_VECTOR_TYPES_REDUCE

#include <cassert>
#include "../Vector_types_base.h" 

namespace Grid 
{
    namespace Opt = Grid::Optimization;
    
    template <class float_t>
    inline float_t Reduce(const Grid_simd<float_t, Opt::vec<float_t>> &in) 
    {
        float_t ret{0.0};
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret += in.v.v[i];
        }
        
        return ret;
    }
        
    template <class float_t>
    inline std::complex<float_t> Reduce(const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &in) 
    {
        float_t re{0.0};
        float_t im{0.0};
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            re += in.v.v[i];
            im += in.v.v[i+1];
        }
        
        return std::complex<float_t>(re, im);
    }
}

#endif
