#ifndef GRID_VECTOR_TYPES_PERM
#define GRID_VECTOR_TYPES_PERM

#include <cassert>
#include "../Vector_types_base.h"

// Tried to do this with C++ functions, but performance drop with nc++...
#define SPLIT_ROTATE_MACRO(out, in, vl, s, r)                                                   \
    static_assert(sizeof(int) == sizeof(int32_t), "Implementation assumes int to be 32bit");    \
                                                                                                \
    const static uint16_t table[] = { 0, 1, 2, 5, 3, 9, 6, 11, 15, 4, 8, 10, 14, 7, 13, 12 };   \
    const static uint16_t de_bruijn = 2479;                                                     \
    const static uint16_t max16bit = 65535;                                                     \
                                                                                                \
    uint16_t logs = table[ uint16_t(s * de_bruijn) >> 12 ];                                     \
    uint16_t w = vl >> logs;                                                                    \
                                                                                                \
    uint16_t logw = table[ uint16_t(w * de_bruijn) >> 12 ];                                     \
    uint16_t mask = max16bit << logw;                                                           \
                                                                                                \
    VECTOR_FOR(i, vl, 1)                                                                        \
    {                                                                                           \
        out[i] = in[(i+r) - ((i+r) & mask) + (i & mask)];                                       \
    }
    
namespace Grid 
{
    namespace Opt = Grid::Optimization;
    
    // PERM
    // ====
    
    // perm is defined in Cartesian_base.h (ExtendedPermuteType)
    template <class float_t>
    inline void permute(Grid_simd<float_t, Opt::vec<float_t>> &ret,
                        const Grid_simd<float_t, Opt::vec<float_t>> &b, 
                        int perm)
    {
        auto a = reinterpret_cast<const uint16_t *>(&perm);
        auto nrot   = a[0];
        auto nsplit = a[1];
        
        SPLIT_ROTATE_MACRO(ret.v.v, b.v.v, Opt::W<float_t>::r, nsplit, nrot);       
    }
    
    template <class float_t>
    inline void permute(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &ret, 
                        const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &b, 
                        int perm)
    {
        auto a = reinterpret_cast<const uint16_t *>(&perm);
        auto nrot   = 2*a[0];
        auto nsplit =   a[1];
        
        SPLIT_ROTATE_MACRO(ret.v.v, b.v.v, Opt::W<float_t>::r, nsplit, nrot);     
    }
    
    // ROTATE
    // ======
    
    /* not yet implemented */
    
    // SPLIT_ROTATE
    // ============
    
    template <class float_t>
    inline void splitRotate(Grid_simd<float_t, Opt::vec<float_t>> &ret,
                            const Grid_simd<float_t, Opt::vec<float_t>> &b, 
                            int nrot, int nsplit)
    {
        SPLIT_ROTATE_MACRO(ret.v.v, b.v.v, Opt::W<float_t>::r, nsplit, nrot);     
    }

    template <class float_t>
    inline void splitRotate(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &ret, 
                            const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &b, 
                            int nrot, int nsplit)
    {
        SPLIT_ROTATE_MACRO(ret.v.v, b.v.v, Opt::W<float_t>::r, nsplit, 2*nrot);  
    }
}

#undef SPLIT_ROTATE_MACRO

#endif
