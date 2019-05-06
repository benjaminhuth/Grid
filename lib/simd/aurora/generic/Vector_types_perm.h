#ifndef GRID_VECTOR_TYPES_PERM
#define GRID_VECTOR_TYPES_PERM

#include <cassert>
#include "../Vector_types_base.h"

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
        const static uint16_t table[] = { 0, 1, 2, 5, 3, 9, 6, 11, 15, 4, 8, 10, 14, 7, 13, 12 };
        const static uint16_t de_bruijn = 2479;
        
        auto a = reinterpret_cast<const uint16_t *>(&perm);
        auto nrot = a[0];
        auto nsplit = a[1];
        
        int logs = table[ uint16_t(nsplit * de_bruijn) >> 12 ];
        
        int w = Opt::W<float_t>::r >> logs;
        int logw = table[ uint16_t(w * de_bruijn) >> 12 ];
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret.v.v[i] = b.v.v[(i+nrot) % w + ((nrot >> logw) << logw)];
        }        
    }
    
    template <class float_t>
    inline void permute(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &ret, 
                        const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &b, 
                        int perm)
    {
        const static uint16_t table[] = { 0, 1, 2, 5, 3, 9, 6, 11, 15, 4, 8, 10, 14, 7, 13, 12 };
        const static uint16_t de_bruijn = 2479;
        
        auto a = reinterpret_cast<const uint16_t *>(&perm);
        auto nrot = 2*a[0];
        auto nsplit = a[1];
        
        int logs = table[ uint16_t(nsplit * de_bruijn) >> 12 ];
        
        int w = Opt::W<float_t>::r >> logs;
        int logw = table[ uint16_t(w * de_bruijn) >> 12 ];
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret.v.v[i] = b.v.v[(i+nrot) % w + ((nrot >> logw) << logw)];
        }
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
        const static uint16_t table[] = { 0, 1, 2, 5, 3, 9, 6, 11, 15, 4, 8, 10, 14, 7, 13, 12 };
        const static uint16_t de_bruijn = 2479;
        
        int logs = table[ uint16_t(nsplit * de_bruijn) >> 12 ];
        
        int w = Opt::W<float_t>::r >> logs;
        int logw = table[ uint16_t(w * de_bruijn) >> 12 ];
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret.v.v[i] = b.v.v[(i+nrot) % w + ((nrot >> logw) << logw)];
        }
    }

    template <class float_t>
    inline void splitRotate(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &ret, 
                            const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &b, 
                            int nrot, int nsplit)
    {
        const static uint16_t table[] = { 0, 1, 2, 5, 3, 9, 6, 11, 15, 4, 8, 10, 14, 7, 13, 12 };
        const static uint16_t de_bruijn = 2479;
        
        int logs = table[ uint16_t(nsplit * de_bruijn) >> 12 ];
        
        int w = Opt::W<float_t>::r >> logs;
        int logw = table[ uint16_t(w * de_bruijn) >> 12 ];
        
        nrot *= 2;
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret.v.v[i] = b.v.v[(i + nrot) % w + ((nrot >> logw) << logw)];
        }
    }
}

#endif
