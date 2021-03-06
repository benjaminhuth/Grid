#ifndef GRID_VECTOR_TYPES_ARITH
#define GRID_VECTOR_TYPES_ARITH

#include <cassert>
#include "../Vector_types_base.h"

namespace Grid 
{
    namespace Opt = Grid::Optimization;
    
    // MULT
    // ====
    
    // V = V * V
    template <class float_t>
    inline void mult(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = lhs->v.v[i] * rhs->v.v[i];
        }
    }
    
    template <class float_t>
    inline Grid_simd<float_t, Opt::vec<float_t>> operator*(const Grid_simd<float_t, Opt::vec<float_t>> &lhs, 
                                                           const Grid_simd<float_t, Opt::vec<float_t>> &rhs)
    {
        Grid_simd<float_t, Opt::vec<float_t>> ret;
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret.v.v[i] = lhs.v.v[i] * rhs.v.v[i];
        }
        
        return ret;
    };
    
    template <class float_t>
    inline void mult(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs) 
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = lhs->v.v[i] * rhs->v.v[i]   - lhs->v.v[i+1] * rhs->v.v[i+1];
            ret->v.v[i+1] = lhs->v.v[i] * rhs->v.v[i+1] + lhs->v.v[i+1] * rhs->v.v[i];
        }
    }
    
    template <class float_t>
    inline Grid_simd<std::complex<float_t>, Opt::vec<float_t>> operator*(const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &lhs, 
                                                                         const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &rhs)
    {
        Grid_simd<std::complex<float_t>, Opt::vec<float_t>> ret;
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret.v.v[i]   = lhs.v.v[i] * rhs.v.v[i]   - lhs.v.v[i+1] * rhs.v.v[i+1];
            ret.v.v[i+1] = lhs.v.v[i] * rhs.v.v[i+1] + lhs.v.v[i+1] * rhs.v.v[i];
        }
        
        return ret;
    };
    
    
    // V = S * V
    template <class float_t>
    inline void mult(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const float_t *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = *lhs * rhs->v.v[i];
        }
    }
    
    template <class float_t>
    inline void mult(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const std::complex<float_t> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs) 
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = reinterpret_cast<const float_t(&)[2]>( *lhs)[0] * rhs->v.v[i]   - reinterpret_cast<const float_t(&)[2]>( *lhs)[1] * rhs->v.v[i+1];
            ret->v.v[i+1] = reinterpret_cast<const float_t(&)[2]>( *lhs)[0] * rhs->v.v[i+1] + reinterpret_cast<const float_t(&)[2]>( *lhs)[1] * rhs->v.v[i];
        }
    }
    
    
    // V = V * S
    template <class float_t>
    inline void mult(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs,
                     const float_t *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = *lhs * rhs->v.v[i];
        }
    }
    
    template <class float_t>
    inline void mult(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs,
                     const std::complex<float_t> *__restrict__ rhs) 
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = lhs->v.v[i] * reinterpret_cast<const float_t(&)[2]>( *rhs)[0] - lhs->v.v[i+1] * reinterpret_cast<const float_t(&)[2]>( *rhs)[1];
            ret->v.v[i+1] = lhs->v.v[i] * reinterpret_cast<const float_t(&)[2]>( *rhs)[1] + lhs->v.v[i+1] * reinterpret_cast<const float_t(&)[2]>( *rhs)[0];
        }
    }
    
    // MAC
    // ===
    
    // V = V * V
    template <class float_t>
    inline void mac(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = ret->v.v[i] + lhs->v.v[i] * rhs->v.v[i];
        }
    };
    
    
    template <class float_t>
    inline void mac(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = ret->v.v[i]   + lhs->v.v[i] * rhs->v.v[i]   - lhs->v.v[i+1] * rhs->v.v[i+1];
            ret->v.v[i+1] = ret->v.v[i+1] + lhs->v.v[i] * rhs->v.v[i+1] + lhs->v.v[i+1] * rhs->v.v[i];
        }
    };
    
    // V = S * V
    template <class float_t>
    inline void mac(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const float_t *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = ret->v.v[i] + *lhs * rhs->v.v[i];
        }
    }
    
    template <class float_t>
    inline void mac(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const std::complex<float_t> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs) 
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = ret->v.v[i]   + reinterpret_cast<const float_t(&)[2]>( *lhs)[0] * rhs->v.v[i]   - reinterpret_cast<const float_t(&)[2]>( *lhs)[1] * rhs->v.v[i+1];
            ret->v.v[i+1] = ret->v.v[i+1] + reinterpret_cast<const float_t(&)[2]>( *lhs)[0] * rhs->v.v[i+1] + reinterpret_cast<const float_t(&)[2]>( *lhs)[1] * rhs->v.v[i];
        }
    }
    
    // V = V * S
    template <class float_t>
    inline void mac(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs,
                     const float_t *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = *lhs * rhs->v.v[i];
        }
    }
    
    template <class float_t>
    inline void mac(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs,
                     const std::complex<float_t> *__restrict__ rhs) 
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = ret->v.v[i]   + lhs->v.v[i] * reinterpret_cast<const float_t(&)[2]>( *rhs)[0] - lhs->v.v[i+1] * reinterpret_cast<const float_t(&)[2]>( *rhs)[1];
            ret->v.v[i+1] = ret->v.v[i+1] + lhs->v.v[i] * reinterpret_cast<const float_t(&)[2]>( *rhs)[1] + lhs->v.v[i+1] * reinterpret_cast<const float_t(&)[2]>( *rhs)[0];
        }
    }
    
    // ADD
    // ===
    
    template <class num_t, class float_t>
    inline void add(Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = lhs->v.v[i] + rhs->v.v[i];
        }
    }
    
    template <class num_t, class float_t>
    inline Grid_simd<num_t, Opt::vec<float_t>> operator+(const Grid_simd<num_t, Opt::vec<float_t>> &lhs, 
                                                         const Grid_simd<num_t, Opt::vec<float_t>> &rhs)
    {
        Grid_simd<num_t, Opt::vec<float_t>> ret;
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret.v.v[i] = lhs.v.v[i] + rhs.v.v[i];
        }
        
        return ret;
    };
    
    template <class float_t>
    inline void add(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const float_t *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = *lhs + rhs->v.v[i];
        }
    }
    
    template <class float_t>
    inline void add(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret,
                     const std::complex<float_t> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = reinterpret_cast<const float_t(&)[2]>( *lhs)[0] + rhs->v.v[i];
            ret->v.v[i+1] = reinterpret_cast<const float_t(&)[2]>( *lhs)[1] + rhs->v.v[i+1];
        }
    }

    template <class float_t>
    inline void add(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs,
                     const float_t *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = lhs->v.v[i] + *rhs;
        }
    }
        
    template <class float_t>
    inline void add(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs,
                     const std::complex<float_t> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = lhs->v.v[i]   + reinterpret_cast<const float_t(&)[2]>( *rhs)[0];
            ret->v.v[i+1] = lhs->v.v[i+1] + reinterpret_cast<const float_t(&)[2]>( *rhs)[1];
        }
    }
    
    // SUB
    // ===
    
    template <class num_t, class float_t>
    inline void sub(Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ lhs, 
                     const Grid_simd<num_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = lhs->v.v[i] - rhs->v.v[i];
        }
    }
        
    template <class num_t, class float_t>
    inline Grid_simd<num_t, Opt::vec<float_t>> operator-(const Grid_simd<num_t, Opt::vec<float_t>> &lhs, 
                                                         const Grid_simd<num_t, Opt::vec<float_t>> &rhs)
    {
        Grid_simd<num_t, Opt::vec<float_t>> ret;
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret.v.v[i] = lhs.v.v[i] - rhs.v.v[i];
        }
        
        return ret;
    };
    
    template <class float_t>
    inline void sub(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const float_t *__restrict__ lhs, 
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = *lhs - rhs->v.v[i];
        }
    }
    
    template <class float_t>
    inline void sub(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret,
                     const std::complex<float_t> *__restrict__ lhs, 
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = reinterpret_cast<const float_t(&)[2]>( *lhs)[0] - rhs->v.v[i];
            ret->v.v[i+1] = reinterpret_cast<const float_t(&)[2]>( *lhs)[1] - rhs->v.v[i+1];
        }
    }

    template <class float_t>
    inline void sub(Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<float_t, Opt::vec<float_t>> *__restrict__ lhs,
                     const float_t *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 1)
        {
            ret->v.v[i] = lhs->v.v[i] - *rhs;
        }
    }
        
    template <class float_t>
    inline void sub(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ ret,
                     const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> *__restrict__ lhs,
                     const std::complex<float_t> *__restrict__ rhs)
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret->v.v[i]   = lhs->v.v[i]   - reinterpret_cast<const float_t(&)[2]>( *rhs)[0];
            ret->v.v[i+1] = lhs->v.v[i+1] - reinterpret_cast<const float_t(&)[2]>( *rhs)[1];
        }
    }
    
    // IMAGINARY UNIT
    // ==============
    
    template <class float_t>
    inline void timesMinusI(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &ret, 
                            const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &in) 
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret.v.v[i]   =  in.v.v[i+1];
            ret.v.v[i+1] = -in.v.v[i];
        }
    }
    
    template <class float_t>
    inline Grid_simd<std::complex<float_t>, Opt::vec<float_t>> timesMinusI(const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &in) 
    {
        Grid_simd<std::complex<float_t>, Opt::vec<float_t>> ret;
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret.v.v[i]   =  in.v.v[i+1];
            ret.v.v[i+1] = -in.v.v[i];
        }
        
        return ret;
    }
    
    template <class float_t>
    inline Grid_simd<float_t, Opt::vec<float_t>> timesMinusI(const Grid_simd<float_t, Opt::vec<float_t>> &in) 
    {
        return in;
    }
    
    template <class float_t>
    inline void timesI(Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &ret, 
                            const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &in)  
    {
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret.v.v[i]   = -in.v.v[i+1];
            ret.v.v[i+1] =  in.v.v[i];
        }
    }
    
    template <class float_t>
    inline Grid_simd<std::complex<float_t>, Opt::vec<float_t>> timesI(const Grid_simd<std::complex<float_t>, Opt::vec<float_t>> &in) 
    {
        Grid_simd<std::complex<float_t>, Opt::vec<float_t>> ret;
        
        VECTOR_FOR(i, Opt::W<float_t>::r, 2)
        {
            ret.v.v[i]   = -in.v.v[i+1];
            ret.v.v[i+1] =  in.v.v[i];
        }
        
        return ret;
    }
    
    template <class float_t>
    inline Grid_simd<float_t, Opt::vec<float_t>> timesI(const Grid_simd<float_t, Opt::vec<float_t>> &in) 
    {
        return in;
    }
    
}

#endif
