#include <Grid/Grid.h>

using namespace Grid;
using namespace Grid::QCD;

template<class Lattice_type>
void test(const Lattice_type &lat, GridCartesian &grid)
{ 
    int complex_factor;
    bool is_totally_correct = true;
    
    if(std::is_same<typename Lattice_type::scalar_type, std::complex<double>>::value)
    {
        std::cout << "Testing Complex data" << std::endl;
        complex_factor = 2;
    }
    else if(std::is_same<typename Lattice_type::scalar_type, double>::value)
    {
        std::cout << "Testing Real data" << std::endl;
        complex_factor = 1;
    }
    else
    {
        std::cout << "No valid type, stop test" << std::endl;
        return;
    }
    
    for(auto split : grid._split)
    {
        for(auto rot : grid._rotate)
        {
            bool is_correct = true;
            
            Lattice_type test(&grid);
            Lattice_type ref(&grid);
            
            auto w = Optimization::W<double>::r / split;
            
            auto nrot = complex_factor * rot;
            
            for(std::size_t i=0; i < lat._grid->oSites(); ++i)
            {
                splitRotate(test._odata[i], lat._odata[i], rot, split);
                
                for(std::size_t j=0; j < lat._odata[i]._internal.Nsimd(); ++j)
                {
                    ref._odata[i]._internal.v.v[j] = lat._odata[i]._internal.v.v[(j+nrot) % w + (j/w)*w];
                }
                
                for(std::size_t j=0; j < lat._odata[i]._internal.Nsimd(); ++j)
                {
                    if( ref._odata[i]._internal.v.v[j] != test._odata[i]._internal.v.v[j] )
                    {
                        is_correct = false;
                        is_totally_correct = false;
//                         if(i == 0)
//                             std::cout << "DATA["<<i<<"], INTERNAL["<<j<<"]\tSOURCE:"<<lat._odata[i]._internal.v.v[j]<<"\tREF: "<<ref._odata[i]._internal.v.v[j]<<"\tTEST:"<<test._odata[i]._internal.v.v[j]<<std::endl;
                    }
                }
            }
            
            if(is_correct)
                std::cout << "CORRECT for s=" << split << ", r=" << rot << std::endl;
            else
            {
                std::cout << "ERROR for s=" << split << ", r=" << rot << std::endl;
            }
        }
    }
    
    if(!is_totally_correct)
    {
        std::cout << "Did not pass test. Exit." << std::endl;
        std::exit(1);
    }
    
    std::cout << std::endl;
}

int main (int argc, char ** argv)
{
    Grid_init(&argc,&argv);

    std::vector<int> latt_size   = GridDefaultLatt();
    std::vector<int> simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
    std::vector<int> mpi_layout  = GridDefaultMpi();

    double volume = latt_size[0]*latt_size[1]*latt_size[2]*latt_size[3];
        
    GridCartesian         grid(latt_size,simd_layout,mpi_layout);
    GridParallelRNG       rng(&grid);
    
    std::vector<int> seeds({1,2,3,4});
    rng.SeedFixedIntegers(seeds);
    
    Lattice<iScalar<vRealD>> real_lat(&grid);
    Lattice<iScalar<vComplexD>> complex_lat(&grid);
    
    random(rng, real_lat);
    random(rng, complex_lat);
    
    test(real_lat, grid);
    test(complex_lat, grid);
    
    std::cout << "Successfully passed all tests!" << std::endl;
    
    return 0;
}
                
    
  
  
  
