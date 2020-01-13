    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./benchmarks/Benchmark_su3.cc

    Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: Peter Boyle <peterboyle@Peters-MacBook-Pro-2.local>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */
    
#include <Grid/Grid.h>
#include <chrono>

using namespace std;
using namespace Grid;
using namespace Grid::QCD;

int main (int argc, char ** argv)
{
    Grid_init(&argc,&argv);
#define LMAX (32)
#define LMIN (8)
#define LINC (4)
    
//     typedef LatticeColourMatrix field_t;
    typedef LatticeComplex field_t;
    typedef typename field_t::vector_object vobj_t;
    typedef typename vobj_t::scalar_object sobj_t;
    
    std::vector<int> displacements = {-1,1};
    int Nloop=2000;

    std::vector<int> simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
    std::vector<int> mpi_layout  = GridDefaultMpi();	
    std::cout<<GridLogMessage << "Grid uses mpi layout "<<mpi_layout<<std::endl;

    int64_t threads = GridThread::GetThreads();
    std::cout<<GridLogMessage << "Grid is setup to use "<<threads<<" threads"<<std::endl;

    std::vector<int> latt_size = GridDefaultLatt();
    int64_t vol = latt_size[0]*latt_size[1]*latt_size[2]*latt_size[3];
    std::cout<<GridLogMessage << "Grid lattice size    "<<latt_size<<std::endl;
    
    std::vector<int> stencil_dims;
    std::vector<int> stencil_disps;

    for( int dim=0; dim<latt_size.size(); ++dim )
    {
        if( mpi_layout[dim] > 1 )
        {
            for( auto disp : displacements )
            {
                stencil_dims.push_back(dim);
                stencil_disps.push_back(disp);
            }
        }
    }

    if( stencil_dims.empty() )
    {
        for( auto disp : displacements )
        {
            stencil_dims.push_back(0);
            stencil_disps.push_back(disp);
        }
    }
   
    std::cout<<GridLogMessage << std::endl; 
    std::cout<<GridLogMessage << "Benchmark " << stencil_dims.size() << "-point stencil:" << std::endl;
    std::cout<<GridLogMessage << "\tpoint dimensions:    " << stencil_dims << std::endl;
    std::cout<<GridLogMessage << "\tpoint displacements: " << stencil_disps << std::endl;
    std::cout<<GridLogMessage << std::endl;   
    GridCartesian grid(latt_size,simd_layout,mpi_layout);
    GridParallelRNG pRNG(&grid);      
    pRNG.SeedFixedIntegers(std::vector<int>({45,12,81,9}));

    field_t a(&grid); random(pRNG,a);
    field_t b(&grid);

    CartesianStencil<vobj_t,vobj_t> my_stencil(&grid, stencil_dims.size(), 0, stencil_dims, stencil_disps);
    SimpleCompressor<vobj_t> my_compressor;
    my_stencil.ZeroCounters();
        
    double compute_time = 0.0;
        
    auto start = std::chrono::high_resolution_clock::now();
        
    for(int j=0; j<Nloop; ++j)
    {
        my_stencil.HaloExchange(a, my_compressor);
            
        auto t1 = std::chrono::high_resolution_clock::now();
        parallel_for(int i=0; i<b._grid->oSites(); ++i)
        {	                  
            for(int point=0; point<4; ++point)
            {
                int permute_type;
                StencilEntry *SE;
                SE = my_stencil.GetEntry(permute_type,point,i);
            
                if ( SE->_is_local && SE->_permute )
                {
                    std::remove_reference<decltype(b._odata[0])>::type temp;
                    permute(temp,a._odata[SE->_offset],permute_type);
                    b._odata[i] += temp;
                }
                else if (SE->_is_local)
                    b._odata[i] += a._odata[SE->_offset];
                else
                    b._odata[i] += my_stencil.CommBuf()[SE->_offset];
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
            
        compute_time += std::chrono::duration<double>(t2-t1).count();
    }
        
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end-start).count() / Nloop;
    compute_time /= Nloop;

    std::cout<<GridLogMessage << "TIME REPORT:" << std::endl;
    std::cout<<GridLogMessage << " total time    = " << time*1.0e6 << " us" << std::endl;
    std::cout<<GridLogMessage << " compute time  = " << compute_time*1.0e6 << " us\t(" << compute_time / time * 100.0 << "%)" << std::endl;
    std::cout<<GridLogMessage << " exchange time = " << (time-compute_time)*1.0e6 << " us\t(" << (time-compute_time) / time * 100.0 << "%)" << std::endl;
    std::cout<<GridLogMessage << std::endl;
    std::cout<<GridLogMessage << "STENCIL REPORT:" << std::endl;
    my_stencil.Report();
    
    Grid_finalize();
}
