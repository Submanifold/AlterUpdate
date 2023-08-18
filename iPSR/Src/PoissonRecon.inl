/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

/*
This file is edited by Fei Hou and Chiyu Wang, Institute of Software, Chinese Academy of Sciences.
*/

#undef SHOW_WARNINGS							// Display compilation warnings
#undef USE_DOUBLE								// If enabled, double-precesion is used
#undef FAST_COMPILE								// If enabled, only a single version of the reconstruction code is compiled
#undef ARRAY_DEBUG								// If enabled, array access is tested for validity
#define DATA_DEGREE 1							// The order of the B-Spline used to splat in data for color interpolation
												// This can be changed to zero if more interpolatory performance is desired.
#define WEIGHT_DEGREE 2							// The order of the B-Spline used to splat in the weights for density estimation
#define NORMAL_DEGREE 2							// The order of the B-Spline used to splat in the normals for constructing the Laplacian constraints
#define DEFAULT_FEM_DEGREE 2					// The default finite-element degree
#define DEFAULT_FEM_BOUNDARY BOUNDARY_NEUMANN	// The default finite-element boundary type							// The dimension of the system

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>
#include "MyMiscellany.h"
#include "CmdLineParser.h"
#include "PPolynomial.h"
#include "FEMTree.h"
#include "Ply.h"
#include "PointStreamData.h"
MessageWriter messageWriter;

const float DefaultPointWeightMultiplier = 2.f;

cmdLineParameter< char* >
In("in"),
Out("out"),
TempPath("temppath"),
MyOut("myout"),
TempDir("tempDir"),
VoxelGrid("voxel"),
Tree("tree"),
Transform("xForm");

cmdLineReadable
Performance("performance"),
ShowResidual("showResidual"),
NoComments("noComments"),
PolygonMesh("polygonMesh"),
NonManifold("nonManifold"),
ASCII("ascii"),
Density("density"),
LinearFit("linearFit"),
PrimalVoxel("primalVoxel"),
ExactInterpolation("exact"),
Normals("normals"),
Colors("colors"),
Verbose("verbose");

cmdLineParameter< int >
#ifndef FAST_COMPILE
Degree("degree", DEFAULT_FEM_DEGREE),
#endif // !FAST_COMPILE
Depth("depth", 8), NormalK("k", 0),
KernelDepth("kernelDepth"),
Iters("iters", 8),
FullDepth("fullDepth", 5),
BaseDepth("baseDepth", 0),
BaseVCycles("baseVCycles", 1),
#ifndef FAST_COMPILE
BType("bType", DEFAULT_FEM_BOUNDARY + 1),
#endif // !FAST_COMPILE
MaxMemoryGB("maxMemory", 0),
Threads("threads", omp_get_num_procs());

cmdLineParameter< float >
DataX("data", 32.f),
SamplesPerNode("samplesPerNode", 1.5f),
Scale("scale", 1.1f),
Width("width", 0.f),
Confidence("confidence", 0.f),
ConfidenceBias("confidenceBias", 0.f),
CGSolverAccuracy("cgAccuracy", 1e-3f),
PointWeight("pointWeight");

cmdLineReadable* params[] =
{
#ifndef FAST_COMPILE
	&Degree , &BType ,&NormalK,
#endif // !FAST_COMPILE
	&In , &Depth , &Out , &Transform ,&TempPath,&MyOut,
	&Width ,
	&Scale , &Verbose , &CGSolverAccuracy , &NoComments ,
	&KernelDepth , &SamplesPerNode , &Confidence , &NonManifold , &PolygonMesh , &ASCII , &ShowResidual ,
	&ConfidenceBias ,
	&BaseDepth , &BaseVCycles ,
	&PointWeight ,
	&VoxelGrid , &Threads ,
	&Tree ,
	&Density ,
	&FullDepth ,
	&Iters ,
	&DataX ,
	&Colors ,
	&Normals ,
	&LinearFit ,
	&PrimalVoxel ,
	&TempDir ,
	&ExactInterpolation ,
	&Performance ,
	&MaxMemoryGB ,
	NULL
};

void ShowUsage(char* ex)
{
	printf( "Usage: %s\n" , ex );
	printf( "\t --%s <input points>\n" , In.name );
	printf( "\t[--%s <ouput triangle mesh>]\n" , Out.name );
	printf( "\t[--%s <ouput voxel grid>]\n" , VoxelGrid.name );
	printf( "\t[--%s <ouput fem tree>]\n" , Tree.name );
#ifndef FAST_COMPILE
	printf( "\t[--%s <b-spline degree>=%d]\n" , Degree.name , Degree.value );
	printf( "\t[--%s <boundary type>=%d]\n" , BType.name , BType.value );
	for( int i=0 ; i<BOUNDARY_COUNT ; i++ ) printf( "\t\t%d] %s\n" , i+1 , BoundaryNames[i] );
#endif // !FAST_COMPILE
	printf( "\t[--%s <maximum reconstruction depth>=%d]\n" , Depth.name , Depth.value );
	printf( "\t[--%s <voxel width>]\n" , Width.name );
	printf( "\t[--%s <full depth>=%d]\n" , FullDepth.name , FullDepth.value );
	printf( "\t[--%s <coarse MG solver depth>=%d]\n" , BaseDepth.name , BaseDepth.value );
	printf( "\t[--%s <coarse MG solver v-cycles>=%d]\n" , BaseVCycles.name , BaseVCycles.value );
	printf( "\t[--%s <scale factor>=%f]\n" , Scale.name , Scale.value );
	printf( "\t[--%s <minimum number of samples per node>=%f]\n" , SamplesPerNode.name, SamplesPerNode.value );
	printf( "\t[--%s <interpolation weight>=%.3e * <b-spline degree>]\n" , PointWeight.name , DefaultPointWeightMultiplier );
	printf( "\t[--%s <iterations>=%d]\n" , Iters.name , Iters.value );
	printf( "\t[--%s]\n" , ExactInterpolation.name );
	printf( "\t[--%s <pull factor>=%f]\n" , DataX.name , DataX.value );
	printf( "\t[--%s]\n" , Colors.name );
	printf( "\t[--%s]\n" , Normals.name );
#ifdef _OPENMP
	printf( "\t[--%s <num threads>=%d]\n" , Threads.name , Threads.value );
#endif // _OPENMP
	printf( "\t[--%s <normal confidence exponent>=%f]\n" , Confidence.name , Confidence.value );
	printf( "\t[--%s <normal confidence bias exponent>=%f]\n" , ConfidenceBias.name , ConfidenceBias.value );
	printf( "\t[--%s]\n" , NonManifold.name );
	printf( "\t[--%s]\n" , PolygonMesh.name );
	printf( "\t[--%s <cg solver accuracy>=%g]\n" , CGSolverAccuracy.name , CGSolverAccuracy.value );
	printf( "\t[--%s <maximum memory (in GB)>=%d]\n" , MaxMemoryGB.name , MaxMemoryGB.value );
	printf( "\t[--%s]\n" , Performance.name );
	printf( "\t[--%s]\n" , Density.name );
	printf( "\t[--%s]\n" , LinearFit.name );
	printf( "\t[--%s]\n" , PrimalVoxel.name );
	printf( "\t[--%s]\n" , ASCII.name );
	printf( "\t[--%s]\n" , NoComments.name );
	printf( "\t[--%s]\n" , TempDir.name );
	printf( "\t[--%s]\n" , Verbose.name );
}

double Weight( double v , double start , double end )
{
	v = ( v - start ) / ( end - start );
	if     ( v<0 ) return 1.;
	else if( v>1 ) return 0.;
	else
	{
		// P(x) = a x^3 + b x^2 + c x + d
		//		P (0) = 1 , P (1) = 0 , P'(0) = 0 , P'(1) = 0
		// =>	d = 1 , a + b + c + d = 0 , c = 0 , 3a + 2b + c = 0
		// =>	c = 0 , d = 1 , a + b = -1 , 3a + 2b = 0
		// =>	a = 2 , b = -3 , c = 0 , d = 1
		// =>	P(x) = 2 x^3 - 3 x^2 + 1
		return 2. * v * v * v - 3. * v * v + 1.;
	}
}

template< unsigned int Dim , class Real >
struct FEMTreeProfiler
{
	FEMTree< Dim , Real >& tree;
	double t;

	FEMTreeProfiler( FEMTree< Dim , Real >& t ) : tree(t) { ; }
	void start( void ){ t = Time() , FEMTree< Dim , Real >::ResetLocalMemoryUsage(); }
	void print( const char* header ) const
	{
		FEMTree< Dim , Real >::MemoryUsage();
		if( header ) printf( "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
		else         printf(    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
	}
	void dumpOutput( const char* header ) const
	{
		FEMTree< Dim , Real >::MemoryUsage();
		if( header ) messageWriter( "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
		else         messageWriter(    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
	}
	void dumpOutput2( std::vector< char* >& comments , const char* header ) const
	{
		FEMTree< Dim , Real >::MemoryUsage();
		if( header ) messageWriter( comments , "%s %9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n" , header , Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
		else         messageWriter( comments ,    "%9.1f (s), %9.1f (MB) / %9.1f (MB) / %9.1f (MB)\n" ,          Time()-t , FEMTree< Dim , Real >::LocalMemoryUsage() , FEMTree< Dim , Real >::MaxMemoryUsage() , MemoryInfo::PeakMemoryUsageMB() );
	}
};

template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetBoundingBoxXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real scaleFactor )
{
	Point< Real , Dim > center = ( max + min ) / 2;
	Real scale = max[0] - min[0];
	for( int d=1 ; d<Dim ; d++ ) scale = std::max< Real >( scale , max[d]-min[d] );
	scale *= scaleFactor;
	for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
	XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity();
	for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
	return sXForm * tXForm;
}
template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetBoundingBoxXForm( Point< Real , Dim > min , Point< Real , Dim > max , Real width , Real scaleFactor , int& depth )
{
	// Get the target resolution (along the largest dimension)
	Real resolution = ( max[0]-min[0] ) / width;
	for( int d=1 ; d<Dim ; d++ ) resolution = std::max< Real >( resolution , ( max[d]-min[d] ) / width );
	resolution *= scaleFactor;
	depth = 0;
	while( (1<<depth)<resolution ) depth++;

	Point< Real , Dim > center = ( max + min ) / 2;
	Real scale = (1<<depth) * width;

	for( int i=0 ; i<Dim ; i++ ) center[i] -= scale/2;
	XForm< Real , Dim+1 > tXForm = XForm< Real , Dim+1 >::Identity() , sXForm = XForm< Real , Dim+1 >::Identity();
	for( int i=0 ; i<Dim ; i++ ) sXForm(i,i) = (Real)(1./scale ) , tXForm(Dim,i) = -center[i];
	return sXForm * tXForm;
}

template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetPointXForm( InputPointStream< Real , Dim >& stream , Real width , Real scaleFactor , int& depth )
{
	Point< Real , Dim > min , max;
	stream.boundingBox( min , max );
	return GetBoundingBoxXForm( min , max , width , scaleFactor , depth );
}
template< class Real , unsigned int Dim >
XForm< Real , Dim+1 > GetPointXForm( InputPointStream< Real , Dim >& stream , Real scaleFactor )
{
	Point< Real , Dim > min , max;
	stream.boundingBox( min , max );
	return GetBoundingBoxXForm( min , max , scaleFactor );
}

template< unsigned int Dim , typename Real >
struct ConstraintDual
{
	Real target , weight;
	ConstraintDual( Real t , Real w ) : target(t) , weight(w){ }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p ) const { return CumulativeDerivativeValues< Real , Dim , 0 >( target*weight ); };
};
template< unsigned int Dim , typename Real >
struct SystemDual
{
	Real weight;
	SystemDual( Real w ) : weight(w){ }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< Real , Dim , 0 >& dValues ) const { return dValues * weight; };
	CumulativeDerivativeValues< double , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< double , Dim , 0 >& dValues ) const { return dValues * weight; };
};
template< unsigned int Dim >
struct SystemDual< Dim , double >
{
	typedef double Real;
	Real weight;
	SystemDual( Real w ) : weight(w){ }
	CumulativeDerivativeValues< Real , Dim , 0 > operator()( const Point< Real , Dim >& p , const CumulativeDerivativeValues< Real , Dim , 0 >& dValues ) const { return dValues * weight; };
};

/*******************Modified by Fei Hou and Chiyu Wang*************************/

template< class Vertex, class Real, int Dim >
std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>> export_mesh(CoredMeshData< Vertex >* mesh)
{
	using namespace std;
	int nr_vertices = int(mesh->outOfCorePointCount() + mesh->inCorePoints.size());
	int nr_faces = mesh->polygonCount();

	mesh->resetIterator();

	// write vertices
	vector<Point<Real, Dim>> vertices;
	vertices.reserve(mesh->inCorePoints.size() + mesh->outOfCorePointCount());
	for (int i = 0; i<int(mesh->inCorePoints.size()); i++)
	{
		vertices.push_back(mesh->inCorePoints[i].point);
	}
	for (int i = 0; i < mesh->outOfCorePointCount(); i++)
	{
		Vertex vertex;
		mesh->nextOutOfCorePoint(vertex);
		vertices.push_back(vertex.point);
	}

	// write faces
	vector<vector<int>> faces(nr_faces);
	std::vector< CoredVertexIndex > polygon;
	for (int i = 0; i < nr_faces; i++)
	{
		//
		// create and fill a struct that the ply code can handle
		//
		mesh->nextPolygon(polygon);
		vector<int> face_id(polygon.size());
		for (size_t j = 0; j < polygon.size(); j++)
		{
			if (polygon[j].inCore) face_id[j] = polygon[j].idx;
			else face_id[j] = polygon[j].idx + int(mesh->inCorePoints.size());
		}
		faces[i] = face_id;
	}  // for, write faces

	return std::make_pair(vertices, faces);
}

template< class Real , int Dim , class StreamDataInfo , class Vertex , unsigned int ... FEMSigs >
std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>> Execute(const std::vector<double>* weight_samples, int argc , char* argv[] , const std::vector<std::pair<Point<Real, Dim>, typename StreamDataInfo::Type>>& points_normals, UIntPack< FEMSigs ... > )
{
	typedef UIntPack< FEMSigs ... > Sigs;
	typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > Degrees;
	typedef UIntPack< FEMDegreeAndBType< NORMAL_DEGREE , DerivativeBoundary< FEMSignature< FEMSigs >::BType , 1 >::BType >::Signature ... > NormalSigs;
	static const unsigned int DataSig = FEMDegreeAndBType< DATA_DEGREE , BOUNDARY_FREE >::Signature;
	typedef Point< Real , 3 > Color;
	typedef typename FEMTree< Dim , Real >::template DensityEstimator< WEIGHT_DEGREE > DensityEstimator;
	typedef typename FEMTree< Dim , Real >::template InterpolationInfo< Real , 0 > InterpolationInfo;
	typedef InputPointStreamWithData< Real , Dim , typename StreamDataInfo::Type > InputPointStream;
	typedef TransformedInputPointStreamWithData< Real , Dim , typename StreamDataInfo::Type > XInputPointStream;
	std::vector< char* > comments;
	messageWriter( comments , "*************************************************************\n" );
	messageWriter( comments , "*************************************************************\n" );
	messageWriter( comments , "** Running Screened Poisson Reconstruction (Version %s) **\n" , VERSION );
	messageWriter( comments , "*************************************************************\n" );
	messageWriter( comments , "*************************************************************\n" );

	XForm< Real , Dim+1 > xForm , iXForm;
	if(Transform.set )
	{
		FILE* fp = fopen(Transform.value , "r" );
		if( !fp )
		{
			fprintf( stderr , "[WARNING] Could not read x-form from: %s\n" , Transform.value );
			xForm = XForm< Real , Dim+1 >::Identity();
		}
		else
		{
			for( int i=0 ; i<Dim+1 ; i++ ) for( int j=0 ; j<Dim+1 ; j++ )
			{
				float f;
				if( fscanf( fp , " %f " , &f )!=1 ) fprintf( stderr , "[ERROR] Execute: Failed to read xform\n" ) , exit( 0 );
				xForm(i,j) = (Real)f;
			}
			fclose( fp );
		}
	}
	else xForm = XForm< Real , Dim+1 >::Identity();

	char str[1024];
	for( int i=0 ; params[i] ; i++ )
		if( params[i]->set )
		{
			params[i]->writeValue( str );
			if( strlen( str ) ) messageWriter( comments , "\t--%s %s\n" , params[i]->name , str );
			else                messageWriter( comments , "\t--%s\n" , params[i]->name );
		}

	double startTime = Time();
	Real isoValue = 0;

	FEMTree< Dim , Real > tree( MEMORY_ALLOCATOR_BLOCK_SIZE );
	FEMTreeProfiler< Dim , Real > profiler( tree );

	if(Depth.set && Width.value>0 )
	{
		fprintf( stderr , "[WARNING] Both --%s and --%s set, ignoring --%s\n" , Depth.name , Width.name , Width.name );
		Width.value = 0;
	}

	int pointCount;

	Real pointWeightSum;
	std::vector< typename FEMTree< Dim , Real >::PointSample >* samples = new std::vector< typename FEMTree< Dim , Real >::PointSample >();
	std::vector< typename StreamDataInfo::Type >* sampleData = NULL;
	DensityEstimator* density = NULL;
	SparseNodeData< Point< Real , Dim > , NormalSigs >* normalInfo = NULL;
	Real targetValue = (Real)0.5;

	// Read in the samples (and color data)
	{
		profiler.start();
		InputPointStream* pointStream = new MemoryInputPointStreamWithData<Real, Dim, typename StreamDataInfo::Type>(points_normals.size(), points_normals.data());
		sampleData = new std::vector< typename StreamDataInfo::Type >();
		/*char* ext = GetFileExtension(In.value );
		if     ( !strcasecmp( ext , "bnpts" ) ) pointStream = new BinaryInputPointStreamWithData< Real , Dim , typename StreamDataInfo::Type >(In.value , StreamDataInfo::ReadBinary );
		else if( !strcasecmp( ext , "ply"   ) ) pointStream = new    PLYInputPointStreamWithData< Real , Dim , typename StreamDataInfo::Type >(In.value , StreamDataInfo::PlyProperties , StreamDataInfo::PlyPropertyNum , StreamDataInfo::ValidPlyProperties );
		else                                    pointStream = new  ASCIIInputPointStreamWithData< Real , Dim , typename StreamDataInfo::Type >(In.value , StreamDataInfo::ReadASCII );
		delete[] ext;*/
		XInputPointStream _pointStream( typename StreamDataInfo::Transform( xForm ) , *pointStream );
		//if(Width.value>0 ) xForm = GetPointXForm< Real , Dim >( _pointStream , Width.value , (Real)(Scale.value>0 ? Scale.value : 1. ) , Depth.value ) * xForm;
		//else                xForm = Scale.value>0 ? GetPointXForm< Real , Dim >( _pointStream , (Real)Scale.value ) * xForm : xForm;
		{
            XInputPointStream _pointStream( typename StreamDataInfo::Transform( xForm ) , *pointStream );
			if (Confidence.value > 0) pointCount = FEMTreeInitializer< Dim, Real >::template Initialize< typename StreamDataInfo::Type >(tree.spaceRoot(), _pointStream, Depth.value, *samples, *sampleData, true, tree.nodeAllocator, tree.initializer(), [&](const Point< Real, Dim >&p, typename StreamDataInfo::Type& d) { return (Real)pow(StreamDataInfo::ProcessDataWithConfidence(p, d), Confidence.value); });
			else                    pointCount = FEMTreeInitializer< Dim, Real >::template Initialize< typename StreamDataInfo::Type >(tree.spaceRoot(), _pointStream, Depth.value, *samples, *sampleData, true, tree.nodeAllocator, tree.initializer(), StreamDataInfo::ProcessData);
		}
		iXForm = xForm.inverse();
		delete pointStream;
		for (size_t i = 0; i < samples->size(); i++)
		{
			double weight_sample = (*weight_samples)[i];
            (*samples)[i].sample.weight= weight_sample;
			(*samples)[i].sample.data.coords[0] *= weight_sample;
			(*samples)[i].sample.data.coords[1] *= weight_sample;
			(*samples)[i].sample.data.coords[2] *= weight_sample;
			(*sampleData)[i].normal[0] *= weight_sample;
			(*sampleData)[i].normal[1] *= weight_sample;
			(*sampleData)[i].normal[2] *= weight_sample;
		}

		messageWriter( "Input Points / Samples: %d / %d\n" , pointCount , samples->size() );
		profiler.dumpOutput2( comments , "# Read input into tree:" );
	}
	int kernelDepth = KernelDepth.set ? KernelDepth.value : Depth.value-2;
	if( kernelDepth> Depth.value )
	{
		fprintf( stderr,"[WARNING] %s can't be greater than %s: %d <= %d\n" , KernelDepth.name , Depth.name , KernelDepth.value , Depth.value );
		kernelDepth = Depth.value;
	}

	DenseNodeData< Real , Sigs > solution;
	{
		DenseNodeData< Real , Sigs > constraints;
		InterpolationInfo* iInfo = NULL;
		int solveDepth = Depth.value;

		tree.resetNodeIndices();

		// Get the kernel density estimator
		{
			profiler.start();
			density = tree.template setDensityEstimator< WEIGHT_DEGREE >( *samples , kernelDepth , SamplesPerNode.value , 1 );
			profiler.dumpOutput2( comments , "#   Got kernel density:" );
		}

		// Transform the Hermite samples into a vector field
		{
			profiler.start();
			normalInfo = new SparseNodeData< Point< Real , Dim > , NormalSigs >();
			if(ConfidenceBias.value>0 ) *normalInfo = tree.setNormalField( NormalSigs() , *samples , *sampleData , density , pointWeightSum , [&]( Real conf ){ return (Real)( log( conf ) * ConfidenceBias.value / log( 1<<(Dim-1) ) ); } );
			else                         *normalInfo = tree.setNormalField( NormalSigs() , *samples , *sampleData , density , pointWeightSum );
#pragma omp parallel for
			for( int i=0 ; i<normalInfo->size() ; i++ ) (*normalInfo)[i] *= (Real)-1.;
			profiler.dumpOutput2( comments , "#     Got normal field:" );
			messageWriter( "Point weight / Estimated Area: %g / %g\n" , pointWeightSum , pointCount*pointWeightSum );
		}

		if( !Density.set ) delete density , density = NULL;
		if(DataX.value<=0 || ( !Colors.set && !Normals.set ) ) delete sampleData , sampleData = NULL;

		// Trim the tree and prepare for multigrid
		{
			profiler.start();
			constexpr int MAX_DEGREE = NORMAL_DEGREE > Degrees::Max() ? NORMAL_DEGREE : Degrees::Max();
			tree.template finalizeForMultigrid< MAX_DEGREE >(FullDepth.value , typename FEMTree< Dim , Real >::template HasNormalDataFunctor< NormalSigs >( *normalInfo ) , normalInfo , density );
			profiler.dumpOutput2( comments , "#       Finalized tree:" );
		}
		// Add the FEM constraints
		{
			profiler.start();
			constraints = tree.initDenseNodeData( Sigs() );
			typename FEMIntegrator::template Constraint< Sigs , IsotropicUIntPack< Dim , 1 > , NormalSigs , IsotropicUIntPack< Dim , 0 > , Dim > F;
			unsigned int derivatives2[Dim];
			for( int d=0 ; d<Dim ; d++ ) derivatives2[d] = 0;
			typedef IsotropicUIntPack< Dim , 1 > Derivatives1;
			typedef IsotropicUIntPack< Dim , 0 > Derivatives2;
			for( int d=0 ; d<Dim ; d++ )
			{
				unsigned int derivatives1[Dim];
				for( int dd=0 ; dd<Dim ; dd++ ) derivatives1[dd] = dd==d ?  1 : 0;
				F.weights[d][ TensorDerivatives< Derivatives1 >::Index( derivatives1 ) ][ TensorDerivatives< Derivatives2 >::Index( derivatives2 ) ] = 1;
			}
			tree.addFEMConstraints( F , *normalInfo , constraints , solveDepth );
			profiler.dumpOutput2( comments , "#  Set FEM constraints:" );
		}

		// Free up the normal info
		delete normalInfo , normalInfo = NULL;

		// Add the interpolation constraints
		if(PointWeight.value>0 )
		{
			//printf("PointWeight: %f")
			profiler.start();
			if(ExactInterpolation.set ) iInfo = FEMTree< Dim , Real >::template       InitializeExactPointInterpolationInfo< Real , 0 > ( tree , *samples , ConstraintDual< Dim , Real >( targetValue , (Real)PointWeight.value * pointWeightSum ) , SystemDual< Dim , Real >( (Real)PointWeight.value * pointWeightSum ) , true , false );
			else                         iInfo = FEMTree< Dim , Real >::template InitializeApproximatePointInterpolationInfo< Real , 0 > ( tree , *samples , ConstraintDual< Dim , Real >( targetValue , (Real)PointWeight.value * pointWeightSum ) , SystemDual< Dim , Real >( (Real)PointWeight.value * pointWeightSum ) , true , 1 );
			tree.addInterpolationConstraints( constraints , solveDepth , *iInfo );
			profiler.dumpOutput2( comments , "#Set point constraints:" );
		}

		messageWriter( "Leaf Nodes / Active Nodes / Ghost Nodes: %d / %d / %d\n" , (int)tree.leaves() , (int)tree.nodes() , (int)tree.ghostNodes() );
		messageWriter( "Memory Usage: %.3f MB\n" , float( MemoryInfo::Usage())/(1<<20) );
		
		// Solve the linear system
		{
			profiler.start();
			typename FEMTree< Dim , Real >::SolverInfo sInfo;
			sInfo.cgDepth = 0 , sInfo.cascadic = true , sInfo.vCycles = 1 , sInfo.iters = Iters.value , sInfo.cgAccuracy = CGSolverAccuracy.value , sInfo.verbose = Verbose.set , sInfo.showResidual = ShowResidual.set , sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE , sInfo.sliceBlockSize = 1;
			sInfo.baseDepth = BaseDepth.value , sInfo.baseVCycles = BaseVCycles.value;
			typename FEMIntegrator::template System< Sigs , IsotropicUIntPack< Dim , 1 > > F( { 0. , 1. } );
			solution = tree.solveSystem( Sigs() , F , constraints , solveDepth , sInfo , iInfo );
			profiler.dumpOutput2( comments , "# Linear system solved:" );
			if( iInfo ) delete iInfo , iInfo = NULL;
		}
	}

	{
		profiler.start();
		double valueSum = 0 , weightSum = 0;
		typename FEMTree< Dim , Real >::template MultiThreadedEvaluator< Sigs , 0 > evaluator( &tree , solution );
#pragma omp parallel for reduction( + : valueSum , weightSum )
		for( int j=0 ; j<samples->size() ; j++ )
		{
			ProjectiveData< Point< Real , Dim > , Real >& sample = (*samples)[j].sample;
			Real w = sample.weight;
			if( w>0 ) weightSum += w , valueSum += evaluator.values( sample.data / sample.weight , omp_get_thread_num() , (*samples)[j].node )[0] * w;
		}
		isoValue = (Real)( valueSum / weightSum );
		if(DataX.value<=0 || ( !Colors.set && !Normals.set ) ) delete samples , samples = NULL;
		profiler.dumpOutput( "Got average:" );
		messageWriter( "Iso-Value: %e = %g / %g\n" , isoValue , valueSum , weightSum );
	}
	if(Tree.set )
	{
		FILE* fp = fopen(Tree.value , "wb" );
		if( !fp ) fprintf( stderr , "[ERROR] Failed to open file for writing: %s\n" , Tree.value ) , exit( 0 );
		FEMTree< Dim , Real >::WriteParameter( fp );
		DenseNodeData< Real , Sigs >::WriteSignatures( fp );
		tree.write( fp );
		solution.write( fp );
		fclose( fp );
	}

	if(VoxelGrid.set )
	{
		FILE* fp = fopen(VoxelGrid.value , "wb" );
		if( !fp ) fprintf( stderr , "Failed to open voxel file for writing: %s\n" , VoxelGrid.value );
		else
		{
			int res = 0;
			profiler.start();
			Pointer( Real ) values = tree.template regularGridEvaluate< true >( solution , res , -1 , PrimalVoxel.set );
#pragma omp parallel for
			for( int i=0 ; i<res*res*res ; i++ ) values[i] -= isoValue;
			profiler.dumpOutput( "Got voxel grid:" );
			fwrite( &res , sizeof(int) , 1 , fp );
			if( typeid(Real)==typeid(float) ) fwrite( values , sizeof(float) , res*res*res , fp );
			else
			{
				float *fValues = new float[res*res*res];
				for( int i=0 ; i<res*res*res ; i++ ) fValues[i] = float( values[i] );
				fwrite( fValues , sizeof(float) , res*res*res , fp );
				delete[] fValues;
			}
			fclose( fp );
			DeletePointer( values );
		}
	}

	std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>> mesh_model;
	if(Out.set )
	{
		/*char tempHeader[1024];
		{
			char tempPath[1024];
			tempPath[0] = 0;
			if(TempDir.set ) strcpy( tempPath , TempDir.value );
			else SetTempDirectory( tempPath , sizeof(tempPath) );
			if( strlen(tempPath)==0 ) sprintf( tempPath , ".%c" , FileSeparator );
			if( tempPath[ strlen( tempPath )-1 ]==FileSeparator ) sprintf( tempHeader , "%sPR_" , tempPath );
			else                                                  sprintf( tempHeader , "%s%cPR_" , tempPath , FileSeparator );
		}*/

		CoredVectorMeshData< Vertex > mesh;

		profiler.start();
		typename IsoSurfaceExtractor< Dim , Real , Vertex >::IsoStats isoStats;
		if( sampleData )
		{
			SparseNodeData< ProjectiveData< typename StreamDataInfo::Type , Real > , IsotropicUIntPack< Dim , DataSig > > _sampleData = tree.template setDataField< DataSig , false >( *samples , *sampleData , (DensityEstimator*)NULL );
			for( const RegularTreeNode< Dim , FEMTreeNodeData >* n = tree.tree().nextNode() ; n ; n=tree.tree().nextNode( n ) )
			{
				ProjectiveData< typename StreamDataInfo::Type , Real >* clr = _sampleData( n );
				if( clr ) (*clr) *= (Real)pow(DataX.value , tree.depth( n ) );
			}
			delete sampleData , sampleData = NULL;

			isoStats = IsoSurfaceExtractor< Dim , Real , Vertex >::template Extract< typename StreamDataInfo::Type >( Sigs() , UIntPack< WEIGHT_DEGREE >() , UIntPack< DataSig >() , tree , density , &_sampleData , solution , isoValue , mesh , StreamDataInfo::template VertexSetter< Vertex >::SetValue , StreamDataInfo::template VertexSetter< Vertex >::SetData , !LinearFit.set , !NonManifold.set , PolygonMesh.set , false );
		}
		else isoStats = IsoSurfaceExtractor< Dim , Real , Vertex >::template Extract< typename StreamDataInfo::Type >( Sigs() , UIntPack< WEIGHT_DEGREE >() , UIntPack< DataSig >() , tree , density , NULL , solution , isoValue , mesh , StreamDataInfo::template VertexSetter< Vertex >::SetValue , StreamDataInfo::template VertexSetter< Vertex >::SetData , !LinearFit.set , !NonManifold.set , PolygonMesh.set , false );
		messageWriter( "Vertices / Polygons: %d / %d\n" , mesh.outOfCorePointCount()+mesh.inCorePoints.size() , mesh.polygonCount() );
		messageWriter( "Corners / Vertices / Edges / Surface / Set Table / Copy Finer: %.1f / %.1f / %.1f / %.1f / %.1f / %.1f (s)\n" , isoStats.cornersTime , isoStats.verticesTime , isoStats.edgesTime , isoStats.surfaceTime , isoStats.setTableTime , isoStats.copyFinerTime );
		if(PolygonMesh.set ) profiler.dumpOutput2( comments , "#         Got polygons:" );
		else                  profiler.dumpOutput2( comments , "#        Got triangles:" );

		mesh_model = export_mesh<Vertex, Real, Dim>(&mesh);
	}
	if( density ) delete density , density = NULL;
	messageWriter( comments , "#          Total Solve: %9.1f (s), %9.1f (MB)\n" , Time()-startTime , FEMTree< Dim , Real >::MaxMemoryUsage() );

	return mesh_model;
}

#ifndef FAST_COMPILE
template< class Real , unsigned int Dim, class InfoType , class Vertex >
std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>> Execute(const std::vector<double>* weight_samples, int argc , char* argv[] , const std::vector<std::pair<Point<Real, Dim>, typename InfoType::Type>>& points_normals)
{
	switch(BType.value )
	{
	case BOUNDARY_FREE+1:
	{
		switch(Degree.value )
		{
			case 1: return Execute< Real , Dim , InfoType , Vertex >(weight_samples, argc , argv , points_normals, IsotropicUIntPack< Dim , FEMDegreeAndBType< 1 , BOUNDARY_FREE >::Signature >() );
			case 2: return Execute< Real ,Dim , InfoType , Vertex >(weight_samples, argc , argv , points_normals, IsotropicUIntPack< Dim , FEMDegreeAndBType< 2 , BOUNDARY_FREE >::Signature >() );
//			case 3: return Execute< Real , InfoType , Vertex >( weight_samples, argc , argv , points_normals,IsotropicUIntPack< Dim , FEMDegreeAndBType< 3 , BOUNDARY_FREE >::Signature >() );
//			case 4: return Execute< Real , InfoType , Vertex >(weight_samples,  argc , argv , points_normals,IsotropicUIntPack< Dim , FEMDegreeAndBType< 4 , BOUNDARY_FREE >::Signature >() );
			default: fprintf( stderr , "[ERROR] Only B-Splines of degree 1 - 2 are supported" ) ; //return EXIT_FAILURE;
				return std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>>();
		}
	}
	case BOUNDARY_NEUMANN+1:
	{
	switch(Degree.value )
		{
			case 1: return Execute< Real , Dim , InfoType , Vertex >(weight_samples, argc , argv , points_normals, IsotropicUIntPack< Dim , FEMDegreeAndBType< 1 , BOUNDARY_NEUMANN >::Signature >() );
			case 2: return Execute< Real , Dim , InfoType , Vertex >(weight_samples, argc , argv , points_normals, IsotropicUIntPack< Dim , FEMDegreeAndBType< 2 , BOUNDARY_NEUMANN >::Signature >() );
//			case 3: return Execute< Real , InfoType , Vertex >(weight_samples,  argc , argv , points_normals,IsotropicUIntPack< Dim , FEMDegreeAndBType< 3 , BOUNDARY_NEUMANN >::Signature >() );
//			case 4: return Execute< Real , InfoType , Vertex >(weight_samples,  argc , argv , points_normals,IsotropicUIntPack< Dim , FEMDegreeAndBType< 4 , BOUNDARY_NEUMANN >::Signature >() );
			default: fprintf( stderr , "[ERROR] Only B-Splines of degree 1 - 2 are supported" ) ;   //return EXIT_FAILURE;
				return std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>>();
		}
	}
	case BOUNDARY_DIRICHLET+1:
	{
		switch(Degree.value )
		{
			case 1: return Execute< Real , Dim , InfoType , Vertex >(weight_samples, argc , argv , points_normals, IsotropicUIntPack< Dim , FEMDegreeAndBType< 1 , BOUNDARY_DIRICHLET >::Signature >() );
			case 2: return Execute< Real , Dim , InfoType , Vertex >(weight_samples, argc , argv , points_normals, IsotropicUIntPack< Dim , FEMDegreeAndBType< 2 , BOUNDARY_DIRICHLET >::Signature >() );
//			case 3: return Execute< Real , InfoType , Vertex >(weight_samples, argc , argv , points_normals,IsotropicUIntPack< Dim , FEMDegreeAndBType< 3 , BOUNDARY_DIRICHLET >::Signature >() );
//			case 4: return Execute< Real , InfoType , Vertex >(weight_samples, argc , argv , points_normals,IsotropicUIntPack< Dim , FEMDegreeAndBType< 4 , BOUNDARY_DIRICHLET >::Signature >() );
			default: fprintf( stderr , "[ERROR] Only B-Splines of degree 1 - 2 are supported" ) ; //return EXIT_FAILURE;
				return std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>>();
		}
	}
	default: fprintf( stderr , "[ERROR] Not a valid boundary type: %d\n" , BType.value ) ; //return EXIT_FAILURE;
		return std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>>();
	}
}
#endif // !FAST_COMPILE

template< class Real, unsigned int Dim>
std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>> poisson_reconstruction(int argc , char* argv[] , const std::vector<std::pair<Point<Real, Dim>, Normal<Real, Dim>>>& points_normals, const std::vector<double>* weight_samples)
{
	Timer timer;
#ifdef ARRAY_DEBUG
	fprintf( stderr , "[WARNING] Array debugging enabled\n" );
#endif // ARRAY_DEBUG

	cmdLineParse( argc-1 , &argv[1] , params );
	if(MaxMemoryGB.value>0 ) SetPeakMemoryMB(MaxMemoryGB.value<<10 );
	omp_set_num_threads(Threads.value > 1 ? Threads.value : 1 );
	messageWriter.echoSTDOUT = Verbose.set;

	if( !In.set )
	{
		ShowUsage( argv[0] );
		return std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>>();
	}
	if(DataX.value<=0 ) Normals.set = Colors.set = false;
	if(BaseDepth.value> FullDepth.value )
	{
		if(BaseDepth.set ) fprintf( stderr , "[WARNING] Base depth must be smaller than full depth: %d <= %d\n" , BaseDepth.value , FullDepth.value );
		BaseDepth.value = FullDepth.value;
	}

	std::pair<std::vector<Point<Real, Dim>>, std::vector<std::vector<int>>> mesh;

#ifdef FAST_COMPILE
	static const int Degree = DEFAULT_FEM_DEGREE;
	static const BoundaryType BType = DEFAULT_FEM_BOUNDARY;
	typedef IsotropicUIntPack< Dim , FEMDegreeAndBType< Degree , BType >::Signature > FEMSigs;
	fprintf( stderr , "[WARNING] Compiled for degree-%d, boundary-%s, %s-precision _only_\n" , Degree , BoundaryNames[ BType ] , sizeof(Real)==4 ? "single" : "double" );
	if( !PointWeight.set ) PointWeight.value = DefaultPointWeightMultiplier*Degree;
	if(Normals.set )
		if(Colors.set )
			if(Density.set ) Execute< Real , NormalAndColorInfo< Real , Dim > , FullPlyVertex< float , Dim , true  , true  , true > >( argc , argv , FEMSigs() );
			else              Execute< Real , NormalAndColorInfo< Real , Dim > , FullPlyVertex< float , Dim , true  , false , true > >( argc , argv , FEMSigs() );
		else
			if(Density.set ) Execute< Real , NormalInfo< Real , Dim > , FullPlyVertex< float , Dim , true  , true  , false > >( argc , argv , FEMSigs() );
			else              Execute< Real , NormalInfo< Real , Dim > , FullPlyVertex< float , Dim , true  , false , false > >( argc , argv , FEMSigs() );
	else
		if(Colors.set )
			if(Density.set ) Execute< Real , NormalAndColorInfo< Real , Dim > , FullPlyVertex< float , Dim , false , true  , true > >( argc , argv , FEMSigs() );
			else              Execute< Real , NormalAndColorInfo< Real , Dim > , FullPlyVertex< float , Dim , false , false , true > >( argc , argv , FEMSigs() );
		else
			if(Density.set ) Execute< Real , NormalInfo< Real , Dim > , FullPlyVertex< float , DIMENSION , false , true  , false > >( argc , argv , FEMSigs() );
			else              Execute< Real , NormalInfo< Real , Dim > , FullPlyVertex< float , DIMENSION , false , false , false > >( argc , argv , FEMSigs() );
#else // !FAST_COMPILE
	if( !PointWeight.set ) PointWeight.value = DefaultPointWeightMultiplier* Degree.value;
	mesh = Execute<Real, Dim, NormalInfo< Real, Dim >, FullPlyVertex< float, Dim, false, false, false > >(weight_samples,argc, argv, points_normals);
#endif // FAST_COMPILE
	if(Performance.set )
	{
		printf( "Time (Wall/CPU): %.2f / %.2f\n" , timer.wallTime() , timer.cpuTime() );
		printf( "Peak Memory (MB): %d\n" , MemoryInfo::PeakMemoryUsageMB() );
	}
	return mesh;
}

template< class Real, unsigned int Dim, class StreamDataInfo>
std::vector<std::pair<Point<Real, Dim>, Normal<Real, Dim>>> _SamplePoints(int argc, char* argv[], const std::vector<std::pair<Point<Real, Dim>, typename StreamDataInfo::Type>>& points_normals,XForm<Real,Dim+1>& iXForm, std::vector<double> *weight_samples)
{
	typedef typename FEMTree< Dim, Real >::template DensityEstimator< WEIGHT_DEGREE > DensityEstimator;
	typedef typename FEMTree< Dim, Real >::template InterpolationInfo< Real, 0 > InterpolationInfo;
	typedef InputPointStreamWithData< Real, Dim, typename StreamDataInfo::Type > InputPointStream;
	typedef TransformedInputPointStreamWithData< Real, Dim, typename StreamDataInfo::Type > XInputPointStream;
	std::vector< char* > comments;

	XForm< Real, Dim + 1 > xForm;
	if (Transform.set)
	{
		FILE* fp = fopen(Transform.value, "r");
		if (!fp)
		{
			fprintf(stderr, "[WARNING] Could not read x-form from: %s\n", Transform.value);
			xForm = XForm< Real, Dim + 1 >::Identity();
		}
		else
		{
			for (int i = 0; i < Dim + 1; i++) for (int j = 0; j < Dim + 1; j++)
			{
				float f;
				if (fscanf(fp, " %f ", &f) != 1) fprintf(stderr, "[ERROR] Execute: Failed to read xform\n"), exit(0);
				xForm(i, j) = (Real)f;
			}
			fclose(fp);
		}
	}
	else xForm = XForm< Real, Dim + 1 >::Identity();

	char str[1024];
	for (int i = 0; params[i]; i++)
		if (params[i]->set)
		{
			params[i]->writeValue(str);
			if (strlen(str)) messageWriter(comments, "\t--%s %s\n", params[i]->name, str);
			else                messageWriter(comments, "\t--%s\n", params[i]->name);
		}

	double startTime = Time();
	Real isoValue = 0;

	FEMTree< Dim, Real > tree(MEMORY_ALLOCATOR_BLOCK_SIZE);
	FEMTreeProfiler< Dim, Real > profiler(tree);

	if (Depth.set && Width.value > 0)
	{
		fprintf(stderr, "[WARNING] Both --%s and --%s set, ignoring --%s\n", Depth.name, Width.name, Width.name);
		Width.value = 0;
	}

	int pointCount;

	std::vector< typename FEMTree< Dim, Real >::PointSample >* samples = new std::vector< typename FEMTree< Dim, Real >::PointSample >();
	std::vector< typename StreamDataInfo::Type >* sampleData = NULL;

	// Read in the samples (and color data)
	{
		profiler.start();
		InputPointStream* pointStream = new MemoryInputPointStreamWithData<Real, Dim, typename StreamDataInfo::Type>(points_normals.size(), points_normals.data());
		sampleData = new std::vector< typename StreamDataInfo::Type >();

		XInputPointStream _pointStream(typename StreamDataInfo::Transform(xForm), *pointStream);
		if (Width.value > 0) xForm = GetPointXForm< Real, Dim >(_pointStream, Width.value, (Real)(Scale.value > 0 ? Scale.value : 1.), Depth.value) * xForm;
		else                xForm = Scale.value > 0 ? GetPointXForm< Real, Dim >(_pointStream, (Real)Scale.value) * xForm : xForm;
		{
			XInputPointStream _pointStream(typename StreamDataInfo::Transform(xForm), *pointStream);
			if (Confidence.value > 0) pointCount = FEMTreeInitializer< Dim, Real >::template Initialize< typename StreamDataInfo::Type >(tree.spaceRoot(), _pointStream, Depth.value, *samples, *sampleData, true, tree.nodeAllocator, tree.initializer(), [&](const Point< Real, Dim >&p, typename StreamDataInfo::Type& d) { return (Real)pow(StreamDataInfo::ProcessDataWithConfidence(p, d), Confidence.value); });
			else                     pointCount = FEMTreeInitializer< Dim, Real >::template Initialize< typename StreamDataInfo::Type >(tree.spaceRoot(), _pointStream, Depth.value, *samples, *sampleData, true, tree.nodeAllocator, tree.initializer(), StreamDataInfo::ProcessData);
		}
		iXForm = xForm.inverse();
		delete pointStream;

		std::vector<std::pair<Point<Real, Dim>, Normal<Real, Dim>>> sample_points(samples->size());

		weight_samples->resize(samples->size());
		for (size_t i = 0; i < samples->size(); i++)
		{
			double weight_sample = (*samples)[i].sample.weight;
			Point<Real, Dim> p;
			p[0] = (*samples)[i].sample.data.coords[0] / weight_sample;
			p[1] = (*samples)[i].sample.data.coords[1] / weight_sample;
			p[2] = (*samples)[i].sample.data.coords[2] / weight_sample;
			Normal<Real, Dim > n;
			n.normal[0] = (*sampleData)[i].normal[0] / weight_sample;
			n.normal[1] = (*sampleData)[i].normal[1] / weight_sample;
			n.normal[2] = (*sampleData)[i].normal[2] / weight_sample;
			sample_points[i] = std::make_pair(p, n);
			(*weight_samples)[i] = weight_sample;
		}
		return sample_points;
	}
}
template< class Real, unsigned int Dim>
std::vector<std::pair<Point<Real, Dim>, Normal<Real, Dim>>> SamplePoints(int argc, char* argv[], const std::vector<std::pair<Point<Real, Dim>, Normal<Real, Dim>>>& points_normals, XForm<Real, Dim+1>& iXForm, std::vector<double> *weight_samples)
{
#ifdef ARRAY_DEBUG
	fprintf(stderr, "[WARNING] Array debugging enabled\n");
#endif // ARRAY_DEBUG

	cmdLineParse(argc - 1, &argv[1], params);
	if (MaxMemoryGB.value > 0) SetPeakMemoryMB(MaxMemoryGB.value << 10);
	omp_set_num_threads(Threads.value > 1 ? Threads.value : 1);
	messageWriter.echoSTDOUT = Verbose.set;

	if (!In.set)
	{
		ShowUsage(argv[0]);
		return std::vector<std::pair<Point<Real, Dim>, Normal<Real, Dim>>>();
	}
	if (DataX.value <= 0) Normals.set = Colors.set = false;
	if (BaseDepth.value > FullDepth.value)
	{
		if (BaseDepth.set) fprintf(stderr, "[WARNING] Base depth must be smaller than full depth: %d <= %d\n", BaseDepth.value, FullDepth.value);
		BaseDepth.value = FullDepth.value;
	}

	if (!PointWeight.set) PointWeight.value = DefaultPointWeightMultiplier * Degree.value;

	return _SamplePoints<Real,Dim, NormalInfo< Real,Dim >>(argc, argv, points_normals, iXForm,weight_samples);
}
