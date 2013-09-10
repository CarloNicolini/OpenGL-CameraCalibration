// This file is part of CNCSVision, a computer vision related library
// This software is developed under the grant of Italian Institute of Technology
//
// Copyright (C) 2013 Carlo Nicolini <carlo.nicolini@iit.it>
//
//
// CNCSVision is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// CNCSVision is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// CNCSVision. If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <stdexcept>

#include "CameraDirectLinearTransformation.h"

using namespace std;
using namespace Eigen;

/**
 * @brief loadImages
 * @param filename
 * @return
 */
std::vector<Vector3d> loadImages(const string &filename)
{
    ifstream is;
    is.open(filename.c_str());
    std::vector<Vector3d> vals;
    double u,v;
    while( (is >> u ) &&  (is >> v)  )
    {
        vals.push_back(Vector3d(u,v,1.0));
    }
    //cerr << "Tot " << vals.size() << " image points" << endl;
    return vals;
}

/**
 * @brief loadWorldCoords
 * @param filename
 * @return
 */
std::vector<Vector4d> loadWorldCoords(const string &filename)
{
    ifstream is;
    is.open(filename.c_str());
    std::vector<Vector4d> vals;
    double x,y,z;
    while( (is >> x ) &&  (is >> y) && (is >> z ) )
    {
        vals.push_back(Vector4d(x,y,z,1.0));
    }
    //cerr << "Tot " << vals.size() << " world points" << endl;
    return vals;
}

void writeToFile(ostream &is, const Affine3d& MV, const Projective3d &Proj)
{
    is << "MV=\n";
    for (int i=0; i<4; i++)
    {
        for (int j=0; j<4; j++)
        {
            is << MV.matrix()(i,j) << " " ;
            if (i==3 && j==3)
                is << ";";
            else
                is << ",";
        }
        is << endl;
    }

    is << "P=\n";
    for (int i=0; i<4; i++)
    {
        for (int j=0; j<4; j++)
        {
            is << Proj.matrix()(i,j) << " " ;

            if (i==3 && j==3)
                is << ";";
            else
                is << ",";
        }
        is << endl;
    }
}

int main(int argc, char *argv[])
{
    int width=600;
    int height=600;
    double zNear=0.1;
    double zFar=100.0;

    std::vector<Vector3d> m = loadImages(std::string(argv[1]));
    std::vector<Vector4d> M = loadWorldCoords(std::string(argv[2]));

    CameraDirectLinearTransformation cam(m,M,true,true,0,0,width,height,zNear,zFar);
    Eigen::Matrix<double,3,4> P =  cam.getProjectionMatrix();

    cout << "Projection matrix=\n" << cam.getProjectionMatrix() << endl;
    cout << "Intrinsinc matrix=\n" <<cam.getIntrinsicMatrix() << endl;
    cout << "Rotation matrix=\n"<< cam.getRotationMatrix() << endl << endl;

    cout << "Reproduction error= " << cam.getReprojectionError(P,m,M) << endl;
    cout << "Reproduction errorGL=" << cam.getReproductionErrorOpenGL(cam.getOpenGLProjectionMatrix(),cam.getOpenGLModelViewMatrix(),Vector4i(0,0,width,height),m,M)  << endl;
    //cout << "Principal point=" << cam.getPrincipalPoint().transpose() << endl;
    //cout << "Principal axis= " << cam.getPrincipalAxis().transpose() << endl;
    writeToFile(cout,cam.getOpenGLModelViewMatrix(),cam.getOpenGLProjectionMatrix());

    cerr << "Principal point=\n" << cam.getPrincipalPoint() << endl ;
    cerr << "OpenGL ModelView=\n" << cam.getOpenGLModelViewMatrix().matrix() << endl;
    cerr << "OpenGL Projection=\n" << cam.getOpenGLProjectionMatrix().matrix() << endl;

    cout << "OpenGL ModelViewProjection=\n" << cam.getOpenGLModelViewProjectionMatrix().matrix() << endl;

    //cout << cam.getModelViewProjectionMatrix().matrix() << endl;

    return 0;
}
