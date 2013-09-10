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
#include <stdexcept>
#include "CameraDirectLinearTransformation.h"

CameraDirectLinearTransformation::CameraDirectLinearTransformation(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector4d> &X, bool decomposeProjectionMatrix, bool computeOpenGLMatrices, double x0, double y0, int width, int height, double znear, double zfar)
    : ModelViewProjectionInitialized(false),DecompositionComputed(false)
{
    this->init(x,X,decomposeProjectionMatrix,computeOpenGLMatrices,x0,y0,width,height,znear,zfar);
}

CameraDirectLinearTransformation::CameraDirectLinearTransformation(const string &imagesFileName, const string &worldCoordsFileName, bool decomposeProjectionMatrix, bool computeOpenGLMatrices, double x0, double y0, int width, int height, double znear, double zfar)
{
    std::vector<Eigen::Vector3d> x = this->loadImages(imagesFileName);
    std::vector<Eigen::Vector4d> X = this->loadWorldCoords(worldCoordsFileName);
    this->init(x,X,decomposeProjectionMatrix,computeOpenGLMatrices,x0,y0,width,height,znear,zfar);
}

void CameraDirectLinearTransformation::init(const std::vector<Vector3d> &x, const std::vector<Vector4d> &X, bool decomposeProjectionMatrix, bool computeOpenGLMatrices, double x0, double y0, int width, int height, double znear, double zfar)
{
    if (x.size() != X.size() )
        throw std::logic_error("There must be the same number of correspondencies");

    // Reset all the matrices used in the class
    this->K.setZero();
    this->P.setZero();
    this->R.setZero();
    this->t.setZero();
    this->C.setZero();
    this->OpenGLModelViewMatrix.setIdentity();
    this->OpenGLProjectionMatrix.matrix().setZero();
    this->OpenGLModelViewProjectionMatrix.matrix().setZero();

    unsigned int n=x.size();
    Eigen::MatrixXd A;
    A.setZero(2*n,12);

    for ( unsigned int i=0; i<n; i++ )
    {
        Vector3d m = x.at(i);
        RowVector4d M = X.at(i).transpose();
        A.row(2*i) << 0,0,0,0, -m[2]*M, m[1]*M;
        A.row(2*i+1) << m[2]*M,0,0,0,0, -m[0]*M;
    }

    // ofstream matrixA("A.txt");
    // http://my.safaribooksonline.com/book/-/9781449341916/4dot4-augmented-reality/id2706803
    // matrixA << A << endl;

    JacobiSVD<MatrixXd> svd(A, ComputeFullV );
    // Copy the data in the last column of matrix V (the eigenvector with the smallest eigenvalue of A^T*A)
    // maintaining the correct order
    int k=0;
    for (int i=0; i<3;i++)
    {
        for (int j=0; j<4; j++)
        {
            double t = svd.matrixV().col(11).coeffRef(k);
            P(i,j)= t;
            ++k;
        }
    }


    //When putting these values in the projection matrix:
    /*
    P.row(0) << 3.53553E2, 3.39645E2, 2.77744E2, -1.44946E6;
    P.row(1) << -1.03528E2, 2.33212E1, 4.59607E2, -6.32525E5;
    P.row(2) << 7.07107E-1, -3.53553E-1, 6.12372E-1, -9.18559E2;
*/

    //one should obtain the following values:
    /*
Intrinsinc camera matrix=
   0.0144213 -0.000328564   0.00343275
 -1.0842e-19   0.00709935   0.00254141
          -0           -0  8.54968e-06

Extrinsic camera matrix=
 0.0887467 -0.0363712   -0.99539
  0.994285  0.0627698  0.0863546
 0.0593396  -0.997365   0.041734

Camera Center C=-16.6029   255.09 -20.1428

Camera T= -9.29854 2.23552 256.243

Camera Principal axis= 0.0593396 -0.997365  0.041734

Camera Principal point= 401.506 297.252
OpenGL ModelView=
 0.0887467  0.0363712    0.99539   -9.29854
  0.994285 -0.0627698 -0.0863546    2.23552
 0.0593396   0.997365  -0.041734    256.243
         0          0          0          1

OpenGL Projection=
3.83545e-05 8.73841e-07    0.999991           0
          0 2.95806e-05   -0.999989           0
          0           0     -1.0002    -0.20002
          0           0          -1           0

OpenGL ModelViewProjection=
 0.0593433   0.997357 -0.0416955    256.241
-0.0593095  -0.997356   0.041731   -256.241
-0.0593514  -0.997565  0.0417423   -256.495
-0.0593396  -0.997365   0.041734   -256.243
*/

    if (decomposeProjectionMatrix)
    {
        this->decomposePMatrix(this->P);
        this->DecompositionComputed=true;
        if (computeOpenGLMatrices)
        {
            this->computeOpenGLProjectionMatrix(x0,y0,width,height,znear,zfar);
            this->computeOpenGLModelViewMatrix(this->R, this->t);

            this->OpenGLModelViewProjectionMatrix = OpenGLProjectionMatrix* OpenGLModelViewMatrix;
            this->ModelViewProjectionInitialized=true;
        }
    }
}

void CameraDirectLinearTransformation::decomposePMatrix(const Eigen::Matrix<double,3,4> &P)
{
    Matrix3d M = P.topLeftCorner<3,3>();
    Vector3d m3 = M.row(2).transpose();
    // Follow the HartleyZisserman - "Multiple view geometry in computer vision" implementation chapter 3

    Matrix3d P123,P023,P013,P012;
    P123 << P.col(1),P.col(2),P.col(3);
    P023 << P.col(0),P.col(2),P.col(3);
    P013 << P.col(0),P.col(1),P.col(3);
    P012 << P.col(0),P.col(1),P.col(2);

    double X = P123.determinant();
    double Y = -P023.determinant();
    double Z = P013.determinant();
    double T = -P012.determinant();
    C << X/T,Y/T,Z/T;

    // Image Principal points computed with homogeneous normalization
    this->principalPoint = (M*m3).eval().hnormalized().head<2>();

    // Principal vector  from the camera centre C through pp pointing out from the camera.  This may not be the same as  R(:,3)
    // if the principal point is not at the centre of the image, but it should be similar.
    this->principalVector  =  (M.determinant()*m3).normalized();
    this->R = this->K = Matrix3d::Identity();
    this->rq3(M,this->K,this->R);
    K/=K(2,2); // EXTREMELY IMPORTANT TO MAKE THE K(2,2)==1 !!!
    // http://ksimek.github.io/2012/08/14/decompose/
    // Negate the second column of K and R because the y window coordinates and camera y direction are opposite is positive
    // This is the solution I've found personally to correct the behaviour using OpenGL gluPerspective convention
    this->R.row(2)=-R.row(2);

    // t is the location of the world origin in camera coordinates.
    t = -R*C;
}

/**
 * @brief CameraDirectLinearTransformation::computeOpenGLProjectionMatrix
 * @param x0
 * @param y0
 * @param width
 * @param height
 * @param znear
 * @param zfar
 * @param windowCoordsYUp
 */
void CameraDirectLinearTransformation::computeOpenGLProjectionMatrix(double x0,double y0,double width,double height,double znear,double zfar,bool windowCoordsYUp)
{
    eigen_assert(DecompositionComputed && "You did not asked for the P = K[R t] matrix decomposition, explicitly ask in constructor");
    double depth = zfar - znear;
    double q =  -(zfar + znear) / depth;
    double qn = -2.0 * (zfar * znear) / depth;
    // This follows the OpenGL convention where positive Y coordinates goes down
    if (windowCoordsYUp)
    {
        this->OpenGLProjectionMatrix.matrix() <<  2.0*K(0,0)/width, -2.0*K(0,1)/width, (-2.0*K(0,2)+width+2.0*x0)/width, 0 ,
                0,             -2.0*K(1,1)/height,(-2.0*K(1,2)+height+2.0*y0)/height, 0,
                0,0,q,qn,
                0,0,-1,0;
    }
    else // y_down convention
    {
        this->OpenGLProjectionMatrix.matrix() << 2.0*K(0,0)/width, -2.0*K(0,1)/width, (-2.0*K(0,2)+width+2.0*x0)/width, 0 ,
                0,              2.0*K(1,1)/height,( 2.0*K(1,2)-height+2.0*y0)/height, 0,
                0,0,q,qn,
                0,0,-1,0;
    }
}

/**
 * @brief CameraDirectLinearTransformation::getT
 * @return The location of the world origin in camera coordinates.
 *  The sign of tx,ty,tz should reflect where the world origin appears in the camera (left/right of center, above/below center, in front/behind camera, respectively).
 */
const Eigen::Vector3d & CameraDirectLinearTransformation::getT()
{
    return t;
}

/**
 * @brief CameraDirectLinearTransformation::rq3
 * Perform 3 RQ decomposition of matrix A and save them in matrix R and matrix Q
 * http://www.csse.uwa.edu.au/~pk/research/matlabfns/Projective/rq3.m
 * @param A
 * @param R
 * @param Q
 */
void CameraDirectLinearTransformation::rq3(const Matrix3d &A, Matrix3d &R, Matrix3d& Q)
{
    // Find rotation Qx to set A(2,1) to 0
    double c = -A(2,2)/sqrt(A(2,2)*A(2,2)+A(2,1)*A(2,1));
    double s = A(2,1)/sqrt(A(2,2)*A(2,2)+A(2,1)*A(2,1));
    Matrix3d Qx,Qy,Qz;
    Qx << 1 ,0,0, 0,c,-s, 0,s,c;
    R = A*Qx;
    // Find rotation Qy to set A(2,0) to 0
    c = R(2,2)/sqrt(R(2,2)*R(2,2)+R(2,0)*R(2,0) );
    s = R(2,0)/sqrt(R(2,2)*R(2,2)+R(2,0)*R(2,0) );
    Qy << c, 0, s, 0, 1, 0,-s, 0, c;
    R*=Qy;

    // Find rotation Qz to set A(1,0) to 0
    c = -R(1,1)/sqrt(R(1,1)*R(1,1)+R(1,0)*R(1,0));
    s =  R(1,0)/sqrt(R(1,1)*R(1,1)+R(1,0)*R(1,0));
    Qz << c ,-s, 0, s ,c ,0, 0, 0 ,1;
    R*=Qz;

    Q = Qz.transpose()*Qy.transpose()*Qx.transpose();
    // Adjust R and Q so that the diagonal elements of R are +ve
    for (int n=0; n<3; n++)
    {
        if (R(n,n)<0)
        {
            R.col(n) = - R.col(n);
            Q.row(n) = - Q.row(n);
        }
    }
}

/**
 * @brief CameraDirectLinearTransformation::computeOpenGLModelViewMatrix
 * This matrix describes how to transform points in world coordinates to camera coordinates.
 * The vector t can be interpreted as the position of the world origin in camera coordinates,
 * and the columns of R represent the directions of the world-axes in camera coordinates.
 * @param Rot
 * @param trans
 */
void CameraDirectLinearTransformation::computeOpenGLModelViewMatrix(const Eigen::Matrix3d &Rot, const Vector3d &trans)
{
    this->OpenGLModelViewMatrix.setIdentity();
    this->OpenGLModelViewMatrix.linear().matrix() << Rot;
    this->OpenGLModelViewMatrix.translation() << trans;
  /*
    Eigen::Affine3d CoordXForm = Eigen::Affine3d::Identity();
    CoordXForm.matrix().coeffRef(1,1)=-1; // flip Y coordinate in eye space (OpenGL has +Y as up, Hartley Zisserman has -Y)
    CoordXForm.matrix().coeffRef(2,2)=-1; // flip Z coordinate in eye space (OpenGL has -Z in front of camera, Hartley Zisserman has +Z)
    */
    //CoordXForm.linear() = Eigen::AngleAxis<double>(M_PI,Vector3d::UnitX()).toRotationMatrix();
    //this->OpenGLModelViewMatrix.matrix()*=CoordXForm.matrix();
}

/**
 * @brief CameraDirectLinearTransformation::getCameraCenter
 * @return
 */
const Eigen::Vector3d &CameraDirectLinearTransformation::getCameraCenter() const
{
    eigen_assert(DecompositionComputed && "You did not asked for the P matrix decomposition, explicitly ask in constructor");
    return this->C;
}

/**
 * @brief CameraDirectLinearTransformation::loadImages Load the projected pixel coordinates points
 * @param filename
 * @return The pixel coordinates homogenized (3D because (u,v,1 ) )
 */
std::vector<Vector3d> CameraDirectLinearTransformation::loadImages(const string &filename)
{
    std::ifstream is;
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
 * @brief CameraDirectLinearTransformation::loadWorldCoords Load the points of the 3D world
 * @param filename
 * @return The world coordinates in homogenized format (x,y,z,1)
 */
std::vector<Vector4d> CameraDirectLinearTransformation::loadWorldCoords(const string &filename)
{
    std::ifstream is;
    is.open(filename.c_str());
    std::vector<Vector4d> vals;
    double x,y,z;
    while( (is >> x ) &&  (is >> y) && (is >> z ) )
    {
        vals.push_back(Vector4d(x,y,z,1.0));
    }
    return vals;
}

/**
 * @brief getReprojectionError
 * @param P
 * @param x
 * @param X
 * @return
 */
double CameraDirectLinearTransformation::getReprojectionError(const Eigen::Matrix<double,3,4> &P, const vector<Vector3d> &x, const vector<Vector4d> &X)
{
    int n=x.size();
    double error=0.0;
    for (int i=0; i<n;i++)
    {
        Vector3d mr = P*X.at(i);
        mr/=mr(2);  // switch back to cartesian coordinates
        double d = ( Vector2d(mr.x(),mr.y()) - Vector2d(x.at(i).x(),x.at(i).y())).norm();
        error+=d;
    }
    error/=n;
    return error;
}

/**
 * @brief getReproductionError
 * @param P
 * @param MV
 * @param Viewport
 * @param x
 * @return
 */
double CameraDirectLinearTransformation::getReproductionErrorOpenGL(const Eigen::Projective3d &P, const Eigen::Affine3d &MV, const Vector4i &viewport, const vector<Vector3d> &x, const vector<Vector4d> &X)
{
    int n=x.size();
    double error=0.0;
    for (int i=0; i<n;i++)
    {
        Vector3d point = X.at(i).head<3>();
        Vector3d v = ( P*(MV*point).homogeneous() ).eval().hnormalized();
        Vector2d vPixel(viewport(0) + viewport(2)*(v.x()+1)/2,viewport(1) + viewport(3)*(v.y()+1)/2);

        cout << i << "-->" << vPixel.transpose() << " " << x.at(i).head<2>().transpose() << endl;
        error += ( vPixel- x.at(i).head<2>() ).norm();
    }
    error/=n;
    return error;
}

/*
void CameraDirectLinearTransformation::decomposePMatrix2(const Eigen::Matrix<double,3,4> &_P)
{
    // Normalized Projection Matrix
    Matrix<double,3,4> P=_P;
    P/= _P.row(2).head<3>().norm();

    this->R = this->K = Matrix3d::Zero();
    // Principal Point
    K(0,2) = P.row(0).head<3>().dot(P.row(2).head<3>()); //P(0,0:2)*P(2,0:2)';
    K(1,2) = P.row(1).head<3>().dot(P.row(2).head<3>()); // P(1,0:2)*P(2,0:2)';
    this->principalPoint << K(0,2),K(1,2) ;
    // Focal Length
    K(0,0)   = sqrt(pow(P.row(0).head<3>().norm(),2) - K(0,2)*K(0,2));
    K(1,1)   = sqrt(pow(P.row(1).head<3>().norm(),2) - K(1,2)*K(1,2));
    K.row(2) << 0, 0, 1;

    // Rotation Matrix
    R(2,0)   = P(2,0);
    R(2,1)   = P(2,1);
    R(2,2)   = P(2,2);
    R.row(0) << (P.row(0).head<3>()-K(0,2)*P.row(2).head<3>())/K(0,0);
    R.row(1) << (P.row(1).head<3>()-K(1,2)*P.row(2).head<3>())/K(1,1);

    if (K(0,0)<0)
    {
        Matrix3d J=Matrix3d::Identity();
        J(0,0)=-1;
        K*=J;
        R*=J;
    }
    if (K(1,1)>0)
    {
        Matrix3d H=Matrix3d::Identity();
        H(1,1)=-1;
        K*=H;
        R*=H;
    }

    // Translation Vector
    this->t(2)   = P(2,3);
    this->t(0)   = (P(0,3)-K(0,2)*t(2))/K(0,0);
    this->t(1)   = (P(1,3)-K(1,2)*t(2))/K(1,1);
    // t can also be computed by mean of this formula
    Vector3d t = K.inverse()*P.col(3).head<3>();

    // Orthogonality Enforcement
    JacobiSVD<Matrix3d> svd(R,ComputeFullU | ComputeFullV);
    R=svd.matrixU()*Matrix3d::Identity()*svd.matrixV().transpose();

    // XXX carlo
    this->principalVector << R.col(2);
    // Camera position in world coordinates
    this->C = -R.transpose()*t;
    //cout << "CameraPosWorld=\n" << C.transpose() << endl;

    // Transform some signs to fit OpenGL conventions
    this->t.tail<2>() =-this->t.tail<2>();
    //cout << "t=\n" << t.transpose() << endl;

    R.row(2)=-R.row(2);
    R.col(1)=-R.col(1);
}
*/
