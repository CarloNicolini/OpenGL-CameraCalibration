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

#ifndef _HOMOGRAPHY_H_
#define _HOMOGRAPHY_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/SVD>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Vector4d)
/**
 * @brief The CameraDirectLinearTransformation class
 */
class CameraDirectLinearTransformation
{
public:
    CameraDirectLinearTransformation(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector4d> &X, bool decomposeProjectionMatrix=false, bool computeOpenGLMatrices=false, double x0=0.0, double y0=0.0, int width=640, int height=480, double znear=0.1, double zfar=1000.0);

        CameraDirectLinearTransformation(const std::string &imagesFileName, const std::string &worldCoordsFileName, bool decomposeProjectionMatrix=false, bool computeOpenGLMatrices=false, double x0=0.0, double y0=0.0, int width=640, int height=480, double znear=0.1, double zfar=1000.0);

        void init(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector4d> &X, bool decomposeProjectionMatrix=false, bool computeOpenGLMatrices=false, double x0=0.0, double y0=0.0, int width=640, int height=480, double znear=0.1, double zfar=1000.0);

    std::vector<Vector3d> loadImages(const string &filename);
    std::vector<Vector4d> loadWorldCoords(const string &filename);

    double getReprojectionError(const Eigen::Matrix<double,3,4> &P, const vector<Vector3d> &x, const vector<Vector4d> &X);
    double getReproductionErrorOpenGL(const Eigen::Projective3d &P, const Eigen::Affine3d &MV, const Vector4i &viewport, const vector<Vector3d> &x, const vector<Vector4d> &X);

    const Eigen::Vector3d &getCameraCenter() const;
    // For backward compatibility
    const Eigen::Vector3d &getCameraPositionWorld() const
    {
        return getCameraCenter();
    }

    void computeOpenGLProjectionMatrix(double x0, double y0, double width, double height, double znear, double zfar, bool windowCoordsYUp=false);

    void computeOpenGLModelViewMatrix(const Eigen::Matrix3d &Rot, const Vector3d &trans);

    void decomposePMatrix(const Eigen::Matrix<double,3,4> &P);

    const Eigen::Matrix<double,3,4> &getProjectionMatrix()
    {
        return this->P;
    }

    const Eigen::Matrix3d &getIntrinsicMatrix()
    {
        eigen_assert(DecompositionComputed && "You did not asked for the P matrix decomposition, explicitly ask in constructor");
        return this->K;
    }

    const Eigen::Matrix3d &getRotationMatrix()
    {
        eigen_assert(DecompositionComputed && "You did not asked for the P = K[R,t] matrix decomposition, explicitly ask in constructor");
        return R;
    }

    const Eigen::Affine3d &getOpenGLModelViewMatrix()
    {
        eigen_assert(DecompositionComputed && "You did not asked for the P = K[R t] matrix decomposition, explicitly ask in constructor");
        eigen_assert(ModelViewProjectionInitialized && "You did not asked for the OpenGL ModelViewProjection matrix to be computed, explicitly ask in constructor and provide appropriate parameters");
        return this->OpenGLModelViewMatrix;
    }

    const Eigen::Projective3d &getOpenGLProjectionMatrix()
    {
        eigen_assert(DecompositionComputed && "You did not asked for the P = K[R t] matrix decomposition, explicitly ask in constructor");
        eigen_assert(ModelViewProjectionInitialized && "You did not asked for the OpenGL ModelViewProjection matrix to be computed, explicitly ask in constructor and provide appropriate parameters");
        return this->OpenGLProjectionMatrix;
    }

    const Eigen::Vector3d & getT();

    const Eigen::Projective3d &getOpenGLModelViewProjectionMatrix()
    {
        eigen_assert(DecompositionComputed && "You did not asked for the P = K[R t] matrix decomposition, explicitly ask in constructor");
        eigen_assert(ModelViewProjectionInitialized && "You did not asked for the OpenGL ModelViewProjection matrix to be computed, explicitly ask in constructor and provide appropriate parameters");
        return this->OpenGLModelViewProjectionMatrix;
    }

    const Vector2d &getPrincipalPoint()
    {
        eigen_assert(DecompositionComputed && "You did not asked for the P = K[R t] matrix decomposition, explicitly ask in constructor");
        return this->principalPoint;
    }

    const Vector3d &getPrincipalAxis()
    {
        eigen_assert(DecompositionComputed && "You did not asked for the P = K[R t] matrix decomposition, explicitly ask in constructor");
        return this->principalVector;
    }

    // OLD DEPRECATED, I HAVE CHECKED THE RESULTS WITH THE Hartley Zisserman and with the Andrew Straw implementation too
    //void decomposePMatrix2(const Eigen::Matrix<double,3,4> &_P);

private:

    void rq3(const Matrix3d &A, Matrix3d &R, Matrix3d& Q);
    /**
     * @brief P The matrix x = P X that projects 3D world homogenous points to their images
     */
    Eigen::Matrix<double,3,4> P;
    /**
     * @brief ModelViewProjectionMatrix The OpenGL matrix computed starting from P
     */
    Eigen::Projective3d OpenGLModelViewProjectionMatrix;
    /**
     * @brief R The orthogonal rotation matrix obtained decomposing P, it's part of the extrinsic pose P= K [R t]
     */
    Eigen::Matrix3d R;
    /**
     * @brief K The intrinsic matrix P = K [R t]
     */
    Eigen::Matrix3d K;
    /**
     * @brief t The camera center in camera coordinates
     */
    Eigen::Vector3d C;

    /**
     * @brief t
     */
    Eigen::Vector3d t;

    /**
     * @brief principalPoint The principal point
     */
    Vector2d principalPoint;
    /**
     * @brief principalVector
     */
    Vector3d principalVector;

    Affine3d OpenGLModelViewMatrix;
    Projective3d OpenGLProjectionMatrix;
    bool ModelViewProjectionInitialized;
    bool DecompositionComputed;
};

#endif
