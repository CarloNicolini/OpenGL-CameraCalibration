OpenGL-CameraCalibration
========================

A C++ class to help computation of OpenGL projection and modelview matrices starting from points 2D-3D correspondencies (homography).

I heavily rely on Eigen for efficient matrix operation as well as Singular Value Decomposition.

While this code can be made faster, I wanted to keep the implementation as clear as possible, since I had bad times in understanding of 
to correclty change coordinate system between OpenGL and Zisserman book. There was no clear implementation of it that explicitly show
the single steps to be done, they all rely on OpenCV automagic to do that. I think that my simple class can clarify much of the steps taken.

Once you get the GL Projection matrix and GL modelview matrix you can load them via
glLoadMatrixd command of OpenGL, being sure to specify the matrix mode or alternatively, can pass those matrices
to your shader as mat4.

glMatrixMode(GL_PROJECTION);
glLoadMatrixd(cam.getOpenGLProjectionMatrix().matrix().data());
glMatrixMode(GL_MODELVIEW);
glLoadMatrixd(cam.getOpenGLModelViewMatrix().matrix().data());

// Draw your geometry


References:
http://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
http://my.safaribooksonline.com/book/-/9781449341916/4dot4-augmented-reality/id2706803
http://urbanar.blogspot.it/2011/04/from-homography-to-opengl-modelview.html
http://cvrr.ucsd.edu/publications/2008/MurphyChutorian_Trivedi_CVGPU08.pdf

Zisserman and Hartley - Multiple view geometry in computer vision



