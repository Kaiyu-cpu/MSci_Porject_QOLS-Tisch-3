This is the GitHub repository of MSci Project (QOLS-Tisch-3) conducted by Kaiyu Hu and Chao Fan, under supervision of Prof John Tisch from Imperial College London.

Aim: Machine learning to optimise the fringe pattern of the MZ interferometer. 


Module Description:

camera_reader: read image from camera. Crop the image to the central 64*64 pixels.

fringe_analysis: take the fringe image from the camera and calculate the visibility from it using Fourier Transform.

main: run different optimisation algorithms to maximise the visibility of the fringes. The fitness function is defined as $$V - log(1-V) -1$$

Action table:

29500798 -- Beam Splitter -- Horizontal

29501050 -- Beam Splitter -- Vertical

29500795 -- Mirror -- Horizontal

29500948 -- Mirror -- Vertical
