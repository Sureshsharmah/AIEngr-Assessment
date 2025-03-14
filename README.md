# AIEngr-Assessment

CV-Pipe-Removal-Project
The goal of this project is to develop an algorithm that removes all connecting lines (pipes) from a technical engineering diagram while preserving the symbols (equipment, sensors, etc.).

Approach
This project uses OpenCV and image processing techniques such as:
Edge detection (Canny, Sobel) to detect lines
Morphological operations (dilation, erosion) to remove pipes while preserving symbols
Contour analysis to differentiate between symbols and pipes

How to Run the Code 
Clone the repository: 
git clone https://github.com/Sureshsharmah/AIEngr-Assessment.git
cd AIEngr-Assessment/CV-Pipe-Removal-Project

Install dependencies:
pip install -r requirements.txt
python main.py

The processed images will be saved in the results/ folder.

Before-and-After Images 
The input and output images are stored in the results/ folder:

Input: results/input_image.jpg
Output: results/output_image.jpg

Watch the full explanation and demo here:
https://youtu.be/i4qmOMk3PBI

Thank You
