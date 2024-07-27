# Filter Bank and Image Processing

This repository contains a Python script that generates various filter banks (DoG, LM, Gabor, and Half-Disk) and applies them to images to create texton maps, brightness maps, and texture maps. It also calculates gradients and visualizes the results.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

## Overview

The script performs several tasks related to filter banks and image processing, including:
- Generating different types of filters (DoG, LM, Gabor, Half-Disk)
- Applying these filters to images
- Calculating gradients and creating texton maps
- Visualizing filters and processing results

## Installation

To run this script, you need Python 3 and the following Python packages:
- `numpy`
- `opencv-python`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `imutils`

You can install these packages using pip:

```bash
pip install numpy opencv-python scipy scikit-learn matplotlib imutils
```

## Usage
* Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

* Place your input image in the same directory as the script and name it input.jpg.

* Run the script:

```bash
python script_name.py
```

* The script will generate and save the following outputs:

1. Filter visualizations (DoG_filters.png, LM_filters.png, Gabor_filters.png, Half_Disk_filters.png)
2. Texton map (Texton_Map.png)
3. Gradients (Gradient_0.png, Gradient_1.png, Gradient_2.png)

