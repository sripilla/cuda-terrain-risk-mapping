# cuda-terrain-risk-mapping
High-performance CUDA-based terrain risk mapping system that analyzes grayscale heightmaps using separable convolution, generates cost-based navigation maps, and visualizes terrain safety using color-coded outputs, with detailed CPU vs GPU benchmarking.

## 🚀 Features
- CUDA-based parallel processing
- Separable convolution (row + column)
- CPU vs GPU performance comparison
- Color-coded terrain visualization

## 🧠 Methodology
- Input image represents terrain elevation
- Convolution is applied to detect slope
- Cost is assigned based on gradient magnitude:
  - Green → Safe
  - Yellow → Moderate
  - Red → High Risk
  - Black → Obstacles

## ⚡ Performance
- GPU provides significant speedup over CPU
- Uses parallel execution and optimized memory access

## 🖼️ Input
- Grayscale image (`input.png`)

## 📤 Output
- Risk map image (`output.png`)
- Performance metrics (console + CSV)

## 🛠️ Technologies Used
- CUDA (NVIDIA GPU computing)
- C/C++
- stb_image for image handling

## 📌 Note
Minor differences between CPU and GPU results are expected due to separable convolution and floating-point precision.
