# Swab Counter

A Python project for counting swabs with only CV(Non-AI)

---

## ✨ Algorithm
![Overview](/attached/overview.png)

### Resize & Masking
![Mask](/attached/mask_img.jpg)

### Adjust HSV
![HSV](/attached/hsv_correction_img.jpg)

### Remove Edge
![Remove Edge](/attached/fine_img.jpg)

### Gaussian Blur & binarization (init)
![Blur1](/attached/blurred_image.jpg)
![binarization1](/attached/background.jpg)

### Skeletonization
![skeleton](/attached/skeleton.jpg)

### Gaussian Blur & binarization (refinement)
![Blur2](/attached/blurred_skeleton.jpg)
![binarization2](/attached/background2.jpg)

### Labeling (init & refinement)
![labeling](/attached/final_image1.jpg)

### Final
![final](/attached/final_image2.jpg)



---

## 🛠️ Tech Stack

-   Python
-   OpenCV

---

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

**1. Clone the Repository**
```bash
git clone https://github.com/spilak-stack/swab_counter.git
cd swab_counter
```

**2. Create and Activate Virtual Environment**
> **Note**: This project has been tested in a Python 3.11 environment.

```bash
# Create a virtual environment using uv
uv venv -p python3.11

# Activate the virtual environment (Windows PowerShell)
.venv\Scripts\activate
# source .venv/bin/activate
```

**3. Install Dependencies**
Install all required packages using the `requirements.txt` file.
```bash
uv pip install -r requirements.txt
```

**4. Run the Program**
```bash
python main.py
```
---

### ***Actually, We don't need this process, and I think it will work in any environment with opencv***



## 📄 License


This project is licensed under the MIT License.
