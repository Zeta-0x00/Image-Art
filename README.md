# Image-Art
 IA-Project1 - Image Art, ITCR, 2023

# How to run the program
## 1. Install dependencies
```haskell
pip install -r requirements.txt
```
## 2. Helo & Usage
```haskell
usage: main.py [-h] -i INPUT -o OUTPUT [-t]

Image processing script

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input image file path
  -o OUTPUT, --output OUTPUT
                        Output image file name
  -t, --time            Print the start time and end time with timestap
  ```
## 2. Run the program

```haskell
python main.py -i <image input> -o <image result> -t
```
## 3. Example
```haskell
python main.py -i images/saber2.png -o saber2_result.png -t
```
## 4. Results
### 4.1 Original image
![alt text](img/saber2.jpg)
<p><center>Figure 1. Original image</center></p>

![alt text](img/result_saber2_fitness1.jpg)
<p><center>Figure 2. Result image using the 2nd fitness implementation</center></p>