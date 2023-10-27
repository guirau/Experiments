# Ask the PDF
## Project: Flower Image Classifier

Inside this directory you will find one Jupyter notebook:

- **notebook.ipynb**: Complete project from the Udacity ML with TensorFlow course.

And two Python scripts:

- **predict.py**: Command-line app that uses a pre-trained network to predict the top flower names from an image along with their corresponding probabilities.
- **utility.py**: Supporting modules for `predict.py`.

### Rendered HTML view from nbviewer

- **image_classifier.ipynb**: [https://nbviewer.org/github/guirau/Udacity/blob/main/ML%20with%20TensorFlow/Flower%20Image%20Classifier/Project_Image_Classifier_Project.ipynb](https://nbviewer.org/github/guirau/Udacity/blob/main/ML%20with%20TensorFlow/Flower%20Image%20Classifier/Project_Image_Classifier_Project.ipynb)

---

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Matplotlib](http://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Hub](https://tfhub.dev/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Pillow (PIL Fork)](https://pillow.readthedocs.io/en/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Run

In a terminal or command window, navigate to the top-level project directory `image_classifier/` (that contains this README) and run one of the following commands:

```bash
ipython notebook image_classifier.ipynb
```  
or
```bash
jupyter notebook image_classifier.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Code

The `predict.py` module should predict the top flower names from an image along with their corresponding probabilities.

**Basic usage**:

```bash
python predict.py /path/to/image saved_model
```

**Options**:

- `--top_k` or `-k`: Return the top K most likely classes:

```bash
python predict.py /path/to/image saved_model --top_k K
```

- `--category_names` or `-c`: Path to a JSON file mapping labels to flower names:

```bash
python predict.py /path/to/image saved_model --category_names map.json
```

- `--verbose` or `-v`: Show verbose information.

### App
