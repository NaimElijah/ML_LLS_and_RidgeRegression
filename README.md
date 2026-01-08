# Linear Least Squares & Ridge Regression

This project demonstrates **Linear Least Squares (LLS)** and **Ridge Regression** on a real dataset, focusing on numerical stability, generalization, and the effect of regularization.

The implementation is written in **Python** and follows standard machine‑learning practice taught in introductory ML / optimization courses.

---

## 📁 Project Structure

```
LLS_and_RidgeRegression/
├── RR_and_LLS.py        # Main script: LLS and Ridge Regression implementation
├── lsdata.mat           # Dataset (MATLAB format)
└── README.md            # Project documentation
```

---

## 🧠 Concepts Covered

* Linear Least Squares (closed‑form solution)
* Ridge Regression (L2 regularization)
* Overfitting vs. regularization
* Training error vs. test error
* Numerical stability and matrix invertibility

---

## 📊 Dataset

The dataset is provided in `lsdata.mat` and contains:

* Feature matrix **X**
* Target vector **y**
* Separate **training** and **test** splits

The script loads the dataset using `scipy.io.loadmat`.

---

## ⚙️ Requirements

Only standard scientific Python packages are used:

```text
numpy
scipy
matplotlib
```

(Optional: `cvxopt` if extended with optimization‑based solvers.)

---

## ▶️ How to Run

```bash
python RR_and_LLS.py
```

The script will:

1. Load the dataset
2. Train a Linear Least Squares model
3. Train Ridge Regression models with different λ values
4. Compare training and test errors
5. Visualize the results

---

## 🧮 Mathematical Formulation

### Linear Least Squares

Given:

* Feature matrix (X \in \mathbb{R}^{n \times d})
* Target vector (y \in \mathbb{R}^n)

The LLS solution is:

[ w = (X^T X)^{-1} X^T y ]

> ⚠️ If (X^T X) is **not invertible**, numerical techniques such as:
>
> * pseudo‑inverse (`np.linalg.pinv`)
> * Ridge regularization
>   are required.

---

### Ridge Regression

Adds L2 regularization:

[ w = (X^T X + \lambda I)^{-1} X^T y ]

Where:

* (\lambda > 0) controls the strength of regularization

This improves:

* numerical stability
* generalization performance

---

## 📈 Results & Observations

* **LLS** fits training data very well but may overfit
* **Ridge Regression** slightly increases training error
* Test error often **decreases** for a well‑chosen λ
* Larger λ → simpler model, higher bias

This illustrates the **bias–variance trade‑off**.

---

## 🧪 Training vs. Test Error

* **Training error**: error on data the model was trained on
* **Test error**: error on unseen data

A growing gap between them usually indicates **overfitting**.

---

## 🧩 Extending the Project

Possible extensions:

* Cross‑validation for λ selection
* Optimization‑based solvers (gradient descent)
* Feature normalization
* Polynomial feature expansion

---

## 📌 Notes

* Code is written for clarity and educational value
* Matrix operations are explicit to match lecture material
* Regularization is introduced as a principled fix, not a hack

---

## 📜 License

This project is intended for portfolio and educational use.
