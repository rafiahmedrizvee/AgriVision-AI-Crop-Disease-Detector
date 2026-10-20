# 🌿 AgriVision AI — AI-Powered Crop Disease Detector

**An intelligent crop disease detection system powered by Deep Learning, TensorFlow, FastAPI, and React.**

AgriVision AI is a full-stack AI-powered web application that helps users identify tomato leaf diseases by uploading an image. The system uses a trained deep learning model to analyze the leaf image, predict the most likely disease category, and return prediction confidence scores through a REST API.

The project combines machine learning, backend API development, and modern frontend technologies to demonstrate a practical application of AI in agriculture.

## 🚀 Live Demo

- **Live Application:** [Visit AgriVision AI](YOUR_NETLIFY_URL)
- **Backend API:** [FastAPI Backend](YOUR_RENDER_URL)
- **API Documentation:** [Explore API Docs](YOUR_RENDER_URL/docs)

> Replace the placeholder URLs with your actual Netlify and Render deployment URLs.

## ✨ Key Features

- 🌱 **AI-Powered Disease Detection:** Predicts tomato leaf diseases using a trained TensorFlow deep learning model.
- 🖼️ **Image Upload:** Upload tomato leaf images for AI-based analysis.
- 🎯 **Prediction Confidence:** Displays confidence scores for the predicted class and other disease categories.
- ⚡ **Fast REST API:** Uses FastAPI to handle image uploads and model inference.
- 💻 **Modern User Interface:** Built with React and Vite for a responsive and interactive user experience.
- 🔗 **Frontend–Backend Integration:** Connects the React application to the FastAPI prediction endpoint.
- 📚 **Interactive API Documentation:** Provides Swagger UI for testing API endpoints.
- ☁️ **Cloud Deployment:** Deployed frontend and backend using Netlify and Render.
- 🧠 **Deep Learning Inference:** Loads a trained `.keras` model to classify uploaded leaf images.

## 🦠 Supported Tomato Leaf Classes

The current model supports four tomato leaf categories:

| Class | Description |
|---|---|
| `Tomato_healthy` | Healthy tomato leaves |
| `Tomato_Bacterial_spot` | Bacterial spot |
| `Tomato_Early_blight` | Early blight |
| `Tomato_Late_blight` | Late blight |

*Note: The application supports only the classes used to train the current model. Predictions are not a substitute for professional agricultural diagnosis.*

## 🛠️ Technology Stack

### Frontend
- React
- Vite
- JavaScript (ES6+)
- HTML5 and CSS3
- Lucide React icons

### Backend
- Python
- FastAPI
- Uvicorn
- REST API
- Pillow for image processing

### AI / Machine Learning
- TensorFlow
- Keras
- NumPy
- Convolutional Neural Network (CNN)
- Image classification
- Transfer learning / deep learning model inference

### Tools and Deployment
- Git and GitHub
- Visual Studio Code
- Netlify — frontend hosting
- Render — backend hosting
- Swagger UI — API testing

## 🏗️ System Architecture

```text
                   ┌──────────────────────┐
                   │       User           │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ React + Vite Frontend│
                   │   Image Upload UI    │
                   └──────────┬───────────┘
                              │
                       HTTP REST API
                              │
                              ▼
                   ┌──────────────────────┐
                   │   FastAPI Backend    │
                   │  Image Preprocessing │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ TensorFlow / Keras   │
                   │ Trained CNN Model    │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Prediction & Scores  │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Results in Frontend  │
                   └──────────────────────┘
```

## 📂 Project Structure

```text
agrivision-ai/
│
├── backend/
│   ├── app/
│   │   └── ...                  # FastAPI application modules
│   ├── models/
│   │   └── tomato_disease_model.keras
│   ├── requirements.txt
│   └── ...                      # Backend configuration and entry point
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── services/            # API integration
│   │   ├── components/          # Reusable UI components
│   │   ├── App.jsx
│   │   └── ...
│   ├── package.json
│   └── ...
│
├── ml/
│   ├── data/
│   │   └── processed/
│   │       └── tomato/
│   └── ...                      # ML training and preprocessing scripts
│
├── .gitignore
└── README.md
```

*The tree is illustrative. Adjust the paths to match the exact files and directories in your repository.*

## ⚙️ Getting Started

Follow these instructions to run AgriVision AI locally.

### Prerequisites

Install the following tools before starting:

- Python 3.13
- Node.js and npm
- Git

A compatible Python environment is recommended for TensorFlow and the project's dependencies.

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd agrivision-ai
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual clone URL.

### 2. Set Up the Backend

Open a terminal in the project root.

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server using the application's actual entry-point module. For example, if the application object is `app` in `app/main.py`:

```bash
uvicorn app.main:app --reload
```

If your entry point differs, replace `app.main:app` with the correct module and application object.

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

### 3. Set Up the Frontend

Open another terminal:

```bash
cd frontend
npm install
```

Configure the frontend API base URL to point to your local FastAPI server. If the application uses a Vite environment variable, configure the appropriate variable in `frontend/.env`, for example:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Use the variable name expected by your existing API service.

Start the frontend:

```bash
npm run dev
```

Open the local application at the URL displayed by Vite, typically:

```text
http://localhost:5173
```

## 🔄 How It Works

1. **Image Upload:** The user uploads a tomato leaf image through the React interface.
2. **API Request:** The frontend sends the image to the FastAPI backend using an HTTP request.
3. **Image Preprocessing:** The backend prepares the image according to the model's expected input format.
4. **Model Inference:** The trained TensorFlow/Keras model predicts the relevant disease category.
5. **Confidence Calculation:** The backend processes the model's output into prediction scores.
6. **Result Display:** The frontend presents the predicted category and confidence information to the user.

## 🧪 Model and Dataset

The model was developed for a multi-class tomato leaf image classification task using a selected subset of the PlantVillage dataset.

### Dataset Preparation

The prepared dataset contains four classes:

| Dataset split | Images |
|---|---:|
| Training | 5,300 |
| Validation | 1,327 |
| **Total** | **6,627** |

The dataset was divided into training and validation sets to support model development and validation.

**Important:** These figures describe the prepared training and validation data. They do not represent independent test-set performance or prove a particular real-world accuracy.

### Model Information

- **Framework:** TensorFlow / Keras
- **Model file:** `tomato_disease_model.keras`
- **Task:** Multi-class image classification
- **Input:** Tomato leaf image
- **Output:** Predicted disease category and confidence scores

### Dataset Source

- [PlantVillage Dataset — GitHub](https://github.com/spMohanty/PlantVillage-Dataset)

Please follow the dataset's licensing and attribution requirements when redistributing data or publishing derived assets.

## ☁️ Deployment

AgriVision AI is designed as a separately deployed frontend and backend application.

| Component | Platform | Purpose |
|---|---|---|
| Frontend | Netlify | Hosts the React web application |
| Backend | Render | Runs the FastAPI inference API |
| Model | Backend model directory | Stores the trained Keras model |

### Production Checklist

- [x] Backend deployment completed
- [x] API health check and documentation verified
- [x] Model prediction endpoint tested
- [x] Frontend and backend integration tested
- [ ] Add actual live URLs to this README
- [ ] Verify production CORS and environment configuration
- [ ] Monitor cold starts, inference latency, and application errors

## 🔐 Production Considerations

For a reliable production application:

- Validate image file types and maximum upload size.
- Handle malformed images and model inference errors gracefully.
- Configure CORS to allow only trusted frontend origins.
- Keep secrets and API credentials out of source control.
- Use logging and monitoring to investigate deployment issues.
- Avoid making disease-treatment recommendations without reliable agricultural validation.
- Evaluate the model on a separate test set and images collected outside the training dataset.

## 🔮 Future Improvements

- Expand support to additional crops and diseases.
- Improve generalization using diverse, real-world field images.
- Add treatment and prevention guidance reviewed by agricultural experts.
- Provide downloadable prediction reports.
- Add prediction history and user accounts.
- Build a mobile-friendly or mobile application experience.
- Introduce model evaluation dashboards and explainability tools.
- Add automated testing and continuous integration / continuous deployment (CI/CD).

## 🎯 Project Objectives

- Apply deep learning to a real-world agricultural problem.
- Build an end-to-end AI application, from model inference to user interface.
- Develop practical skills in TensorFlow, FastAPI, React, and cloud deployment.
- Demonstrate full-stack AI engineering capabilities through a deployable project.

## 👨‍💻 Author

**Mazharul Alam Rafi**

MERN Stack Web Developer & AI Engineer

Interested in AI engineering, machine learning, deep learning, and AI-powered applications.

### Connect With Me

- **GitHub:** [GitHub Profile](https://github.com/rafiahmedrizvee)
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/mazharul-alam-rafi)
- **Portfolio:** [Visit My Portfolio](https://rafiahmedrizvee.netlify.app/)

---

⭐ If you find this project useful, consider giving the repository a star.

**Built with ❤️ to explore the potential of AI in agriculture.**
