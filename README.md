# 🚀 Overview

**Lawverse** is an advanced Retrieval-Augmented Generation (RAG) application designed to provide accurate, context-aware legal assistance. By leveraging the speed and reasoning capabilities of **Google's Gemini 2.5 Flash** model and **LangChain's** orchestration, Lawverse allows users to interact with complex legal documentation through a conversational interface.

## 🚀 Key Features

## 

-   **RAG Architecture:** Retrievies relevant legal context before generating answers to ensure high factual accuracy and reduced hallucinations.
    
-   **Conversational Memory:** Utilizes LangChain to maintain context across multiple turns of conversation, mimicking a real legal consultation.
    
-   **High-Performance LLM:** Powered by `google/gemini-2.5-flash` for rapid inference and strong reasoning capabilities.
    
-   **User Authentication:** Secure Sign-Up and Sign-In functionality to manage user sessions.
    
-   **Dockerized Deployment:** Fully containerized application ensuring consistency across development and production environments.

## 🛠️ Technology Stack

-   **Frontend:** HTML5, CSS3, JavaScript
    
-   **Backend:** Python, Flask
    
-   **LLM:** Google Gemini 2.5 Flash
    
-   **Orchestration:** LangChain
    
-   **Deployment:** Docker
    

## ⚙️ Installation & Local Setup

### Prerequisites

## 

-   Git
    
-   Docker (Recommended) or Python 3.9+
    
-   Google AI Studio API Key
    

### Option 1: Running with Docker (Recommended)

1.  **Clone the Repository**
    
        git clone https://github.com/mdmohsin212/Lawverse.git
        cd Lawverse
        
    
2.  **Configure Environment Variables** Create a `.env` file in the root directory:
    
        GOOGLE_API_KEY=your_actual_google_api_key
        SECRET_KEY=your_random_secret_key_for_flask_sessions
        
    
3.  **Build and Run**
    
        docker build -t lawverse .
        docker run -p 7860:7860 --env-file .env lawverse
        
    
    Access the app at `http://localhost:7860`.
    

### Option 2: Manual Installation


1.  **Clone and Navigate**
    
        git clone https://github.com/mdmohsin212/Lawverse.git
        cd Lawverse
        
    
2.  **Create Virtual Environment**
    
        python -m venv venv
        source venv/bin/activate
        
    
3.  **Install Dependencies**
    
        pip install -r requirements.txt
        
    
4.  **Run Application**
    
        python app.py
        
    

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the [MIT License](https://github.com/mdmohsin212/Lawverse/blob/main/LICENCE).