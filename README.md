# TTS_FINETUNE_ENGLISH

Overview
This project implements a Text-to-Speech (TTS) system using the SpeechT5 model fine-tuned on the LJ Speech dataset. The goal is to generate natural-sounding speech from textual input, leveraging state-of-the-art machine learning techniques and for english technical words.
## Introduction

Text-to-Speech (TTS) technology converts written text into spoken words, enabling computers to generate human-like speech. This technology has gained significant traction due to its diverse applications across various fields, including:

- **Accessibility**: TTS assists visually impaired individuals by reading text aloud, enhancing their ability to interact with digital content.
- **Education**: It facilitates language learning and literacy development by providing auditory feedback and pronunciation guidance.
- **Entertainment**: TTS is utilized in audiobooks, video games, and virtual assistants, enhancing user experience through engaging and immersive interactions.
- **Customer Service**: Automated voice response systems use TTS to provide information and support, improving efficiency and accessibility.

### Importance of Fine-Tuning

While pre-trained models like SpeechT5 offer a robust starting point for TTS applications, fine-tuning on specific datasets is crucial for several reasons:

1. **Personalization**: Fine-tuning allows the model to adapt to specific voices, accents, and speaking styles, resulting in more natural and relatable speech output.

2. **Domain Adaptation**: Different applications may require distinct speech characteristics. Fine-tuning enables the model to better handle domain-specific terminology and contexts.

3. **Quality Improvement**: By training on curated datasets like LJ Speech, the model learns to generate clearer, more expressive, and emotionally nuanced speech, enhancing overall user satisfaction.

4. **Performance Optimization**: Fine-tuning can lead to better alignment of text and audio, reducing issues like mispronunciations or unnatural intonations, thus improving the model’s reliability in real-world applications.

In this project, we leverage the LJ Speech dataset to fine-tune the SpeechT5 model, aiming to create a TTS system that excels in delivering high-quality, lifelike speech tailored to specific user needs.


## Methodology

This section outlines the detailed steps taken for model selection, dataset preparation, and fine-tuning of the Text-to-Speech (TTS) system using the SpeechT5 model.

### 1. Model Selection

**Model Choice: SpeechT5**
- SpeechT5 is a transformer-based model designed for TTS tasks. It provides a balance of performance and efficiency, making it suitable for generating high-quality speech.
- We selected SpeechT5 due to its ability to handle various speech generation tasks, including text-to-speech synthesis, making it versatile for our project.

### 2. Dataset Preparation

**Dataset: LJ Speech**
- The LJ Speech dataset comprises approximately 13,100 audio clips of a single speaker reading passages from various texts. It includes:
  - **Audio Files**: WAV format audio samples.
  - **Transcripts**: Text files containing the spoken text for each audio sample.

**Preparation Steps:**
1. **Data Download**: Download the LJ Speech dataset from [keithito.com](https://keithito.com/LJ-Speech-Dataset/) 
   
2. **Preprocessing**:
   - **Audio Processing**:
     - Convert all audio files to a consistent format (16000 Hz sample rate, mono channel).
     - Normalize audio levels to ensure consistent volume across samples.
   - **Text Cleaning**:
     - Remove any extraneous characters or formatting from the transcripts to ensure clean input for the model.
     - Optionally, apply phonetic transcriptions for improved pronunciation.

3. **Alignment**:
   - Generate a mapping between audio files and their corresponding text transcripts, ensuring that each audio clip can be paired with its correct spoken text.

4. **Train-Validation Split**:
   - Split the dataset into training and validation sets (e.g., 90% training, 10% validation) to evaluate model performance during and after fine-tuning.

### 3. Fine-Tuning

**Fine-Tuning Process:**
1. **Environment Setup**:
   - Ensure that all necessary libraries and dependencies (e.g., PyTorch, Transformers, NumPy) are installed as specified in the `requirements.txt` file.

2. **Fine-Tuning Script**:
     - Model loading: Load the pre-trained SpeechT5 model.
     - Data loading: Use a data loader to read audio and text pairs for training.
     - Training loop: Implement a loop that iterates over the training dataset for a specified number of epochs, updating model weights based on the loss calculated from predictions.

3. **Hyperparameter Configuration**:
   - Set hyperparameters such as learning rate, batch size, and the number of epochs. Commonly used values might include:
     - Learning Rate: 1e-5
     - Batch Size: 16
     - Epochs: 10/11

4. **Monitoring**:
   - Monitor training loss and validation metrics during the fine-tuning process to prevent overfitting. Use techniques like early stopping if validation loss starts to increase.

5. **Model Saving**:
   - After fine-tuning, save the trained model and any associated artifacts (e.g., tokenizer) for later use in generating speech.

### 4. Evaluation

**Post-Fine-Tuning Evaluation**:
- After fine-tuning, evaluate the model using the validation dataset. Metrics for evaluation may include:
  - Mean Opinion Score (MOS): A subjective score based on human evaluations of audio quality.
  - Alignment and accuracy: Check how well the generated speech aligns with the input text.

By following this methodology, we can ensure that the SpeechT5 model is effectively fine-tuned on the LJ Speech dataset, resulting in a robust TTS system capable of producing high-quality speech outputs.

MODEL LINK:https://huggingface.co/lithish2602/TTS_FINETUNE_ENGLISH


RESULTS:

Results
In this section, we present the results of our Text-to-Speech (TTS) system, including both objective and subjective evaluations. We tested the model's performance on two types of speech: English technical speecH

1. Objective Evaluations
Metrics Used: Mean Opinion Score (MOS): A numerical measure of perceived audio quality, usually rated on a scale from 1 (poor) to 5 (excellent).
English Technical Speech
Subjective evaluations were conducted through listener studies, where participants rated the audio samples generated by the model. The evaluations focused on clarity, naturalness, and overall satisfaction.

English Technical Speech Feedback
Clarity: Most listeners appreciated the clarity of the speech, especially in technical contexts.
Naturalness: While the speech was generally natural, some participants felt that the intonation could be improved to sound more conversational.
Comments: “Very clear for technical topics, but sometimes feels robotic.”

THANK YOU
