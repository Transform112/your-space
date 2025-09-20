<!-- # Gemma-3 270M Career Guidance Bot

## Model Information
- Model: Gemma-3 270M fine-tuned for career guidance
- Training Dataset: 1600+ career guidance examples + off-topic handling
- Training Platform: Google Colab with T4 GPU
- Training Steps: 1000 steps with LoRA fine-tuning
- Trained on: 2025-09-20 18:52:54

## Usage Instructions

1. Extract this folder to your local project: gemma_career_bot/models/
2. Install required dependencies:
   ```
   pip install unsloth transformers torch
   ```

3. Load and use the model:
   ```python
   from unsloth import FastLanguageModel
   import torch
   
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="./gemma_career_final",
       max_seq_length=2048,
       dtype=torch.float16,
       load_in_4bit=True,
   )
   
   FastLanguageModel.for_inference(model)
   
   # Generate response
   prompt = f"""<start_of_turn>user
   How do I become a data scientist?<end_of_turn>
   <start_of_turn>model
   """
   
   inputs = tokenizer([prompt], return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=256)
   response = tokenizer.batch_decode(outputs)[0]
   print(response)
   ```

## Model Capabilities
- Career planning and advice
- Interview preparation tips
- Skill development guidance
- Industry insights and trends
- Educational pathway recommendations
- Professional development strategies
- Graceful handling of off-topic queries

 -->

---
title: Career Guidance AI Assistant
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI-powered career guidance chatbot trained on professional development data
---

# 🚀 Career Guidance AI Assistant

A specialized AI chatbot designed to provide personalized career guidance and professional development advice. This assistant is powered by a fine-tuned Gemma-2-2b model trained specifically on career-related conversations.

## 🎯 Features

- **Career Planning**: Get advice on career paths and professional growth
- **Interview Preparation**: Tips and strategies for job interviews
- **Skill Development**: Recommendations for building relevant skills
- **Job Search**: Strategies for finding and applying to jobs
- **Career Transitions**: Guidance for changing careers or industries
- **Professional Development**: Advice on advancing in your current role

## 🤖 Model Details

- **Base Model**: Gemma-2-2b-it
- **Fine-tuning**: LoRA fine-tuning with career guidance dataset
- **Training Data**: 1600+ career-related Q&A pairs + off-topic handling
- **Specialization**: Career counseling, interview prep, skill development

## 💬 How to Use

Simply type your career-related question in the chat interface. The AI assistant will provide detailed, helpful responses based on its training in professional development and career guidance.

## 🔒 Privacy

Your conversations are not stored or logged. Each session is independent and private.

## 📝 Disclaimer

This AI provides general career guidance based on its training data. For specific situations or legal advice, please consult with qualified career professionals or counselors.
