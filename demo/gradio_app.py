"""
Gradio Demo Application for Earnings Call Intelligence System
===============================================================

Interactive web demo for earnings call analysis.

Usage:
    python demo/gradio_app.py
    
Opens: http://localhost:7860

For HuggingFace Spaces deployment:
    Copy this file to your Space's app.py
"""

import os
import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_PATH = os.environ.get("MODEL_PATH", "training/outputs/earnings-intelligence-v1")

SYSTEM_PROMPT = """You are an expert financial analyst specializing in earnings call analysis. 
Analyze the provided transcript and provide insights about company performance, beat/miss predictions, 
and key metrics. Be concise but informative."""

def load_model():
    """Load model for inference."""
    try:
        if not Path(MODEL_PATH).exists():
            MODEL_PATH = "unsloth/Llama-3.2-3B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )
        
        return pipe, tokenizer
    except Exception as e:
        print(f"Warning: Could not load fine-tuned model: {e}")
        print("Using base model...")
        
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct",
            load_in_4bit=True,
            device_map="auto",
        )
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
        ), tokenizer

pipe, tokenizer = load_model()

def generate_response(messages, max_new_tokens=512):
    """Generate response from model."""
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        
        inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_earnings(transcript, company_quarter, analysis_type):
    """Main analysis function."""
    if not transcript.strip():
        return "Please enter an earnings call transcript."
    
    if not company_quarter.strip():
        company_quarter = "Company Q1 2024"
    
    if analysis_type == "Beat/Miss Prediction":
        prompt = f"""Analyze this earnings call transcript for {company_quarter} and predict:

1. Did they BEAT or MISS analyst estimates for EPS and Revenue?
2. What is your confidence level (0-100%)?
3. What are the key metrics mentioned?

Transcript:
{transcript[:2500]}

Respond in this format:
### EPS: [BEAT/MISS/MEET] | Confidence: [X%]
### Revenue: [BEAT/MISS/MEET] | Confidence: [X%]
### Key Metrics: [list]
### Analysis: [2-3 sentences]"""

    elif analysis_type == "Financial Q&A":
        prompt = f"""You are a financial analyst. Answer questions about this earnings call transcript for {company_quarter}.

Transcript:
{transcript[:2500]}

Provide detailed answers about:
- Revenue performance and growth
- Profitability metrics
- Forward guidance
- Key highlights and concerns

Be specific and reference numbers from the transcript."""

    elif analysis_type == "Metric Extraction":
        prompt = f"""Extract all financial metrics from this earnings call transcript for {company_quarter}.

Transcript:
{transcript[:2500]}

Extract and format:
### Revenue: [amount] ([YoY change])
### EPS: [amount] ([YoY change])
### Gross Margin: [percentage]
### Operating Margin: [percentage]
### Revenue Guidance: [amount/percentage]
### EPS Guidance: [amount]"""

    else:
        prompt = f"""You are an expert financial analyst. Perform a comprehensive analysis of this earnings call transcript for {company_quarter}.

Transcript:
{transcript[:2500]}

Provide:
1. Executive Summary (2-3 sentences)
2. Beat/Miss Analysis for EPS and Revenue
3. Key Highlights (bullet points)
4. Key Concerns (bullet points)
5. Forward Guidance Summary
6. Stock Outlook"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    return generate_response(messages)

def create_demo():
    """Create Gradio interface."""
    with gr.Blocks(
        title="Earnings Call Intelligence",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        )
    ) as demo:
        gr.Markdown("""
        # 📈 Earnings Call Intelligence System
        
        ### AI-Powered Financial Analysis
        
        ---
        
        **Features:**
        - 🎯 Beat/Miss Prediction - Predict earnings outcomes
        - 💬 Financial Q&A - Ask questions about earnings
        - 📊 Metric Extraction - Extract key financial metrics
        - 📝 Full Analysis - Comprehensive earnings report
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                company_input = gr.Textbox(
                    label="Company & Quarter",
                    placeholder="e.g., Apple Q1 2024",
                    value="Apple Q4 2023"
                )
                
                analysis_type = gr.Dropdown(
                    label="Analysis Type",
                    choices=[
                        "Beat/Miss Prediction",
                        "Financial Q&A",
                        "Metric Extraction",
                        "Full Analysis"
                    ],
                    value="Beat/Miss Prediction"
                )
                
                transcript_input = gr.Textbox(
                    label="Earnings Call Transcript",
                    placeholder="Paste the earnings call transcript here...",
                    lines=15,
                    info="For best results, paste 500+ words"
                )
                
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Analysis Results",
                    lines=20,
                    interactive=False,
                    show_copy_button=True
                )
        
        gr.Markdown("""
        ---
        
        ### 📝 Example Transcripts
        
        Try these sample companies:
        
        | Company | Quarter | Expected |
        |---------|---------|----------|
        | Apple | Q1 2024 | Beat - $119.6B revenue, EPS $2.18 |
        | Tesla | Q3 2023 | Beat - $23.4B revenue, EPS $0.66 |
        | Meta | Q4 2023 | Beat - $40.1B revenue, EPS $5.33 |
        | Intel | Q4 2023 | Miss - Revenue $15.4B (down 10%) |
        
        ---
        
        ### ⚠️ Disclaimer
        
        This tool provides predictions based on historical earnings data and is **NOT financial advice**.
        Always consult professional advisors for investment decisions.
        
        ---
        
        *Built with 🤗 HuggingFace Transformers & Gradio*
        """)
        
        analyze_btn.click(
            fn=analyze_earnings,
            inputs=[transcript_input, company_input, analysis_type],
            outputs=output
        )
        
        examples = gr.Examples(
            examples=[
                [
                    "Apple Q4 2023",
                    "Beat/Miss Prediction",
                    "Apple today reported fiscal Q1 2024 revenue of $119.6 billion, up 2% year over year. EPS of $2.18 compared to analyst estimates of $2.10. Services revenue reached a new all-time high of $22.3 billion, up 16% year over year. iPhone revenue was $69.1 billion. Mac revenue came in at $7.8 billion. For Q2, we expect revenue to be between $110 billion and $114 billion."
                ],
                [
                    "Tesla Q3 2023",
                    "Beat/Miss Prediction",
                    "Tesla reported Q3 2023 revenue of $23.4 billion, up 9% year over year, surpassing analyst expectations of $23.2 billion. EPS of $0.66 exceeded estimates of $0.59. Automotive gross margin was 17.9%, down from 26% a year ago. We delivered 435,059 vehicles in the quarter, a new record."
                ],
                [
                    "Intel Q4 2023",
                    "Full Analysis",
                    "Intel Q4 2023 revenue was $15.4 billion, down 10% year over year, missing expectations. Data center revenue was $4.0 billion, down 7%. Client computing revenue was $8.8 billion, down 8%. EPS of $0.54 missed estimates of $0.64. We expect Q1 2024 revenue between $12.2 billion and $13.2 billion."
                ]
            ],
            inputs=[company_input, analysis_type, transcript_input],
            label="Try Example Transcripts"
        )
    
    return demo

def main():
    """Launch demo."""
    print("=" * 60)
    print("EARNINGS CALL INTELLIGENCE - GRADIO DEMO")
    print("=" * 60)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("\nStarting demo...")
    print("Open: http://localhost:7860")
    print("=" * 60)
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        max_threads=2
    )

if __name__ == "__main__":
    main()
