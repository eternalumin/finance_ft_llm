"""
Inference Module for Earnings Call Intelligence System
=======================================================

Provides easy-to-use interface for predictions.

Usage:
    from inference.predict import EarningsIntelligence
    
    model = EarningsIntelligence()
    result = model.analyze(transcript="...", company="Apple Q1 2024")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    prediction: str
    confidence: float
    eps_status: str
    revenue_status: str
    key_metrics: Dict[str, Any]
    full_response: str

class EarningsIntelligence:
    """
    Main inference class for earnings call analysis.
    """
    
    def __init__(
        self,
        model_path: str = "training/outputs/earnings-intelligence-v1",
        device: Optional[str] = None
    ):
        """
        Initialize the model.
        
        Args:
            model_path: Path to trained model directory
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not Path(self.model_path).exists():
            logger.warning(f"Model not found at {self.model_path}")
            logger.info("Using base model for demonstration")
            self.model_path = "unsloth/Llama-3.2-3B-Instruct"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def analyze(
        self,
        transcript: str,
        company: str = "Company",
        analysis_type: str = "full"
    ) -> AnalysisResult:
        """
        Analyze an earnings call transcript.
        
        Args:
            transcript: Earnings call transcript text
            company: Company name and quarter (e.g., "Apple Q1 2024")
            analysis_type: Type of analysis ("full", "beat_miss", "qa", "metrics")
            
        Returns:
            AnalysisResult with predictions and metrics
        """
        if analysis_type == "beat_miss":
            prompt = self._create_beat_miss_prompt(transcript, company)
        elif analysis_type == "qa":
            prompt = self._create_qa_prompt(transcript, company)
        elif analysis_type == "metrics":
            prompt = self._create_metrics_prompt(transcript, company)
        else:
            prompt = self._create_full_prompt(transcript, company)
        
        response = self._generate(prompt)
        
        return self._parse_response(response, analysis_type)
    
    def _create_beat_miss_prompt(self, transcript: str, company: str) -> str:
        """Create beat/miss prediction prompt."""
        return f"""You are a financial analyst. Analyze this earnings call transcript for {company} and predict:

1. Did they BEAT or MISS analyst estimates for EPS and Revenue?
2. What is your confidence level (0-100%)?
3. What are the key metrics mentioned?

Transcript:
{transcript[:2000]}

Respond in this format:
### EPS: [BEAT/MISS/MEET] | Confidence: [X%]
### Revenue: [BEAT/MISS/MEET] | Confidence: [X%]
### Key Metrics: [list]
### Analysis: [2-3 sentences]"""

    def _create_qa_prompt(self, transcript: str, company: str) -> str:
        """Create Q&A prompt."""
        return f"""You are a financial analyst. Answer questions about this earnings call transcript for {company}.

Transcript:
{transcript[:2000]}

Answer the following types of questions:
- What was the revenue growth?
- How did margins perform?
- What is the forward guidance?
- What are the main highlights?

Provide detailed, factual answers based on the transcript."""

    def _create_metrics_prompt(self, transcript: str, company: str) -> str:
        """Create metrics extraction prompt."""
        return f"""Extract all financial metrics from this earnings call transcript for {company}.

Transcript:
{transcript[:2000]}

Extract and format:
### Revenue: [amount] ([YoY change])
### EPS: [amount] ([YoY change])
### Gross Margin: [percentage]
### Operating Margin: [percentage]
### Revenue Guidance: [amount/percentage]
### EPS Guidance: [amount]"""

    def _create_full_prompt(self, transcript: str, company: str) -> str:
        """Create full analysis prompt."""
        return f"""You are an expert financial analyst. Perform a comprehensive analysis of this earnings call transcript for {company}.

Transcript:
{transcript[:2000]}

Provide:
1. Executive Summary (2-3 sentences)
2. Beat/Miss Analysis for EPS and Revenue
3. Key Highlights (bullet points)
4. Key Concerns (bullet points)
5. Forward Guidance Summary
6. Stock Outlook (based on analysis)

Format your response clearly with headers."""

    def _generate(self, prompt: str) -> str:
        """Generate response from model."""
        try:
            response = self.pipe(prompt, return_full_text=False)[0]["generated_text"]
            return response
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "Error: Could not generate response"

    def _parse_response(
        self,
        response: str,
        analysis_type: str
    ) -> AnalysisResult:
        """Parse model response into structured result."""
        response_upper = response.upper()
        
        eps_status = "UNKNOWN"
        revenue_status = "UNKNOWN"
        confidence = 0.5
        
        if "EPS:" in response_upper:
            eps_line = response.split("EPS:")[-1].split("\n")[0]
            if "BEAT" in eps_line.upper():
                eps_status = "BEAT"
            elif "MISS" in eps_line.upper():
                eps_status = "MISS"
            elif "MEET" in eps_line.upper():
                eps_status = "MEET"
        
        if "REVENUE:" in response_upper:
            rev_line = response.split("REVENUE:")[-1].split("\n")[0]
            if "BEAT" in rev_line.upper():
                revenue_status = "BEAT"
            elif "MISS" in rev_line.upper():
                revenue_status = "MISS"
            elif "MEET" in rev_line.upper():
                revenue_status = "MEET"
        
        if "CONFIDENCE" in response_upper:
            import re
            conf_match = re.search(r'(\d+)%', response)
            if conf_match:
                confidence = int(conf_match.group(1)) / 100
        
        prediction = "BEAT" if eps_status == "BEAT" and revenue_status == "BEAT" else \
                      "MISS" if eps_status == "MISS" or revenue_status == "MISS" else \
                      "PARTIAL"
        
        return AnalysisResult(
            prediction=prediction,
            confidence=confidence,
            eps_status=eps_status,
            revenue_status=revenue_status,
            key_metrics={},
            full_response=response
        )

    def batch_analyze(
        self,
        transcripts: List[Dict[str, str]]
    ) -> List[AnalysisResult]:
        """
        Analyze multiple transcripts.
        
        Args:
            transcripts: List of dicts with 'transcript' and 'company' keys
            
        Returns:
            List of AnalysisResult
        """
        results = []
        for item in transcripts:
            result = self.analyze(
                transcript=item["transcript"],
                company=item.get("company", "Unknown")
            )
            results.append(result)
        return results

def main():
    """Demo usage."""
    print("Loading model...")
    model = EarningsIntelligence()
    
    sample_transcript = """
    Apple today reported fiscal Q1 2024 revenue of $119.6 billion, up 2% year over year.
    EPS of $2.18 compared to analyst estimates of $2.10. Services revenue reached a new 
    all-time high of $22.3 billion, up 16% year over year.
    """
    
    print("\nAnalyzing sample transcript...")
    result = model.analyze(
        transcript=sample_transcript,
        company="Apple Q1 2024",
        analysis_type="beat_miss"
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS RESULT")
    print("=" * 60)
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"EPS Status: {result.eps_status}")
    print(f"Revenue Status: {result.revenue_status}")
    print("=" * 60)

if __name__ == "__main__":
    main()
