# Psychology Benchmark for Evaluating LLM Performance
# Overview
PsychDialogBench is a specialized benchmark designed to evaluate the ability of Large Language Models (LLMs) to analyze psychological support dialogues in a manner aligned with expert human annotations. This project addresses the critical gap in existing NLP benchmarks, which often lack the depth, clinical relevance, and expert validation required for authentic psychological counseling scenarios.

The benchmark is built upon a corpus of over 5,000 authentic, multi-turn dialogues scraped from a professional psychological assistance forum ([B17.ru](https://www.b17.ru/)). A subset of 758 dialogues has been meticulously annotated by psychology-trained experts using a comprehensive, multi-level schema developed through in-depth interviews with practicing psychologists.

# Key Features
Our benchmark introduces a multi-faceted annotation schema that goes beyond basic sentiment to provide a clinically-grounded evaluation framework. This includes emotional segmentation with fine-grained labeling of client utterances into 7 emotion categories (Joy, Sadness, Anger, Fear, Disgust, Surprise, Contempt), psychologist response quality assessment using 5-point scale ratings across four criteria (empathy, ethicality, question productivity, and recommendation usefulness), thematic tagging across 15+ semantic categories covering the spectrum of psychological issues, and final psychological state assessment with holistic ratings on depression level, need for professional help, confusion, thought disorganization, and suicide risk.

The annotations form an expert-aligned gold standard, performed by psychology majors and trained psychologists to ensure clinical relevance and reliability. All dialogues are provided in a structured JSON format with explicit role labeling (Client/Psychologist), timestamps, and metadata for easy integration into research pipelines. The repository includes a complete evaluation pipeline with tools for benchmarking LLM performance across all annotation tasks, featuring metrics for segmentation accuracy (F1, IoU), rating agreement (MAE, accuracy), and thematic classification.

# Repository Contents
- emotion_segmentation_benchmark.py - Core benchmarking script for evaluating LLMs on the emotional segmentation task.
- Segmentation_match.py - Utility for comparing segmentation annotations between models or between model and human annotators.
- output_markup.zip - Contains output_markup.json, the fully annotated dataset of 758 dialogues.

# Dataset Details
The annotated corpus (output_markup.json) includes 758 complete dialogues comprising 17,434 messages. Each dialogue is structured in JSON format with fields for dialogue ID and metadata, message sequences with speaker role (client/psychologist), emotional segmentation tags within client messages, psychologist response quality scores, thematic tags per dialogue, and final psychological state assessments.

The dataset exhibits broad thematic diversity, with family & partner relationships representing 21.6% of dialogues, personal crises & self-esteem at 15.5%, depression & anxiety disorders at 8.0%, and social adaptation & conflicts at 8.5%, plus 12+ other categories. Analysis of emotional tone in client messages reveals a distribution dominated by neutral expressions (52.3%), followed by sadness (22.4%), fear (8.1%), anger (4.9%), joy (4.9%), contempt (3.7%), disgust (2.2%), and surprise (1.6%). This distribution aligns with patterns observed in comparable mental health datasets, affirming the benchmark's representativeness for psycholinguistic analysis.

# Usage
To benchmark LLMs on emotional segmentation, use **emotion_segmentation_benchmark.py** with the command:
```
python emotion_segmentation_benchmark.py
--model <your_model>
--input_data output_markup.json
--output_results <results_path>
```
The script implements the zero-shot prompting setup described in the paper, with strict format validation and alignment checks to ensure reliable evaluation.

For comparing annotations between models or between model and human annotators, **Segmentation_match.py** provides utility functions to compute agreement metrics. You can import and use it with: 
```
from Segmentation_match import calculate_segmentation_metrics
```
followed by 
```
metrics = calculate_segmentation_metrics(gold_annotations, model_predictions)
```

Data exploration is straightforward using Python's json module: 
```
import json
```
then 
```
with open('output_markup.json', 'r', encoding='utf-8') as f:
data = json.load(f)
```
From there you can access dialogue metadata, message sequences, annotations, and assessment scores for analysis and visualization.

# Evaluation Metrics
The benchmark evaluates LLMs across four tasks with specific, clinically-informed metrics. For emotional segmentation, we calculate weighted F1, macro-F1, and intersection over union (IoU) per emotion category, using character-level alignment for precision. Thematic tagging performance is assessed through micro F1, Jaccard similarity, and exact match rates for full tag set reproduction.

For psychological state assessment, we employ accuracy, mean absolute error (MAE), and within ±1 accuracy to gauge how closely model ratings match expert judgments on critical dimensions like depression level and suicide risk. Message quality evaluation uses per-criterion MAE and accuracy, plus exact match rate per message to measure consistency in assessing therapist responses across empathy, ethicality, productivity, and usefulness dimensions.

# Citation
If you use PsychDialogBench in your research, please cite the associated paper:
```
@article{yourcitation,
  title={Psychology Benchmark for Evaluating LLM Performance},
  author={Beliavskaia V. et.al.},
  year={2026}
}
```

# Acknowledgments
This research was supported by HSE University's HPC facilities and the HSE-VK School of Engineering and Mathematics. We thank VK Cloud for providing computational resources for web-application deployment, Yandex for access to the YandexGPT 5 Pro API, and Associate Professor A. V. Vecherin of the Faculty of Social Sciences, Department of Psychology, HSE University, for his assistance in conducting the research.
