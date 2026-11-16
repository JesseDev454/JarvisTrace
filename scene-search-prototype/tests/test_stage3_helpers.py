"""Unit checks that require no GPU, model weights, database or network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_caption_frames import safe_name, DEFAULT_MODEL, PROMPT

def test_model_slug_is_safe():
    assert safe_name('HuggingFaceTB/SmolVLM2-500M-Video-Instruct') == 'HuggingFaceTB_SmolVLM2-500M-Video-Instruct'

def test_unsafe_filename_characters():
    assert safe_name('../A \\ ?') == 'A'

def test_prompt_warns_against_hallucinations():
    assert 'Do not invent' in PROMPT
    assert DEFAULT_MODEL.startswith('HuggingFaceTB/')
