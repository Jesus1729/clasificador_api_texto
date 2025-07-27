# Dependencias
import sys
import pathlib

# Agregamos ruta al directorio raíz del proyecto
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from src.preprocesamiento import clean_text

def test_removes_punctuation_and_lowercases():
    text = "Hello, World!!!"
    expected = "hello world"
    assert clean_text(text) == expected

def test_removes_stopwords():
    text = "This is a simple test"
    expected = "simple test"
    assert clean_text(text) == expected

def test_empty_string():
    text = ""
    expected = ""
    assert clean_text(text) == expected
